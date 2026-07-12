"""LightRAG 第三路检索接入 KnowledgePipeline 的集成测试（ADR-025）

覆盖：
- LIGHTRAG_ENABLED=false → 退回双路，第三路检索器不被调用
- LIGHTRAG_ENABLED=true  → 三路融合
- 第三路检索抛异常 → 降级为双路，问答不中断
- RRF 融合真的收到三路输入
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.knowledge_pipeline import KnowledgePipeline
from app.rag.retriever import RetrievalOutput, RetrievalResult

pytestmark = pytest.mark.unit


def _out(doc_id: int, chunk_index: int, content: str):
    return RetrievalOutput(
        results=[
            RetrievalResult(
                doc_id=doc_id, chunk_index=chunk_index, content=content, score=0.9
            )
        ],
        total=1,
    )


def _mock_db():
    """mock AsyncSession：KB 内至少有 1 份可检索文档"""
    db = MagicMock()
    result = MagicMock()
    result.scalar.return_value = 1
    db.execute = AsyncMock(return_value=result)
    return db


def _make_pipeline(lightrag_output=None, lightrag_side_effect=None):
    """构造注入了三路依赖的管线"""
    vector_out = _out(1, 0, "向量召回内容")
    bm25_out = _out(2, 0, "BM25 召回内容")

    mock_vector = MagicMock()
    mock_vector.search = AsyncMock(return_value=vector_out)

    mock_bm25 = MagicMock()
    mock_bm25.search = AsyncMock(return_value=bm25_out)

    mock_lightrag = MagicMock()
    if lightrag_side_effect is not None:
        mock_lightrag.search = AsyncMock(side_effect=lightrag_side_effect)
    else:
        mock_lightrag.search = AsyncMock(
            return_value=lightrag_output or _out(3, 0, "图谱召回内容")
        )

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(side_effect=lambda q, o: o)
    mock_reranker.name = "MockReranker"

    pipeline = KnowledgePipeline(
        vector_retriever=mock_vector,
        reranker=mock_reranker,
        lightrag_retriever=mock_lightrag,
    )
    pipeline._bm25_retriever = mock_bm25
    return pipeline, mock_lightrag


async def _run(pipeline, question="测试问题"):
    with patch(
        "app.rag.knowledge_pipeline.needs_rewrite", return_value=False
    ), patch(
        "app.rag.knowledge_pipeline.build_prompt"
    ) as mock_prompt:
        mock_prompt.return_value = MagicMock(
            used_chunks=[], system_prompt="", user_prompt="", history_messages=[]
        )
        return await pipeline.execute_knowledge(_mock_db(), question, 1, [])


class TestSwitchOff:
    """LIGHTRAG_ENABLED=false 时完全退回双路"""

    @pytest.mark.asyncio
    async def test_关闭时第三路不被调用(self):
        pipeline, mock_lightrag = _make_pipeline()
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", False):
            await _run(pipeline)

        mock_lightrag.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_关闭时第三路为空由融合层过滤(self):
        """实现始终传三路（保持计时与调用结构统一），关闭时第三路为空，
        由 rrf_fusion 内部的空结果过滤退回双路语义"""
        pipeline, _ = _make_pipeline()
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", False), patch(
            "app.rag.knowledge_pipeline.rrf_fusion"
        ) as mock_fusion:
            mock_fusion.return_value = _out(1, 0, "融合结果")
            await _run(pipeline)

        assert mock_fusion.call_count == 1
        args = mock_fusion.call_args.args
        assert len(args) == 3
        assert args[2].results == []


class TestSwitchOn:
    """LIGHTRAG_ENABLED=true 时三路融合"""

    @pytest.mark.asyncio
    async def test_开启时第三路被调用(self):
        pipeline, mock_lightrag = _make_pipeline()
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", True):
            await _run(pipeline)

        mock_lightrag.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_开启时融合收到三路(self):
        pipeline, _ = _make_pipeline()
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", True), patch(
            "app.rag.knowledge_pipeline.rrf_fusion"
        ) as mock_fusion:
            mock_fusion.return_value = _out(1, 0, "融合结果")
            await _run(pipeline)

        assert len(mock_fusion.call_args.args) == 3

    @pytest.mark.asyncio
    async def test_真实融合保留图谱独有_chunk(self):
        """图谱路径独有（向量与 BM25 都未召回）的 chunk 应出现在融合结果中"""
        pipeline, _ = _make_pipeline(lightrag_output=_out(99, 0, "只有图谱能召回"))
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", True):
            result = await _run(pipeline)

        keys = {(r.doc_id, r.chunk_index) for r in result.reranked_output.results}
        assert (99, 0) in keys


class TestDegradation:
    """第三路异常时降级为双路，不中断问答"""

    @pytest.mark.asyncio
    async def test_第三路抛异常时仍返回结果(self):
        pipeline, _ = _make_pipeline(lightrag_side_effect=RuntimeError("图谱炸了"))
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", True):
            result = await _run(pipeline)

        # 仍然拿到向量/BM25 的结果
        assert len(result.reranked_output.results) >= 1

    @pytest.mark.asyncio
    async def test_第三路异常时融合收到三路但第三路为空(self):
        pipeline, _ = _make_pipeline(lightrag_side_effect=RuntimeError("图谱炸了"))
        with patch("app.rag.knowledge_pipeline.settings.LIGHTRAG_ENABLED", True), patch(
            "app.rag.knowledge_pipeline.rrf_fusion"
        ) as mock_fusion:
            mock_fusion.return_value = _out(1, 0, "融合结果")
            await _run(pipeline)

        args = mock_fusion.call_args.args
        assert len(args) == 3
        assert args[2].results == []  # 降级后的空第三路
