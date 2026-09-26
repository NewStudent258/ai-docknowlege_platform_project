"""LightRAG 第三路检索器单元测试（ADR-025）

覆盖：
- content → (doc_id, chunk_index) 精确映射
- 无图谱 / 构建中 → 空结果降级（不抛异常）
- 无法映射的 chunk 被跳过
- LightRAG 返回异常结构时不炸
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.retriever import RetrievalOutput

pytestmark = pytest.mark.unit

# LightRAGStore 在检索器内被 patch，store.run_on 的真实语义是
# 「在实例专属 loop 上执行协程」，单测里用 asyncio.run 等价替代
_PATCH_BASE = "app.rag.lightrag_retriever"


def _make_store(aquery_return):
    """构造 mock store，其 run_on 真正执行传入的协程"""
    handle = MagicMock()
    handle.rag.aquery_data = AsyncMock(return_value=aquery_return)

    def _run_on(_handle, coro):
        return asyncio.run(coro)

    store = MagicMock()
    store.get_handle = AsyncMock(return_value=handle)
    store.run_on.side_effect = _run_on
    return store


async def _search(store, retriever, question="问题", kb_id=1, has_graph=True, building=False):
    """在 patch 环境下执行检索

    必须是 async：若在同步函数里 `return retriever.search(...)`，
    with 块会在协程被 await 之前退出，补丁提前失效。
    """
    with patch(f"{_PATCH_BASE}.LightRAGStore") as cls, patch(
        f"{_PATCH_BASE}.has_graph_data", return_value=has_graph
    ), patch(f"{_PATCH_BASE}.is_building", return_value=building):
        cls.get.return_value = store
        return await retriever.search(question, kb_id=kb_id)


def _make_retriever(content_index):
    from app.rag.lightrag_retriever import LightRAGRetriever

    r = LightRAGRetriever()
    r._get_content_index = AsyncMock(return_value=content_index)
    return r


class TestContentMapping:
    """content 精确映射（ADR-025 的核心设计）"""

    @pytest.mark.asyncio
    async def test_正常映射回_doc_id_与_chunk_index(self):
        index = {
            "公司薪酬由基本工资构成。": {
                "doc_id": 7,
                "chunk_index": 0,
                "doc_name": "薪酬管理制度.txt",
                "metadata": {"section_title": "第二章 薪酬结构", "page": 1},
            }
        }
        r = _make_retriever(index)
        store = _make_store({"data": {"chunks": [{"content": "公司薪酬由基本工资构成。"}]}})

        out = await _search(store, r)

        assert isinstance(out, RetrievalOutput)
        assert len(out.results) == 1
        hit = out.results[0]
        assert hit.doc_id == 7
        assert hit.chunk_index == 0
        assert hit.doc_name == "薪酬管理制度.txt"
        assert hit.section_title == "第二章 薪酬结构"
        assert hit.page == 1
        # 无向量，由粗排按中性分处理
        assert hit.embedding is None

    @pytest.mark.asyncio
    async def test_无法映射的_chunk_被跳过(self):
        index = {
            "能匹配上的内容": {
                "doc_id": 1,
                "chunk_index": 2,
                "doc_name": "a.txt",
                "metadata": {},
            }
        }
        r = _make_retriever(index)
        store = _make_store(
            {
                "data": {
                    "chunks": [
                        {"content": "完全对不上的内容"},
                        {"content": "能匹配上的内容"},
                    ]
                }
            }
        )

        out = await _search(store, r)

        assert len(out.results) == 1
        assert out.results[0].doc_id == 1
        assert out.stats["lightrag_raw_count"] == 2
        assert out.stats["lightrag_count"] == 1


class TestDegradation:
    """降级路径：任一前置条件不满足都返回空结果而非抛异常"""

    @pytest.mark.asyncio
    async def test_该_KB_无图谱时返回空(self):
        r = _make_retriever({})
        out = await _search(_make_store({}), r, kb_id=99, has_graph=False)

        assert out.results == []
        assert out.stats.get("lightrag_skipped") is True

    @pytest.mark.asyncio
    async def test_图谱构建中时返回空(self):
        r = _make_retriever({})
        out = await _search(_make_store({}), r, kb_id=99, has_graph=True, building=True)

        assert out.results == []
        assert out.stats.get("lightrag_skipped") is True

    @pytest.mark.asyncio
    async def test_LightRAG_返回空_chunks_时返回空(self):
        r = _make_retriever({})
        out = await _search(_make_store({"data": {"chunks": []}}), r)

        assert out.results == []
        assert out.stats["lightrag_count"] == 0

    @pytest.mark.asyncio
    async def test_返回结构缺少_data_键时不炸(self):
        """LightRAG 返回 failure 结构时按空结果处理"""
        r = _make_retriever({})
        out = await _search(_make_store({"status": "failure"}), r)

        assert out.results == []

    @pytest.mark.asyncio
    async def test_返回_None_时不炸(self):
        r = _make_retriever({})
        out = await _search(_make_store(None), r)

        assert out.results == []


class TestInvalidate:
    """索引缓存失效"""

    def test_invalidate_只清除指定_KB(self):
        r = _make_retriever({})
        r._index_cache[1] = ({}, 0.0)
        r._index_cache[2] = ({}, 0.0)

        r.invalidate(1)

        assert 1 not in r._index_cache
        assert 2 in r._index_cache
