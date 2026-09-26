"""LightRAG 第三路检索器 — 与 VectorRetriever / BM25Retriever 签名对齐（ADR-025）

检索链路：图谱混合检索（mix = 实体/关系 + 向量）→ 结构化 chunks → 映射回 (doc_id, chunk_index)

**为什么按 content 映射而不是解析 id**：
LightRAG `aquery_data` 返回的每条 chunk 只有
`{reference_id, content, file_path, chunk_id}`，其中 chunk_id 是 LightRAG 内部
哈希生成的（不受传入 ids 控制），也不含 full_doc_id。而插入时我们传的就是
MySQL 里 chunk 的原文，因此按 content 精确反查最可靠。

索引来自 MySQL `chunks` 表，按 KB 缓存（TTL），避免每次检索都全表拉取。
"""

import asyncio
import logging
import time

from sqlalchemy import select

from app.config import settings
from app.core.database import async_session
from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.lightrag_store import LightRAGStore, has_graph_data, is_building
from app.rag.retriever import RetrievalOutput, RetrievalResult

logger = logging.getLogger(__name__)

# content → (doc_id, chunk_index, metadata) 索引的缓存时长（秒）
_INDEX_TTL = 60


class LightRAGRetriever:
    """LightRAG 图谱检索器"""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory or async_session
        # kb_id -> (content_index, built_at)
        self._index_cache: dict[int, tuple[dict[str, dict], float]] = {}
        self._cache_lock = asyncio.Lock()

    # ── 内容索引 ──

    async def _get_content_index(self, kb_id: int) -> dict[str, dict]:
        """取该 KB 的 content → chunk 元信息索引（带 TTL 缓存）"""
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._index_cache.get(kb_id)
            if cached and (now - cached[1]) < _INDEX_TTL:
                return cached[0]

        index: dict[str, dict] = {}
        async with self._session_factory() as db:
            # join documents 取文件名：LightRAG 结果本身不带文档名，
            # 而 prompt_builder 会用 doc_name 标注上下文来源
            rows = await db.execute(
                select(
                    Chunk.doc_id,
                    Chunk.chunk_index,
                    Chunk.content,
                    Chunk.metadata_,
                    Document.filename,
                )
                .join(Document, Document.id == Chunk.doc_id)
                .where(Chunk.kb_id == kb_id)
            )
            for doc_id, chunk_index, content, meta, filename in rows.all():
                index[content] = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "doc_name": filename or "",
                    "metadata": meta or {},
                }

        async with self._cache_lock:
            self._index_cache[kb_id] = (index, now)
        return index

    def invalidate(self, kb_id: int) -> None:
        """清除内容索引缓存（文档增删后调用）"""
        self._index_cache.pop(kb_id, None)

    # ── 检索 ──

    async def search(
        self,
        query: str,
        kb_id: int,
        top_k: int = settings.LIGHTRAG_TOP_K,
    ) -> RetrievalOutput:
        """执行图谱检索。

        任一前置条件不满足（未建图谱 / 正在构建 / 无匹配）都返回空
        RetrievalOutput，由 RRF 融合层自然降级为双路，不抛异常。
        """
        if not has_graph_data(kb_id) or is_building(kb_id):
            logger.debug("LightRAG 跳过（无图谱或构建中）: kb_id=%s", kb_id)
            return RetrievalOutput(stats={"lightrag_skipped": True})

        from lightrag import QueryParam

        store = LightRAGStore.get()
        handle = await store.get_handle(kb_id)
        param = QueryParam(
            mode=settings.LIGHTRAG_QUERY_MODE,
            top_k=top_k,
            chunk_top_k=top_k,
            only_need_context=True,  # 只取上下文与数据，跳过 LLM 生成
        )

        # LightRAG 内部含同步阻塞（图计算/文件 IO），卸载到实例自身 loop 执行
        data = await asyncio.to_thread(
            store.run_on, handle, handle.rag.aquery_data(query, param=param)
        )

        raw_chunks = (data or {}).get("data", {}).get("chunks", []) or []
        if not raw_chunks:
            return RetrievalOutput(stats={"lightrag_count": 0})

        content_index = await self._get_content_index(kb_id)

        results: list[RetrievalResult] = []
        for rank, item in enumerate(raw_chunks):
            content = item.get("content", "")
            hit = content_index.get(content)
            if hit is None:
                # 内容对不上说明该 chunk 非本系统入库（或已被删除），跳过
                logger.debug("LightRAG 结果无法映射回 chunk，已跳过: %.40s", content)
                continue

            meta = hit["metadata"]
            results.append(
                RetrievalResult(
                    doc_id=hit["doc_id"],
                    chunk_index=hit["chunk_index"],
                    content=content,
                    # RRF 只用排名不用绝对分，此处给递减分仅用于排序可读性
                    score=1.0 / (rank + 1),
                    # 键名必须为 "page"：入库侧 tasks.py 写入的是 meta["page"]，
                    # 向量检索侧 retriever.py 也读 "page"。原写法 "page_number"
                    # 在本索引中不存在，会导致本路结果的页码恒为 None，
                    # 来源卡片丢失页码信息。
                    page=meta.get("page"),
                    doc_name=hit["doc_name"],
                    section_title=meta.get("section_title"),
                    section_path=meta.get("section_path"),
                    embedding=None,  # 无向量，粗排按中性分处理
                )
            )

        logger.info(
            "LightRAG 检索完成: kb_id=%s, 原始 %d 条 → 映射 %d 条",
            kb_id, len(raw_chunks), len(results),
        )
        return RetrievalOutput(
            results=results,
            total=len(results),
            stats={
                "lightrag_count": len(results),
                "lightrag_raw_count": len(raw_chunks),
                "lightrag_mode": settings.LIGHTRAG_QUERY_MODE,
            },
        )
