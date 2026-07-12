"""LightRAG 实例管理 — 图谱构建、实例缓存与删除（ADR-025）

设计要点：
- **按 KB 分 working_dir**（`lightrag_data/{kb_id}/`），与 ChromaDB 的 per-KB collection 策略对齐
- **进程内实例缓存**：LightRAG 实例化要加载图/向量/KV 存储，是同步阻塞 IO，
  故按 kb_id 缓存（threading.Lock 保护 + LRU 上限 + 空闲 TTL），避免每次查询重建
- **失败降级**：图谱构建失败不阻塞文档入库（LightRAG 只是三路检索之一）
- **读写分离**：构建期间写 `.building` 标记文件，检索侧见标记则跳过该 KB，避免读到中间态

与 MySQL 的对齐方式：
  插入时把每个 chunk 作为一条独立文本传给 LightRAG，`ids` 用与 ChromaDB 相同的
  `doc_{doc_id}_chunk_{chunk_index}` 约定。但注意 LightRAG 返回的 chunk 只有
  `{reference_id, content, file_path, chunk_id}`，**不含 full_doc_id**，且 chunk_id
  是内部哈希，因此检索侧改用 **content 精确匹配** 还原 (doc_id, chunk_index)，
  不依赖 LightRAG 的 id 方案。参见 lightrag_retriever.py。
"""

import asyncio
import logging
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# 构建中标记文件名（存在时检索侧跳过该 KB）
_BUILDING_MARKER = ".building"


def _create_lightrag_loop() -> asyncio.AbstractEventLoop:
    """为 LightRAG 实例创建专用事件循环

    Windows 下必须用 SelectorEventLoop（Proactor 会触发「句柄无效」/「loop 已关闭」）。
    这里显式构造而不调用 asyncio.set_event_loop_policy()，避免进程级副作用
    波及 uvicorn 主循环与 aiomysql。
    """
    import sys

    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def kb_dir(kb_id: int) -> Path:
    """某 KB 的 LightRAG 工作目录"""
    return Path(settings.LIGHTRAG_DATA_DIR) / str(kb_id)


def has_graph_data(kb_id: int) -> bool:
    """该 KB 是否已构建过图谱（目录存在且非构建中）"""
    d = kb_dir(kb_id)
    return d.exists() and not (d / _BUILDING_MARKER).exists()


def is_building(kb_id: int) -> bool:
    """该 KB 是否正在构建图谱"""
    return (kb_dir(kb_id) / _BUILDING_MARKER).exists()


def _build_llm_model_func():
    """实体/关系抽取用的 LLM 回调 — 走 DeepSeek（OpenAI 兼容）"""
    from lightrag.llm.openai import openai_complete_if_cache

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            settings.LIGHTRAG_LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            **kwargs,
        )

    return llm_model_func


def _build_embedding_func():
    """Embedding 回调 — 复用现有 DashScope embed_chunks（1024 维）

    embed_chunks 内部每次调用自建 httpx.AsyncClient（embedder.py），
    跨事件循环调用是安全的；但它单批上限为 EMBED_BATCH_SIZE(=10)，此处内部分批。
    """
    import numpy as np
    from lightrag.utils import EmbeddingFunc

    from app.rag.embedder import embed_chunks

    async def embed_func(texts: list[str]) -> "np.ndarray":
        vectors: list[list[float]] = []
        batch = max(1, settings.EMBED_BATCH_SIZE)
        for i in range(0, len(texts), batch):
            result = await embed_chunks(texts[i : i + batch], text_type="document")
            vectors.extend(result.embeddings)
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(embedding_dim=1024, func=embed_func)


@dataclass
class _Handle:
    """一个 KB 对应的 LightRAG 实例及其事件循环"""

    kb_id: int
    rag: object
    loop: asyncio.AbstractEventLoop
    last_access: float = field(default_factory=time.monotonic)


class LightRAGStore:
    """进程内 per-KB LightRAG 实例缓存（LRU + TTL）

    注意：实例不是线程安全的共享缓冲，所有对 rag 的调用都必须在该实例的
    loop 上执行（见 `run_on`）。
    """

    _instance: "LightRAGStore | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._handles: OrderedDict[int, _Handle] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "LightRAGStore":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 实例获取 ──

    def _build(self, kb_id: int) -> _Handle:
        """创建 LightRAG 实例（同步阻塞，需在锁外调用）"""
        from lightrag import LightRAG

        work = kb_dir(kb_id)
        work.mkdir(parents=True, exist_ok=True)

        loop = _create_lightrag_loop()
        rag = LightRAG(
            working_dir=str(work),
            llm_model_func=_build_llm_model_func(),
            llm_model_name=settings.LIGHTRAG_LLM_MODEL,
            embedding_func=_build_embedding_func(),
            chunk_token_size=settings.LIGHTRAG_CHUNK_TOKEN_SIZE,
        )

        # LightRAG 要求实例化后必须初始化存储，否则 ainsert/aquery 会抛
        # PipelineNotInitializedError。该调用是异步的，必须在本实例自己的 loop 上执行。
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag.initialize_storages())

        logger.info("LightRAG 实例已创建并完成存储初始化: kb_id=%s, working_dir=%s", kb_id, work)
        return _Handle(kb_id=kb_id, rag=rag, loop=loop)

    async def get_handle(self, kb_id: int) -> _Handle:
        """取（或创建）指定 KB 的实例，并按 LRU/TTL 淘汰

        构建必须在线程中执行：`_build` 里要为新实例跑
        `loop.run_until_complete(initialize_storages())`，而调用方（FastAPI /
        Celery / 脚本）线程上已经有 loop 在跑，直接调用会触发
        「Cannot run the event loop while another loop is running」。
        """
        now = time.monotonic()
        with self._lock:
            handle = self._handles.get(kb_id)
            if handle is not None:
                handle.last_access = now
                self._handles.move_to_end(kb_id)
                return handle

            # 淘汰：超 TTL 或超容量
            while self._handles:
                oldest_kb, oldest = next(iter(self._handles.items()))
                expired = (now - oldest.last_access) > settings.LIGHTRAG_INSTANCE_TTL
                overflow = len(self._handles) >= settings.LIGHTRAG_MAX_INSTANCES
                if not (expired or overflow):
                    break
                self._handles.pop(oldest_kb, None)
                self._close(oldest)
                logger.info("LightRAG 实例已淘汰: kb_id=%s", oldest_kb)

        # 锁外构建（同步阻塞 IO + 初始化），卸载到工作线程
        handle = await asyncio.to_thread(self._build, kb_id)
        with self._lock:
            self._handles[kb_id] = handle
            self._handles.move_to_end(kb_id)
        return handle

    @staticmethod
    def _close(handle: _Handle) -> None:
        """关闭实例所属 loop（不强制杀线程，交由 GC）"""
        try:
            if not handle.loop.is_closed():
                handle.loop.close()
        except Exception:  # noqa: BLE001 — 清理失败不影响主流程
            logger.debug("关闭 LightRAG loop 失败: kb_id=%s", handle.kb_id, exc_info=True)

    def invalidate(self, kb_id: int, *, remove_data: bool = False) -> None:
        """失效某 KB 的实例缓存；remove_data=True 时同时删除其工作目录"""
        with self._lock:
            handle = self._handles.pop(kb_id, None)
        if handle is not None:
            self._close(handle)

        if remove_data:
            d = kb_dir(kb_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info("LightRAG 工作目录已删除: %s", d)

    def run_on(self, handle: _Handle, coro):
        """在该实例所属 loop 上执行协程（供 asyncio.to_thread 内调用）

        LightRAG 内部含同步阻塞代码（NetworkX 图计算 / 文件 IO），
        不能直接在 FastAPI 事件循环上 await，故统一卸载到实例自己的 loop 执行。
        """
        asyncio.set_event_loop(handle.loop)
        return handle.loop.run_until_complete(coro)


# ── 构建 / 删除（供 Celery 任务与回填脚本共用，避免逻辑漂移）──


async def build_lightrag_graph_for_doc(
    kb_id: int, doc_id: int, chunk_rows: list[dict]
) -> bool:
    """把某文档的 chunks 灌入 LightRAG 图谱

    Args:
        chunk_rows: 至少含 `chroma_id` 与 `content` 的字典列表

    Returns:
        成功 True；失败 False（调用方降级，不阻塞文档入库）
    """
    if not chunk_rows:
        return True

    store = LightRAGStore.get()
    marker = kb_dir(kb_id) / _BUILDING_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    try:
        handle = await store.get_handle(kb_id)
        texts = [row["content"] for row in chunk_rows]
        ids = [row["chroma_id"] for row in chunk_rows]

        await asyncio.to_thread(
            store.run_on, handle, handle.rag.ainsert(texts, ids=ids)
        )
        logger.info(
            "LightRAG 图谱构建完成: kb_id=%s, doc_id=%s, chunks=%d",
            kb_id, doc_id, len(texts),
        )
        return True
    except Exception:  # noqa: BLE001 — 图谱构建失败不应阻塞入库
        logger.exception(
            "LightRAG 图谱构建失败（降级，不阻塞入库）: kb_id=%s, doc_id=%s", kb_id, doc_id
        )
        return False
    finally:
        marker.unlink(missing_ok=True)


async def delete_doc_from_lightrag(kb_id: int, doc_id: int, chroma_ids: list[str]) -> bool:
    """从图谱中删除某文档的所有 chunk（逐条 adelete_by_doc_id）

    LightRAG 以「我们传入的 id」为一个文档单位，因此需按 chunk 逐个删除。
    """
    if not chroma_ids:
        return True

    store = LightRAGStore.get()
    if not kb_dir(kb_id).exists():
        return True

    try:
        handle = await store.get_handle(kb_id)
        for cid in chroma_ids:
            await asyncio.to_thread(
                store.run_on, handle, handle.rag.adelete_by_doc_id(cid)
            )
        logger.info(
            "LightRAG 文档已删除: kb_id=%s, doc_id=%s, chunks=%d",
            kb_id, doc_id, len(chroma_ids),
        )
        return True
    except Exception:  # noqa: BLE001 — 图谱清理失败不影响文档删除主流程
        logger.exception(
            "LightRAG 文档删除失败（忽略，不影响主流程）: kb_id=%s, doc_id=%s", kb_id, doc_id
        )
        return False
