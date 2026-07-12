"""LightRAG 图谱回填脚本 — 对已入库文档补建图谱（ADR-025）

用于存量文档：入库时 LIGHTRAG_ENABLED 尚未开启，事后补建图谱。

图谱构建会对每个 chunk 调用一次 LLM 做实体/关系抽取，**有真实 API 成本**，
因此默认只处理少量文档做验证，确认效果后再全量。

用法：
    # 对指定 KB 的前 10 份文档回填（子集验证）
    python scripts/lightrag_backfill.py --kb-uuid <uuid> --docs 10

    # 指定具体文档 id
    python scripts/lightrag_backfill.py --doc-ids 1,2,3

    # 全量回填（谨慎，消耗大量 LLM 调用）
    python scripts/lightrag_backfill.py --kb-uuid <uuid> --all

注意：脚本与 Celery 任务共用 `build_lightrag_graph_for_doc`，避免两份实现漂移。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 允许从 backend/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.database import async_session  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.enums import DocumentStatus  # noqa: E402
from app.models.knowledge_base import KnowledgeBase  # noqa: E402
from app.rag.lightrag_store import (  # noqa: E402
    build_lightrag_graph_for_doc,
    kb_dir,
)

RETRIEVABLE = [
    DocumentStatus.COMPLETED,
    DocumentStatus.SUCCESS_WITH_WARNINGS,
    DocumentStatus.PARTIAL_FAILED,
]


async def _load_targets(args) -> list[tuple[int, int, str]]:
    """返回 [(doc_id, kb_id, filename), ...]"""
    async with async_session() as db:
        if args.doc_ids:
            ids = [int(x) for x in args.doc_ids.split(",") if x.strip()]
            rows = await db.execute(
                select(Document.id, Document.kb_id, Document.filename).where(
                    Document.id.in_(ids)
                )
            )
            return list(rows.all())

        if not args.kb_uuid:
            raise SystemExit("需提供 --kb-uuid 或 --doc-ids")

        kb = (
            await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.uuid == args.kb_uuid)
            )
        ).scalar_one_or_none()
        if kb is None:
            raise SystemExit(f"未找到知识库: {args.kb_uuid}")

        q = (
            select(Document.id, Document.kb_id, Document.filename)
            .where(Document.kb_id == kb.id, Document.status.in_(RETRIEVABLE))
            .order_by(Document.id)
        )
        if not args.all:
            q = q.limit(args.docs)
        return list((await db.execute(q)).all())


async def _load_chunks(doc_id: int) -> list[dict]:
    async with async_session() as db:
        rows = await db.execute(
            select(Chunk.chroma_id, Chunk.content)
            .where(Chunk.doc_id == doc_id)
            .order_by(Chunk.chunk_index)
        )
        return [{"chroma_id": cid, "content": content} for cid, content in rows.all()]


async def main() -> None:
    parser = argparse.ArgumentParser(description="LightRAG 图谱回填")
    parser.add_argument("--kb-uuid", help="知识库 UUID")
    parser.add_argument("--doc-ids", help="逗号分隔的文档 id 列表")
    parser.add_argument("--docs", type=int, default=10, help="处理前 N 份（默认 10）")
    parser.add_argument("--all", action="store_true", help="处理全部（谨慎）")
    args = parser.parse_args()

    if not settings.LIGHTRAG_ENABLED:
        print("⚠ LIGHTRAG_ENABLED=false，仍继续回填（图谱数据会写入，但检索侧暂不启用）")

    targets = await _load_targets(args)
    if not targets:
        print("没有符合条件的文档")
        return

    total_chunks = 0
    ok_docs = 0
    print(f"待回填 {len(targets)} 份文档")
    print("-" * 60)

    for idx, (doc_id, kb_id, filename) in enumerate(targets, 1):
        chunks = await _load_chunks(doc_id)
        if not chunks:
            print(f"[{idx}/{len(targets)}] doc {doc_id} 无分块，跳过")
            continue

        print(f"[{idx}/{len(targets)}] doc {doc_id} 「{filename}」 {len(chunks)} 个分块 … ", end="", flush=True)
        # 复用与 Celery 任务完全相同的实现
        success = await build_lightrag_graph_for_doc(kb_id, doc_id, chunks)
        if success:
            ok_docs += 1
            total_chunks += len(chunks)
            print("OK")
        else:
            print("失败（详见日志）")

    print("-" * 60)
    print(f"完成：{ok_docs}/{len(targets)} 份文档，{total_chunks} 个分块已入图谱")
    if targets:
        print(f"图谱目录：{kb_dir(targets[0][1]).resolve()}")


if __name__ == "__main__":
    # Windows 下 aiomysql 需要 SelectorEventLoop（与 celery_app.py 同策略）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
