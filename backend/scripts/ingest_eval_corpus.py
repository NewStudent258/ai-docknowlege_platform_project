"""把评测语料入库到独立知识库（目标语料 + KB14 干扰语料）。

用途
----
为重排消融实验构建评测环境：
  - 目标语料：backend/knowledge_samples/eval_corpus/*.md（25 份，脚本生成）
  - 干扰语料：从既有 KB14（100 份薪酬/考勤/安全类制度）复制
两者合并进一个新知识库，干扰文档与目标文档**同属企业制度域但内容无关**，
从而对排序形成真实压力（这是让 Hit@1 / MRR 产生区分度的前提）。

实现说明
--------
复用真实入库链路的分阶段函数（parse_document / chunk_document / embed_chunks /
vector_store.add），不走 Celery，便于评测环境快速重建。
写入 MySQL（documents / chunks）+ ChromaDB，与生产入库结果等价。

用法：
  cd backend
  python scripts/ingest_eval_corpus.py --name "评测语料库" --with-distractors
  python scripts/ingest_eval_corpus.py --name "评测语料库(无干扰)"    # 目标语料 only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import func, select

from app.core.chroma_client import get_vector_store
from app.core.database import async_session
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_chunks
from app.rag.parser import parse_document
from app.config import settings

# 关闭 SQLAlchemy 逐条 SQL 日志，保持评测输出可读
import logging as _logging

_logging.getLogger("sqlalchemy.engine").setLevel(_logging.WARNING)
_logging.getLogger("sqlalchemy.engine.Engine").setLevel(_logging.WARNING)

CORPUS_DIR = _BACKEND_ROOT / "knowledge_samples" / "eval_corpus"
DISTRACTOR_KB_ID = 14  # 既有「公司制度汇编（100 份）」


async def _get_or_create_kb(db, name: str, owner_id: int) -> KnowledgeBase:
    kb = (
        await db.execute(select(KnowledgeBase).where(KnowledgeBase.name == name))
    ).scalar_one_or_none()
    if kb is not None:
        print(f"复用已存在知识库: id={kb.id} name={kb.name}")
        return kb

    kb = KnowledgeBase(
        uuid=str(uuid4()),
        name=name,
        description="重排消融实验评测语料（目标文档 + 同领域干扰文档）",
        visibility="public",
        status="active",
        user_id=owner_id,
        doc_count=0,
        chunk_count=0,
    )
    db.add(kb)
    await db.flush()
    print(f"新建知识库: id={kb.id} name={kb.name}")
    return kb


async def _ingest_file(
    db,
    kb_id: int,
    user_id: int,
    path: Path,
    filename: str | None = None,
) -> int:
    """入库单个文件，返回写入的 chunk 数。"""
    display_name = filename or path.name
    file_type = path.suffix.lstrip(".").lower()

    parse_result = parse_document(str(path), file_type)
    if not parse_result.full_text.strip():
        print(f"    ⚠️  解析为空，跳过: {display_name}")
        return 0

    chunking = chunk_document(parse_result.full_text, parse_result.pages)
    if not chunking.chunks:
        print(f"    ⚠️  无分块，跳过: {display_name}")
        return 0

    # 幂等：同名文档已存在则跳过
    existing = (
        await db.execute(
            select(Document).where(
                Document.kb_id == kb_id, Document.filename == display_name,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return -1  # 已存在标记

    doc = Document(
        uuid=str(uuid4()),
        kb_id=kb_id,
        filename=display_name,
        file_path=str(path),
        file_size=path.stat().st_size,
        file_type=file_type,
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunking.chunks),
        current_stage="completed",
    )
    db.add(doc)
    await db.flush()

    # Embedding（真实调用 DashScope，按 EMBED_BATCH_SIZE 分批）
    contents = [c.content for c in chunking.chunks]
    vectors: list[list[float]] = []
    batch_size = settings.EMBED_BATCH_SIZE
    for i in range(0, len(contents), batch_size):
        batch = contents[i : i + batch_size]
        result = await embed_chunks(batch)
        vectors.extend(result.embeddings)

    if len(vectors) != len(contents):
        raise RuntimeError(
            f"向量数不匹配: {len(vectors)} != {len(contents)} ({display_name})"
        )

    # 写 ChromaDB（沿用生产约定 doc_{doc_id}_chunk_{chunk_index}）
    chroma_ids = [f"doc_{doc.id}_chunk_{c.chunk_index}" for c in chunking.chunks]
    store = get_vector_store()
    metadatas = [
        {
            "kb_id": int(kb_id),
            "doc_id": int(doc.id),
            "chunk_index": int(c.chunk_index),
            "doc_name": display_name,
        }
        for c in chunking.chunks
    ]
    await store.add(
        ids=chroma_ids,
        kb_id=kb_id,
        documents=contents,
        embeddings=vectors,
        metadatas=metadatas,
    )

    # 写 MySQL chunks
    for c, cid in zip(chunking.chunks, chroma_ids):
        db.add(Chunk(
            doc_id=doc.id,
            kb_id=kb_id,
            chroma_id=cid,
            content=c.content,
            chunk_index=c.chunk_index,
            token_count=c.estimated_tokens,
            metadata_={
                "section_title": c.section_title,
                "section_path": c.section_path,
                "page_number": c.page_number,
            },
        ))

    kb = await db.get(KnowledgeBase, kb_id)
    if kb is not None:
        kb.doc_count = (kb.doc_count or 0) + 1
        kb.chunk_count = (kb.chunk_count or 0) + len(chunking.chunks)

    return len(chunking.chunks)


async def _copy_distractors(dst_kb_id: int, owner_id: int) -> int:
    """把源 KB 的全部文档复制到目标 KB（文本级复制，重新 embedding）。"""
    async with async_session() as db:
        rows = (
            await db.execute(
                select(Document.filename, Document.file_path)
                .where(Document.kb_id == DISTRACTOR_KB_ID)
                .order_by(Document.filename)
            )
        ).all()
    print(f"\n复制干扰语料: 源 kb={DISTRACTOR_KB_ID} 共 {len(rows)} 份 → kb={dst_kb_id}")

    total = 0
    for i, (filename, file_path) in enumerate(rows, start=1):
        p = Path(file_path)
        if not p.exists():
            print(f"  [{i}/{len(rows)}] ⚠️  源文件不存在: {filename} ({file_path})")
            continue

        try:
            async with async_session() as db:
                n = await _ingest_file(db, dst_kb_id, owner_id, p, filename=filename)
                await db.commit()
        except Exception as e:
            print(f"  [{i}/{len(rows)}] ❌ {filename}: {type(e).__name__}: {e}")
            continue

        if n == -1:
            print(f"  [{i}/{len(rows)}] 已存在，跳过: {filename}")
        elif n > 0:
            total += n
            print(f"  [{i}/{len(rows)}] ✅ {filename}: {n} chunks")

    return total


async def main_async(name: str, with_distractors: bool) -> None:
    if not CORPUS_DIR.exists():
        print(f"❌ 语料目录不存在: {CORPUS_DIR}")
        print("   请先运行: python scripts/gen_eval_corpus.py")
        return

    corpus_files = sorted(p for p in CORPUS_DIR.glob("*.md"))
    print(f"目标语料: {len(corpus_files)} 份 目录={CORPUS_DIR}")

    async with async_session() as db:
        # owner：沿用 KB14 的 owner，确保与既有数据同权限上下文
        src_kb = await db.get(KnowledgeBase, DISTRACTOR_KB_ID)
        owner_id = src_kb.user_id if src_kb else 1

        kb = await _get_or_create_kb(db, name, owner_id)
        await db.commit()
        kb_id = kb.id

    print(f"\n入库目标语料 → kb={kb_id}")
    ok = 0
    for i, path in enumerate(corpus_files, start=1):
        # 每份文档独立 session，失败不影响后续
        try:
            async with async_session() as db:
                n = await _ingest_file(db, kb_id, owner_id, path)
                await db.commit()
            if n == -1:
                print(f"  [{i}/{len(corpus_files)}] 已存在，跳过: {path.name}")
            else:
                ok += n
                print(f"  [{i}/{len(corpus_files)}] ✅ {path.name}: {n} chunks")
        except Exception as e:
            print(f"  [{i}/{len(corpus_files)}] ❌ {path.name}: {type(e).__name__}: {e}")

    print(f"\n目标语料入库完成: {ok} chunks")

    if with_distractors:
        try:
            dn = await _copy_distractors(kb_id, owner_id)
            print(f"\n干扰语料入库完成: {dn} chunks")
        except Exception as e:
            print(f"❌ 干扰语料入库失败: {type(e).__name__}: {e}")

    async with async_session() as db:
        # 汇总（用真实 COUNT 而非缓存列）
        doc_n = (
            await db.execute(
                select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
            )
        ).scalar()
        chunk_n = (
            await db.execute(
                select(func.count()).select_from(Chunk).where(Chunk.kb_id == kb_id)
            )
        ).scalar()
        print(f"\n{'='*70}")
        print(f"知识库 id={kb_id} name={name}")
        print(f"文档数(实时 COUNT)={doc_n}  chunk 数(实时 COUNT)={chunk_n}")
        print(f"{'='*70}")
        print(f"\n后续评测命令:")
        print(f"  python tests/eval/eval_retrieval.py --kb-id {kb_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把评测语料入库到独立知识库")
    parser.add_argument("--name", required=True, help="知识库名称")
    parser.add_argument(
        "--with-distractors", action="store_true",
        help=f"同时复制 KB{DISTRACTOR_KB_ID} 的 100 份文档作为干扰语料",
    )
    args = parser.parse_args()
    asyncio.run(main_async(name=args.name, with_distractors=args.with_distractors))


if __name__ == "__main__":
    main()
