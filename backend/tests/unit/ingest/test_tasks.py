"""Celery 入库流水线任务测试 — 断点恢复 + last_success_batch 续传 + 阶段检测"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingest.tasks import (
    RESUMABLE_STAGES,
    _build_chroma_metadata,
    _ingest_document_async,
    _replace_sections_and_chunks,
)
from app.models.enums import DocumentStatus
from app.rag.chunker import ChunkResult, ChunkingResult, SectionResult
from tests.helpers import (
    make_mock_doc,
    make_mock_chunks,
    make_mock_embed_result,
    setup_mock_db,
    mock_async_session_ctx,
)


# ==================== RESUMABLE_STAGES ====================


class TestResumableStages:
    """断点恢复阶段常量测试"""

    def test_chunking_done_为可恢复阶段(self):
        assert "chunking_done" in RESUMABLE_STAGES

    def test_embedding_为可恢复阶段(self):
        assert "embedding" in RESUMABLE_STAGES

    def test_vector_storing_为可恢复阶段(self):
        assert "vector_storing" in RESUMABLE_STAGES

    def test_parsing_不在可恢复阶段(self):
        assert "parsing" not in RESUMABLE_STAGES

    def test_chunking_不在可恢复阶段(self):
        assert "chunking" not in RESUMABLE_STAGES


# ==================== 阶段恢复 ====================


class TestStageResume:
    """阶段检测与断点恢复测试"""

    @pytest.mark.asyncio
    async def test_chunking_done阶段_跳过解析分块进入embedding(self):
        """文档 current_stage=chunking_done，应跳过解析+分块，直接进入 Embedding 从 batch 0 开始"""
        doc = make_mock_doc(
            status=DocumentStatus.CHUNKING,
            current_stage="chunking_done",
            last_success_batch=0,
        )
        chunks = make_mock_chunks(5)
        embed_result = make_mock_embed_result(5)
        db = setup_mock_db(doc, chunks)

        with patch("app.ingest.tasks.async_session", return_value=mock_async_session_ctx(db)):
            with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=True):
                with patch("app.ingest.tasks.release_idempotency_lock_async"):
                    with patch("app.ingest.tasks.embed_chunks", AsyncMock(return_value=embed_result)):
                        with patch("app.ingest.tasks.get_vector_store", return_value=AsyncMock()):
                            with patch("app.ingest.tasks.parse_document") as mock_parse:
                                result = await _ingest_document_async(1)
                                mock_parse.assert_not_called()

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_vector_storing阶段_清理chroma并重做embedding(self):
        """文档 current_stage=vector_storing，应清理 ChromaDB + 从 batch 0 重做 Embedding"""
        doc = make_mock_doc(
            status=DocumentStatus.VECTOR_STORING,
            current_stage="vector_storing",
            last_success_batch=3,
        )
        chunks = make_mock_chunks(5)
        embed_result = make_mock_embed_result(5)
        db = setup_mock_db(doc, chunks)

        mock_store = AsyncMock()

        with patch("app.ingest.tasks.async_session", return_value=mock_async_session_ctx(db)):
            with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=True):
                with patch("app.ingest.tasks.release_idempotency_lock_async"):
                    with patch("app.ingest.tasks.embed_chunks", AsyncMock(return_value=embed_result)):
                        with patch("app.ingest.tasks.get_vector_store", return_value=mock_store):
                            result = await _ingest_document_async(1)

        # 验证 ChromaDB 清理被调用（Per-KB：传 kb_id + doc 级 where）
        mock_store.delete.assert_called_with(kb_id=1, where={"doc_id": 1})
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_vector_storing阶段_chroma清理失败标记FAILED(self):
        """vector_storing 阶段 ChromaDB 清理失败应标记 FAILED 并返回"""
        doc = make_mock_doc(
            status=DocumentStatus.VECTOR_STORING,
            current_stage="vector_storing",
            last_success_batch=3,
        )
        chunks = make_mock_chunks(5)
        db = setup_mock_db(doc, chunks)

        mock_store = AsyncMock()
        mock_store.delete.side_effect = RuntimeError("ChromaDB connection failed")

        with patch("app.ingest.tasks.async_session", return_value=mock_async_session_ctx(db)):
            with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=True):
                with patch("app.ingest.tasks.release_idempotency_lock_async"):
                    with patch("app.ingest.tasks.get_vector_store", return_value=mock_store):
                        result = await _ingest_document_async(1)

        assert result["status"] == "failed"
        assert doc.status == DocumentStatus.FAILED
        assert "ChromaDB" in doc.error_msg


# ==================== last_success_batch checkpoint ====================


class TestLastSuccessBatchCheckpoint:
    """last_success_batch checkpoint 更新测试"""

    @pytest.mark.asyncio
    async def test_embedding每批成功后更新last_success_batch(self):
        """验证 embedding 阶段每批成功后会更新 doc.last_success_batch"""
        doc = make_mock_doc(
            status=DocumentStatus.EMBEDDING,
            current_stage="embedding",
            last_success_batch=0,
        )
        # 6 chunks, batch_size=2 → 3 batches
        chunks = make_mock_chunks(6)
        embed_result = make_mock_embed_result(2)  # each batch has 2 chunks
        db = setup_mock_db(doc, chunks)

        # 6 chunks / batch_size=2 = 3 batches
        # 每批完成后 commit 一次（更新 last_success_batch）
        # 全部完成后 commit 一次（最终状态 + token 回写 + KB 统计）
        # 预期至少 4 次 commit: 3 per-batch + 1 final

        with patch("app.ingest.tasks.async_session", return_value=mock_async_session_ctx(db)):
            with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=True):
                with patch("app.ingest.tasks.release_idempotency_lock_async"):
                    with patch("app.ingest.tasks.embed_chunks", AsyncMock(return_value=embed_result)):
                        with patch("app.ingest.tasks.get_vector_store", return_value=AsyncMock()):
                            with patch("app.ingest.tasks.settings") as mock_settings:
                                mock_settings.EMBED_BATCH_SIZE = 2
                                mock_settings.CHROMA_BATCH_SIZE = 20
                                result = await _ingest_document_async(1)

        assert result["status"] == "completed"
        # 3 per-batch commits + 1 final commit = 4
        assert db.commit.call_count >= 4

    @pytest.mark.asyncio
    async def test_last_success_batch为0时从第一批开始(self):
        """last_success_batch=0 时，embedding 从第 0 批开始"""
        doc = make_mock_doc(
            status=DocumentStatus.EMBEDDING,
            current_stage="embedding",
            last_success_batch=0,
        )
        chunks = make_mock_chunks(3)
        embed_result = make_mock_embed_result(3)
        db = setup_mock_db(doc, chunks)

        with patch("app.ingest.tasks.async_session", return_value=mock_async_session_ctx(db)):
            with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=True):
                with patch("app.ingest.tasks.release_idempotency_lock_async"):
                    mock_embed = AsyncMock(return_value=embed_result)
                    with patch("app.ingest.tasks.embed_chunks", mock_embed):
                        with patch("app.ingest.tasks.get_vector_store", return_value=AsyncMock()):
                            result = await _ingest_document_async(1)

        assert result["status"] == "completed"
        # 3 chunks, batch_size 默认 20 → 1 batch
        assert mock_embed.call_count == 1


# ==================== 幂等锁集成 ====================


class TestIdempotencyLockIntegration:
    """幂等锁与流水线集成测试"""

    @pytest.mark.asyncio
    async def test_锁被占用时返回locked(self):
        """幂等锁已被占用时，任务应返回 locked 状态"""
        with patch("app.ingest.tasks.acquire_idempotency_lock_async", return_value=False):
            with patch("app.ingest.tasks.release_idempotency_lock_async"):
                result = await _ingest_document_async(1)

        assert result["status"] == "locked"
        assert result["doc_id"] == 1


class TestSectionPersistence:
    """PR2 章节与分块写入测试"""

    @pytest.mark.asyncio
    async def test_replace_sections_and_chunks写入section_id与兼容metadata(self):
        db = AsyncMock()
        db.execute = AsyncMock()
        db.flush = AsyncMock()
        db.add = MagicMock()

        added_sections = []
        added_chunks = []

        def _add(obj):
            if obj.__class__.__name__ == "Section":
                obj.id = 100 + len(added_sections)
                added_sections.append(obj)
            elif obj.__class__.__name__ == "Chunk":
                added_chunks.append(obj)

        db.add.side_effect = _add

        chunking_result = ChunkingResult(
            sections=[
                SectionResult(
                    title="第一章",
                    path="第一章",
                    level=1,
                    start_offset=0,
                    end_offset=100,
                    start_chunk_index=0,
                    end_chunk_index=1,
                ),
                SectionResult(
                    title="全文",
                    path="全文",
                    level=1,
                    start_offset=100,
                    end_offset=150,
                    start_chunk_index=2,
                    end_chunk_index=2,
                    synthetic=True,
                ),
            ],
            chunks=[
                ChunkResult(
                    content="第一块",
                    chunk_index=0,
                    page_number=1,
                    estimated_tokens=10,
                    section_index=0,
                    section_title="第一章",
                    section_path="第一章",
                ),
                ChunkResult(
                    content="第二块",
                    chunk_index=1,
                    page_number=2,
                    estimated_tokens=8,
                    section_index=0,
                    section_title="第一章",
                    section_path="第一章",
                ),
                ChunkResult(
                    content="前言块",
                    chunk_index=2,
                    page_number=None,
                    estimated_tokens=6,
                    section_index=1,
                    section_title=None,
                    section_path=None,
                ),
            ],
            total_chunks=3,
        )

        await _replace_sections_and_chunks(db, doc_id=1, kb_id=2, chunking_result=chunking_result)

        assert db.execute.await_count == 2
        assert db.flush.await_count == 2
        assert len(added_sections) == 2
        assert len(added_chunks) == 3
        assert added_chunks[0].section_id == 100
        assert added_chunks[1].section_id == 100
        assert added_chunks[2].section_id == 101
        assert added_chunks[0].metadata_ == {
            "page": 1,
            "section_title": "第一章",
            "section_path": "第一章",
        }
        assert added_chunks[2].metadata_ is None


class TestChromaMetadata:
    """Chroma metadata 构建测试"""

    def test_build_chroma_metadata包含section_id并保留兼容字段(self):
        metadata = _build_chroma_metadata(
            kb_id=2,
            doc_id=3,
            chunk_row={
                "chunk_index": 4,
                "section_id": 99,
                "section_title": "第二章",
                "section_path": "第一章 > 第二章",
            },
        )
        assert metadata == {
            "kb_id": 2,
            "doc_id": 3,
            "chunk_index": 4,
            "section_id": 99,
            "section_title": "第二章",
            "section_path": "第一章 > 第二章",
        }

    def test_build_chroma_metadata无section_id时不写入该字段(self):
        metadata = _build_chroma_metadata(
            kb_id=2,
            doc_id=3,
            chunk_row={
                "chunk_index": 4,
                "section_id": None,
                "section_title": "",
                "section_path": "",
            },
        )
        assert metadata == {
            "kb_id": 2,
            "doc_id": 3,
            "chunk_index": 4,
            "section_title": "",
            "section_path": "",
        }
