"""用户模型测试 — U4.1 / U4.2 / U4.3"""
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.section import Section
from app.models.chunk import Chunk
from app.models.enums import DocumentStatus
from app.core.database import async_session, engine


@pytest.fixture(autouse=True)
async def dispose_engine_after():
    """每个测试后清理连接池，避免 Windows ProactorEventLoop 残留连接异常"""
    yield
    await engine.dispose()


class TestUserModel:
    """U4.x — 用户模型测试"""

    @pytest.mark.asyncio
    async def test_user_default_role(self):
        """U4.1: 创建时不指定 role，默认值为 "user" """
        async with async_session() as session:
            user = User(username="test_u41_20260517", password_hash="hashed_xxx")
            session.add(user)
            await session.flush()
            await session.refresh(user)
            assert user.role == "user"

    @pytest.mark.asyncio
    async def test_user_username_unique(self):
        """U4.2: 重复 username 触发 IntegrityError"""
        username = "test_u42_20260517"

        async with async_session() as session:
            async with session.begin_nested() as savepoint:
                user1 = User(username=username, password_hash="hash1")
                session.add(user1)
                await session.flush()

                user2 = User(username=username, password_hash="hash2")
                session.add(user2)

                try:
                    await session.flush()
                except IntegrityError:
                    await savepoint.rollback()
                else:
                    pytest.fail("Expected IntegrityError was not raised")

    @pytest.mark.asyncio
    async def test_user_knowledge_bases_relationship(self):
        """U4.3: 验证 KnowledgeBase 通过 FK 关联 User（空列表 → 关联存在）"""
        username = "test_u43_20260517"

        async with async_session() as session:
            user = User(username=username, password_hash="hash1")
            session.add(user)
            await session.flush()

            # 未创建 KB 时，查 KB 表按 user_id 筛选为空
            stmt = select(KnowledgeBase).where(KnowledgeBase.user_id == user.id)
            result = await session.execute(stmt)
            assert len(result.scalars().all()) == 0

            # 创建 KB 后，通过 FK 可查到该 KB
            kb = KnowledgeBase(name="u43_测试知识库", user_id=user.id, uuid=str(uuid4()))
            session.add(kb)
            await session.flush()

            result = await session.execute(stmt)
            kbs = result.scalars().all()
            assert len(kbs) == 1
            assert kbs[0].name == "u43_测试知识库"
            assert kbs[0].user_id == user.id


class TestSectionModel:
    """Section / Chunk / Document 关系测试"""

    def test_section_chunk_document_relationship(self):
        """Section 关系 wiring 正确，Chunk 可回指 Section 和 Document"""
        doc = Document(
            id=10,
            uuid=str(uuid4()),
            kb_id=1,
            filename="test.md",
            file_type="md",
            status=DocumentStatus.UPLOADED,
        )
        section = Section(
            id=20,
            doc_id=10,
            kb_id=1,
            title="章节一",
            path="章节一",
            level=1,
            start_chunk_index=0,
            end_chunk_index=0,
        )
        chunk = Chunk(
            id=30,
            doc_id=10,
            kb_id=1,
            section_id=20,
            chroma_id="doc_10_chunk_0",
            content="测试内容",
            chunk_index=0,
        )

        doc.sections.append(section)
        doc.chunks.append(chunk)
        section.chunks.append(chunk)

        assert section.document is doc
        assert section in doc.sections
        assert chunk.document is doc
        assert chunk.section is section
        assert chunk in section.chunks
