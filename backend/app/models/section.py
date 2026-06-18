"""章节表 — 存储文档层级结构与 chunk 范围"""

from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models._types import UTCDateTime


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        Index("idx_sections_doc_id", "doc_id"),
        Index("idx_sections_kb_id", "kb_id"),
        Index("idx_sections_doc_level", "doc_id", "level"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文档 ID",
    )
    kb_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属知识库 ID",
    )
    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="章节标题",
    )
    path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="章节路径（如 一级 > 二级 > 当前章节）",
    )
    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="章节层级（1-6）",
    )
    start_chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="章节首个 chunk 的全局索引",
    )
    end_chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="章节最后一个 chunk 的全局索引",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        server_default=func.current_timestamp(),
    )

    document = relationship("Document", back_populates="sections")
    knowledge_base = relationship("KnowledgeBase", back_populates="sections")
    chunks = relationship("Chunk", back_populates="section")
