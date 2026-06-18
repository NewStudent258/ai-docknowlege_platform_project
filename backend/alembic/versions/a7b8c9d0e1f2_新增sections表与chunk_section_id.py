"""新增sections表与chunks.section_id外键

Revision ID: a7b8c9d0e1f2
Revises: f5a6b7c8d9e0
Create Date: 2026-06-28

为 Phase 6 层级检索准备关系模型：
- 新增 sections 表，持久化文档章节结构
- 给 chunks 表增加可空 section_id 外键，兼容老数据渐进回填
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f5a6b7c8d9e0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = :t"
        ),
        {"t": table_name},
    )
    return result.scalar() > 0


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :t AND column_name = :c"
        ),
        {"t": table_name, "c": column_name},
    )
    return result.scalar() > 0


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = :t AND index_name = :i"
        ),
        {"t": table_name, "i": index_name},
    )
    return result.scalar() > 0


def _foreign_key_exists(table_name: str, constraint_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_schema = DATABASE() AND table_name = :t "
            "AND constraint_type = 'FOREIGN KEY' AND constraint_name = :c"
        ),
        {"t": table_name, "c": constraint_name},
    )
    return result.scalar() > 0


def upgrade() -> None:
    if not _table_exists("sections"):
        op.create_table(
            "sections",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("doc_id", sa.BigInteger(), nullable=False, comment="所属文档 ID"),
            sa.Column("kb_id", sa.BigInteger(), nullable=False, comment="所属知识库 ID"),
            sa.Column("title", sa.String(length=512), nullable=False, comment="章节标题"),
            sa.Column(
                "path",
                sa.String(length=1024),
                nullable=False,
                comment="章节路径（如 一级 > 二级 > 当前章节）",
            ),
            sa.Column("level", sa.Integer(), nullable=False, comment="章节层级（1-6）"),
            sa.Column(
                "start_chunk_index",
                sa.Integer(),
                nullable=False,
                comment="章节首个 chunk 的全局索引",
            ),
            sa.Column(
                "end_chunk_index",
                sa.Integer(),
                nullable=False,
                comment="章节最后一个 chunk 的全局索引",
            ),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["doc_id"],
                ["documents.id"],
                name="fk_sections_doc_id_documents",
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["kb_id"],
                ["knowledge_bases.id"],
                name="fk_sections_kb_id_knowledge_bases",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _index_exists("sections", "idx_sections_doc_id"):
        op.create_index("idx_sections_doc_id", "sections", ["doc_id"], unique=False)
    if not _index_exists("sections", "idx_sections_kb_id"):
        op.create_index("idx_sections_kb_id", "sections", ["kb_id"], unique=False)
    if not _index_exists("sections", "idx_sections_doc_level"):
        op.create_index("idx_sections_doc_level", "sections", ["doc_id", "level"], unique=False)

    if not _column_exists("chunks", "section_id"):
        op.add_column(
            "chunks",
            sa.Column("section_id", sa.BigInteger(), nullable=True, comment="所属章节 ID"),
        )

    if not _index_exists("chunks", "ix_chunks_section_id"):
        op.create_index("ix_chunks_section_id", "chunks", ["section_id"], unique=False)

    if not _foreign_key_exists("chunks", "fk_chunks_section_id_sections"):
        op.create_foreign_key(
            "fk_chunks_section_id_sections",
            "chunks",
            "sections",
            ["section_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    if _foreign_key_exists("chunks", "fk_chunks_section_id_sections"):
        op.drop_constraint("fk_chunks_section_id_sections", "chunks", type_="foreignkey")

    if _index_exists("chunks", "ix_chunks_section_id"):
        op.drop_index("ix_chunks_section_id", table_name="chunks")

    if _column_exists("chunks", "section_id"):
        op.drop_column("chunks", "section_id")

    if _table_exists("sections"):
        if _index_exists("sections", "idx_sections_doc_level"):
            op.drop_index("idx_sections_doc_level", table_name="sections")
        if _index_exists("sections", "idx_sections_kb_id"):
            op.drop_index("idx_sections_kb_id", table_name="sections")
        if _index_exists("sections", "idx_sections_doc_id"):
            op.drop_index("idx_sections_doc_id", table_name="sections")
        op.drop_table("sections")
