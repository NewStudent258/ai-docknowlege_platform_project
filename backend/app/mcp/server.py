"""MCP Server — 把 DocMind 知识库能力暴露为 MCP 工具（ADR-026）

提供 4 个工具：
- `list_knowledge_bases` 列出当前用户可访问的知识库
- `list_documents`       列出指定知识库的文档
- `retrieve`             只做检索，返回引用片段（不调用 LLM 生成）
- `ask`                  完整问答，返回答案 + 引用来源

鉴权：复用现有 AuthMiddleware 做 401 门控 + auth.py 的 contextvar 透传身份。
挂载方式见 app/main.py（`/mcp`）。
"""

import logging

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from app.config import settings
from app.core.database import async_session
from app.mcp.auth import MCPUserContextMiddleware, require_current_user
from app.models.knowledge_base import KnowledgeBase
from app.rag.knowledge_pipeline import KnowledgePipeline
from app.services.chat_helpers import build_sources
from app.services.document_service import list_documents as svc_list_documents

logger = logging.getLogger(__name__)

mcp = FastMCP("docmind")

# 进程内单例：KnowledgePipeline 内部含检索器实例与缓存
_pipeline: KnowledgePipeline | None = None


def _get_pipeline() -> KnowledgePipeline:
    global _pipeline
    if _pipeline is None:
        from app.services.chat_service import _get_bm25_retriever

        _pipeline = KnowledgePipeline(bm25_retriever_factory=_get_bm25_retriever)
    return _pipeline


async def _accessible_kbs(db, user: dict) -> list[KnowledgeBase]:
    """当前用户可访问的知识库（自己的 + public）"""
    user_id = user["user_id"]
    rows = await db.execute(
        select(KnowledgeBase).where(
            (KnowledgeBase.user_id == user_id) | (KnowledgeBase.visibility == "public")
        )
    )
    return list(rows.scalars().all())


async def _resolve_kb(db, user: dict, kb_name: str) -> KnowledgeBase:
    """按名称解析知识库；先精确匹配，再退化为包含匹配"""
    kbs = await _accessible_kbs(db, user)
    exact = [k for k in kbs if k.name == kb_name]
    if exact:
        return exact[0]
    fuzzy = [k for k in kbs if kb_name in k.name or k.name in kb_name]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if not fuzzy:
        names = ", ".join(k.name for k in kbs[:10])
        raise ValueError(f"未找到知识库「{kb_name}」。当前可访问：{names or '（无）'}")
    names = ", ".join(k.name for k in fuzzy[:10])
    raise ValueError(f"「{kb_name}」匹配到多个知识库，请指定完整名称：{names}")


@mcp.tool()
async def list_knowledge_bases() -> list[dict]:
    """列出当前用户可访问的知识库（含自己的与公开的）。"""
    user = require_current_user()
    async with async_session() as db:
        kbs = await _accessible_kbs(db, user)
        return [
            {
                "name": kb.name,
                "uuid": kb.uuid,
                "visibility": kb.visibility,
                "doc_count": kb.doc_count,
                "chunk_count": kb.chunk_count,
                "description": kb.description or "",
            }
            for kb in kbs
        ]


@mcp.tool()
async def list_documents(kb_name: str) -> list[dict]:
    """列出指定知识库中的文档。

    Args:
        kb_name: 知识库名称，可先用 list_knowledge_bases 查询
    """
    user = require_current_user()
    async with async_session() as db:
        kb = await _resolve_kb(db, user, kb_name)
        resp = await svc_list_documents(
            db, kb.id, user["user_id"], user["role"], page=1, page_size=100
        )
        return [
            {
                "filename": d.filename,
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                "chunk_count": d.chunk_count,
                "created_at": str(d.created_at),
            }
            for d in resp.items
        ]


@mcp.tool()
async def retrieve(question: str, kb_name: str) -> dict:
    """在指定知识库中检索与问题相关的文档片段（只检索，不生成答案）。

    适合需要自行分析原始依据的场景；若想要直接答案请用 ask。

    Args:
        question: 检索问题
        kb_name: 知识库名称
    """
    user = require_current_user()
    async with async_session() as db:
        kb = await _resolve_kb(db, user, kb_name)
        result = await _get_pipeline().execute_knowledge(
            db, question, kb.id, history_messages=[], recorder=None
        )
        sources = build_sources(result.reranked_output.results, result.doc_map)
        return {
            "kb_name": kb.name,
            "question": question,
            "chunks": [
                {
                    "doc_name": s.doc_name,
                    "content": s.content,
                    "score": s.score,
                    "section_title": s.section_title,
                    "page": s.page,
                }
                for s in sources
            ],
        }


@mcp.tool()
async def ask(question: str, kb_name: str) -> dict:
    """在指定知识库中提问并获取带引用来源的答案。

    Args:
        question: 用户问题
        kb_name: 知识库名称
    """
    from app.core.llm import chat_completion

    user = require_current_user()
    async with async_session() as db:
        kb = await _resolve_kb(db, user, kb_name)
        result = await _get_pipeline().execute_knowledge(
            db, question, kb.id, history_messages=[], recorder=None
        )

        sources = build_sources(result.reranked_output.results, result.doc_map)
        prompt = result.prompt_result

        # 无可用上下文时直接返回，避免让 LLM 凭空作答
        if not prompt.used_chunks:
            return {
                "kb_name": kb.name,
                "question": question,
                "answer": "知识库中未找到与该问题相关的内容。",
                "sources": [],
            }

        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]
        llm_result = await chat_completion(messages, model=settings.LLM_MODEL)

        return {
            "kb_name": kb.name,
            "question": question,
            "answer": llm_result.content,
            "sources": [
                {
                    "doc_name": s.doc_name,
                    "content": s.content,
                    "section_title": s.section_title,
                    "page": s.page,
                }
                for s in sources
            ],
        }


def build_streamable_app():
    """构造带身份透传的 Streamable HTTP 子应用，供 FastAPI mount 到 MCP_PATH

    注意：FastMCP 的 `streamable_http_app()` 内部端点路径默认就是 `/mcp`，
    若直接挂到 `/mcp` 会变成 `/mcp/mcp`。因此先把内部路径改成 `/`，
    由外层 mount 决定最终的对外路径。
    """
    mcp.settings.streamable_http_path = "/"
    app = mcp.streamable_http_app()
    app.add_middleware(MCPUserContextMiddleware)
    return app


def mcp_session_lifespan():
    """启动 MCP session manager 的 task group

    FastAPI mount 子应用时**不会自动运行子应用的 lifespan**，若不显式启动，
    首次请求会抛 `RuntimeError: Task group is not initialized`。
    该调用必须在 `streamable_http_app()` 之后（session_manager 是懒创建的）。
    """
    return mcp.session_manager.run()
