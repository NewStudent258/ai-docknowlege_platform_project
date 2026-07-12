"""MCP Server 单元测试（ADR-026）

覆盖：
- 工具注册齐全
- 鉴权上下文（contextvar 透传）
- 知识库名称解析（精确 / 包含 / 未找到 / 歧义）
- 工具返回结构

工具函数通过 FastMCP 装饰器注册，测试时直接调用其底层函数
（`mcp._tool_manager._tools[name].fn`），避免走协议层。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _tool(name: str):
    """取出 FastMCP 注册的工具函数本体"""
    from app.mcp.server import mcp

    return mcp._tool_manager._tools[name].fn


def _kb(kb_id=1, name="公司制度汇编", visibility="public", doc_count=100):
    kb = MagicMock()
    kb.id = kb_id
    kb.uuid = f"uuid-{kb_id}"
    kb.name = name
    kb.visibility = visibility
    kb.doc_count = doc_count
    kb.chunk_count = doc_count
    kb.description = "描述"
    return kb


class TestToolRegistration:
    def test_四个工具均已注册(self):
        from app.mcp.server import mcp

        names = set(mcp._tool_manager._tools)
        assert names == {"list_knowledge_bases", "list_documents", "retrieve", "ask"}


class TestAuthContext:
    """身份由 AuthMiddleware 写入 scope['state']，经薄中间件存入 contextvar"""

    def test_未认证时_require_current_user_抛错(self):
        from app.mcp.auth import require_current_user, set_current_user

        token = set_current_user(None)
        try:
            with pytest.raises(PermissionError):
                require_current_user()
        finally:
            from app.mcp.auth import _current_user

            _current_user.reset(token)

    def test_认证后可读取身份(self):
        from app.mcp.auth import _current_user, get_current_user, set_current_user

        token = set_current_user({"user_id": 7, "username": "u", "role": "user"})
        try:
            assert get_current_user()["user_id"] == 7
        finally:
            _current_user.reset(token)

    @pytest.mark.asyncio
    async def test_中间件从_scope_state_提取身份(self):
        from app.mcp.auth import MCPUserContextMiddleware, get_current_user

        captured = {}

        async def _app(scope, receive, send):
            captured.update(get_current_user() or {})

        mw = MCPUserContextMiddleware(_app)
        scope = {
            "type": "http",
            "state": {"user_id": 42, "username": "alice", "role": "admin"},
        }
        await mw(scope, None, None)

        assert captured["user_id"] == 42
        assert captured["role"] == "admin"

    @pytest.mark.asyncio
    async def test_中间件无身份时传_None(self):
        from app.mcp.auth import MCPUserContextMiddleware, get_current_user

        captured = {"user_id": "未设置"}

        async def _app(scope, receive, send):
            captured["v"] = get_current_user()

        mw = MCPUserContextMiddleware(_app)
        await mw({"type": "http", "state": {}}, None, None)

        assert captured["v"] is None


class TestResolveKb:
    """知识库名称解析"""

    @pytest.mark.asyncio
    async def test_精确匹配(self):
        from app.mcp.server import _resolve_kb

        user = {"user_id": 1, "role": "user"}
        with patch("app.mcp.server._accessible_kbs", AsyncMock(return_value=[_kb(1, "甲"), _kb(2, "乙")])):
            kb = await _resolve_kb(MagicMock(), user, "乙")
        assert kb.id == 2

    @pytest.mark.asyncio
    async def test_包含匹配(self):
        from app.mcp.server import _resolve_kb

        user = {"user_id": 1, "role": "user"}
        with patch("app.mcp.server._accessible_kbs", AsyncMock(return_value=[_kb(1, "公司制度汇编（100 份）")])):
            kb = await _resolve_kb(MagicMock(), user, "制度汇编")
        assert kb.id == 1

    @pytest.mark.asyncio
    async def test_未找到时抛出含候选列表的错误(self):
        from app.mcp.server import _resolve_kb

        user = {"user_id": 1, "role": "user"}
        with patch("app.mcp.server._accessible_kbs", AsyncMock(return_value=[_kb(1, "甲")])):
            with pytest.raises(ValueError, match="未找到知识库"):
                await _resolve_kb(MagicMock(), user, "不存在的库")

    @pytest.mark.asyncio
    async def test_多个包含匹配时要求澄清(self):
        from app.mcp.server import _resolve_kb

        user = {"user_id": 1, "role": "user"}
        kbs = [_kb(1, "制度汇编A"), _kb(2, "制度汇编B")]
        with patch("app.mcp.server._accessible_kbs", AsyncMock(return_value=kbs)):
            with pytest.raises(ValueError, match="匹配到多个"):
                await _resolve_kb(MagicMock(), user, "制度汇编")


class TestTools:
    @pytest.mark.asyncio
    async def test_list_knowledge_bases_返回结构(self):
        user = {"user_id": 1, "role": "user"}
        with patch("app.mcp.server.require_current_user", return_value=user), patch(
            "app.mcp.server._accessible_kbs", AsyncMock(return_value=[_kb()])
        ), patch("app.mcp.server.async_session") as sess:
            sess.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            sess.return_value.__aexit__ = AsyncMock(return_value=False)
            out = await _tool("list_knowledge_bases")()

        assert len(out) == 1
        assert out[0]["name"] == "公司制度汇编"
        assert out[0]["visibility"] == "public"
        assert out[0]["doc_count"] == 100

    @pytest.mark.asyncio
    async def test_list_documents_状态序列化为字符串(self):
        from app.models.enums import DocumentStatus

        doc = MagicMock()
        doc.filename = "a.txt"
        doc.status = DocumentStatus.COMPLETED
        doc.chunk_count = 3
        doc.created_at = "2026-01-01"

        resp = MagicMock()
        resp.items = [doc]

        user = {"user_id": 1, "role": "user"}
        with patch("app.mcp.server.require_current_user", return_value=user), patch(
            "app.mcp.server._resolve_kb", AsyncMock(return_value=_kb())
        ), patch("app.mcp.server.svc_list_documents", AsyncMock(return_value=resp)), patch(
            "app.mcp.server.async_session"
        ) as sess:
            sess.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            sess.return_value.__aexit__ = AsyncMock(return_value=False)
            out = await _tool("list_documents")("公司制度汇编")

        assert out[0]["filename"] == "a.txt"
        assert isinstance(out[0]["status"], str)

    @pytest.mark.asyncio
    async def test_ask_无可用上下文时不调用_LLM(self):
        """检索为空时直接返回提示，避免模型凭空作答"""
        prompt = MagicMock()
        prompt.used_chunks = []
        result = MagicMock()
        result.prompt_result = prompt
        result.reranked_output.results = []
        result.doc_map = {}

        user = {"user_id": 1, "role": "user"}
        mock_llm = AsyncMock()
        with patch("app.mcp.server.require_current_user", return_value=user), patch(
            "app.mcp.server._resolve_kb", AsyncMock(return_value=_kb())
        ), patch("app.mcp.server._get_pipeline") as pipe, patch(
            "app.mcp.server.build_sources", return_value=[]
        ), patch("app.core.llm.chat_completion", mock_llm), patch(
            "app.mcp.server.async_session"
        ) as sess:
            pipe.return_value.execute_knowledge = AsyncMock(return_value=result)
            sess.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            sess.return_value.__aexit__ = AsyncMock(return_value=False)
            out = await _tool("ask")("随便问问", "公司制度汇编")

        mock_llm.assert_not_called()
        assert out["answer"] == "知识库中未找到与该问题相关的内容。"
        assert out["sources"] == []
