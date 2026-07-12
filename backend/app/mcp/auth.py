"""MCP 鉴权桥 — 把 AuthMiddleware 校验出的用户身份透传给 MCP 工具（ADR-026）

鉴权分两层：
1. **401 门控**由现有 `AuthMiddleware` 完成 —— `/mcp` 不在其 `_PUBLIC_PATHS` 白名单中，
   因此所有 MCP 请求都必须携带有效 `Authorization: Bearer <jwt>`，否则直接 401。
2. **身份透传**由本模块完成 —— AuthMiddleware 把用户信息写入 `request.state`
   （底层即 ASGI `scope["state"]` 共享字典），但挂载的 MCP 子应用拿到的是另一个
   Request 实例，因此在此包一层 ASGI 中间件，从 `scope["state"]` 读出身份存入
   contextvar，供工具函数读取。
"""

import contextvars
import logging

logger = logging.getLogger(__name__)

_current_user: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "mcp_current_user", default=None
)


def set_current_user(user: dict | None):
    """写入当前用户身份（返回 token 供 reset）"""
    return _current_user.set(user)


def get_current_user() -> dict | None:
    """读取当前用户身份：{"user_id": int, "username": str, "role": str}"""
    return _current_user.get()


def require_current_user() -> dict:
    """取当前用户，缺失则抛错（工具内调用）"""
    user = get_current_user()
    if not user or "user_id" not in user:
        raise PermissionError("未认证：缺少有效的访问令牌")
    return user


class MCPUserContextMiddleware:
    """纯 ASGI 中间件 — 从 scope['state'] 提取 AuthMiddleware 写入的用户身份"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        state = scope.get("state") or {}
        user = None
        if "user_id" in state:
            user = {
                "user_id": state["user_id"],
                "username": state.get("username"),
                "role": state.get("role"),
            }

        token = set_current_user(user)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user.reset(token)
