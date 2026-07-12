# ADR-026: MCP Server 暴露知识库能力

- **日期**: 2026-09-21
- **状态**: 已采纳
- **决策者**: 架构评审

## 背景

DocMind 的知识库检索与问答能力目前只能通过自身前端（Vue 3）使用。用户若希望在 Claude Desktop、Cursor 等支持 MCP（Model Context Protocol）的 AI 客户端中直接查询企业知识库，需要一套标准化的对外接口。

MCP 是 Anthropic 提出的开放协议，用于把工具与数据源以标准方式暴露给 LLM 客户端。两个可能的方向：

- **MCP Server**：把 DocMind 的能力暴露出去，外部 AI 客户端接入。
- **MCP Client**：让 DocMind 的问答链路能调用外部 MCP 工具。

**本次只做 Server 方向**——DocMind 已有成熟的知识库能力值得输出，而 Client 方向需要改造 chat 链路加入工具调用循环，复杂度高且依赖外部 MCP 服务配合演示。

## 决策

### 1. 独立挂载于 `/mcp`，复用现有鉴权中间件

MCP Server 不新建服务，而是挂载进现有 FastAPI 应用：`app.mount(settings.MCP_PATH, build_streamable_app())`，由 `MCP_ENABLED` 控制开关。

**鉴权分层**：

1. **401 门控**由现有 `AuthMiddleware` 完成。它是纯 ASGI 中间件、对非白名单路径一律强制 JWT，而 `/mcp` 不在 `_PUBLIC_PATHS` 中，因此自动被覆盖——无需为 MCP 单独写鉴权逻辑。
2. **身份透传**由 `app/mcp/auth.py` 的 `MCPUserContextMiddleware` 完成。`AuthMiddleware` 把用户信息写入 `request.state`（底层是 ASGI `scope["state"]` 共享字典），但挂载的 MCP 子应用拿到的是另一个 Request 实例。因此在子应用外包一层薄 ASGI 中间件，从 `scope["state"]` 读出身份存入 contextvar，工具函数通过 `require_current_user()` 读取。

这样只需一次 JWT 解码，且不侵入现有中间件。

### 2. 提供 4 个工具

| 工具 | 用途 | 复用 |
|:---|:---|:---|
| `list_knowledge_bases` | 列出当前用户可访问的知识库 | `KnowledgeBase` 表查询（自己的 + public） |
| `list_documents` | 列出指定知识库的文档 | `document_service.list_documents` |
| `retrieve` | 只检索，返回引用片段（不生成） | `KnowledgePipeline.execute_knowledge` |
| `ask` | 完整问答，返回答案 + 来源 | `execute_knowledge` + `core.llm.chat_completion` |

知识库通过**名称**而非 UUID 定位（对 LLM 更友好）：先精确匹配，再退化为包含匹配，多个匹配时返回候选列表让 LLM 澄清。

### 3. `ask` 绕过 SSE，走非流式 LLM

`chat_service.chat()` 返回 `StreamingResponse`（SSE），MCP 需要的是结构化结果。因此 `ask` 直接复用管线产出：

```
execute_knowledge() → reranked_output + prompt_result + doc_map
    ↓
messages = [system_prompt, user_prompt]
    ↓
core.llm.chat_completion(messages) → LLMResult
    ↓
{answer, sources}
```

复用同一套 `KnowledgePipeline`，保证 MCP 与 Web 端的检索行为一致（同样的三路融合、粗排、精排、证据审计）。

**空上下文保护**：若 `prompt_result.used_chunks` 为空，直接返回「知识库中未找到相关内容」，不调用 LLM——避免模型凭空作答。

### 4. 锁定 MCP SDK v1

`mcp` 锁 `>=1.28,<2`。v2 已将 `FastMCP` 改名为 `MCPServer` 且 handler 注册方式变更为构造参数，API 不兼容。

### 5. 传递依赖 `sse-starlette` 必须锁 `<3`

**这是实现阶段踩到的坑。** `mcp` 依赖 `sse-starlette>=1.6.1`，pip 会解析到最新的 3.x，而 `sse-starlette 3.x` 要求 `starlette>=0.49.1`，会把 starlette 顶到 1.x —— 但 `fastapi 0.115` 要求 `starlette<0.47.0`，于是应用启动时报：

```
TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'
```

因此在 `requirements.txt` 中显式追加 `sse-starlette<3`。

## 后果

### 正面

- 知识库能力可被任意 MCP 客户端复用，无需改动前端。
- 鉴权、限流、Request ID 等中间件能力自动生效，无需重复实现。
- 与 Web 端共用同一检索管线，行为一致、无逻辑分叉。
- 单进程部署，无额外运维成本。

### 负面与约束

- **MCP 客户端必须携带 JWT**：`/mcp` 被 `AuthMiddleware` 拦截，客户端需在 `Authorization` 头中提供有效 token。token 过期（15 分钟）后需重新获取，这对交互式客户端体验不友好——后续可考虑为 MCP 单独签发长效 token。
- **`retrieve` / `ask` 无多轮上下文**：当前实现传 `history_messages=[]`，MCP 侧的多轮由客户端自行维护（客户端把历史拼进 question）。
- **限流共用**：`/mcp` 走 `RateLimitMiddleware` 的 default 分组（120/分钟），高频调用可能受限。

### 风险与回退

| 风险 | 回退 |
|:---|:---|
| MCP SDK v2 API 漂移 | 已锁 `mcp>=1.28,<2` |
| starlette 版本冲突 | 已锁 `sse-starlette<3` 并注释原因 |
| 挂载子应用与中间件执行顺序问题 | 已验证 `AuthMiddleware` 为纯 ASGI 且覆盖 `/mcp`；若客户端无法带 JWT，备选方案是把 `/mcp` 加入白名单并在工具内自行 `decode_access_token` |
| 工具调用阻塞事件循环 | `retrieve`/`ask` 走 async 管线；LightRAG 部分已做线程卸载（见 ADR-025） |

## 相关

- [ADR-017: SSE 流 DB 会话生命周期解耦](ADR-017-SSE流DB会话生命周期解耦.md) — 为何 `ask` 不走 SSE
- [ADR-022: 问答 Service 层三模块拆分](ADR-022-问答Service层三模块拆分.md) — 复用的服务层结构
- [ADR-025: LightRAG 第三路检索](ADR-025-LightRAG第三路检索.md)
