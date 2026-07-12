# ADR-025: LightRAG 图谱检索作为第三路召回

- **日期**: 2026-09-21
- **状态**: 已采纳
- **决策者**: 架构评审

## 背景

现有检索链路为**双路召回**：向量语义检索（ChromaDB，top_k=10）+ BM25 关键词检索（rank-bm25 + jieba，top_k=10），经 RRF（k=60）融合后由 CoarseRank 粗排、qwen3-rerank 精排。

双路的固有局限：

1. **向量检索擅长语义相似，但丢失结构关系**——问「A 部门和 B 部门在采购流程上如何衔接」时，答案分散在多篇文档的多个段落，单靠向量相似度难以把跨文档的实体关系拼起来。
2. **BM25 依赖字面匹配**——用户用词与文档用词不一致时（如问「请假条」而文档写「请假申请单」）召回不足。
3. **两者都以「chunk 相似度」为单位**，没有「实体—关系」这一层的抽象，无法回答需要多跳推理的问题。

LightRAG（HKUDS）是图谱增强 RAG 框架：入库时用 LLM 抽取实体与关系构建知识图谱，查询时支持 local（实体+一跳关系）/ global（关系+社区摘要）/ hybrid / mix（图谱+向量）等模式。

## 决策

### 1. LightRAG 作为第三路召回，与向量、BM25 经 RRF 三路融合

```
向量检索 ─┐
BM25 检索 ─┼→ RRF 融合 → CoarseRank 粗排 → qwen3-rerank 精排 → ...
LightRAG ─┘   （新增第三路）
```

不改动现有两路与下游逻辑：`rrf_fusion(*retrieval_outputs)` 本就是可变参数设计，第三路只需返回标准的 `RetrievalOutput`，融合去重 key 仍为 `(doc_id, chunk_index)`。

**可开关**：`LIGHTRAG_ENABLED=false` 时管线完全退回双路，对现有链路零影响。

### 2. 按 KB 隔离图谱存储

LightRAG 工作目录按 KB 分：`lightrag_data/{kb_id}/`，与 ChromaDB 的 per-KB collection 策略对齐。删除知识库时整体删除该目录。

### 3. 结果映射采用 content 精确匹配（而非解析 id）

**这是实现阶段对初始设计的重要修正。**

初始设计假设可以从 LightRAG 返回的 chunk 里解析出我们插入时传入的 id，从而还原 `(doc_id, chunk_index)`。实测发现不可行：

- `aquery` 返回的是**字符串**，不是结构化对象；要拿结构化数据必须用 **`aquery_data`**（返回 `dict`，含 `data.chunks`）。
- `aquery_data` 返回的每条 chunk 只有 `{reference_id, content, file_path, chunk_id}`，**不含 `full_doc_id`**。
- `chunk_id` 由 LightRAG 内部哈希生成，不受传入 `ids` 控制；`ainsert_custom_chunks` 虽可传入切好的 chunks，但已标注 deprecated，且 chunk id 仍按 `(doc_id, chunk_content)` 哈希。

因此改为：**插入时传入 MySQL 中 chunk 的原文，检索后按 `content` 精确反查 `(doc_id, chunk_index)`**。索引来自 `chunks` 表（join `documents` 取文件名），按 KB 缓存（TTL 60s）。该方案不依赖 LightRAG 的 id 方案，对版本升级更稳健。

### 4. 查询模式用 `mix` + `only_need_context=True`

- `mix`：图谱检索与向量检索合并，召回面最广。
- `only_need_context=True`：只取上下文与结构化数据，跳过查询期的 LLM 生成——避免每次检索都产生一次 LLM 调用（`global` 模式的重关系摘要尤其昂贵）。

### 5. 执行模型：实例专属 loop + 线程卸载

LightRAG 内部含同步阻塞代码（NetworkX 图计算、本地文件 IO），直接在 FastAPI 事件循环上 await 会阻塞整个进程。

- 每个 KB 实例持有**专属事件循环**（Windows 下显式构造 `SelectorEventLoop`，不改进程级 policy，避免波及 uvicorn 主循环与 aiomysql）。
- 所有 `rag.*` 调用经 `asyncio.to_thread` 卸载到该实例的 loop 执行。
- 实例按需创建，进程内缓存（LRU 上限 `LIGHTRAG_MAX_INSTANCES=4` + 空闲 TTL 300s）。

### 6. 失败降级：图谱构建失败不阻塞文档入库

图谱构建需对每个 chunk 调用一次 LLM 做实体/关系抽取，是入库最慢、最可能失败的环节。因此：

- 构建失败仅记录日志，**文档状态仍为 `completed`**，不新增 `DocumentStatus` 枚举值（避免波及前端状态展示、检索过滤、alembic 迁移）。
- **不加入 `RESUMABLE_STAGES`**——失败不自动重试，避免重复消耗 LLM 抽取费用。
- 构建期间写 `.building` 标记文件，检索侧见标记则跳过该 KB，避免读到中间态。

### 7. 抽取模型用 `deepseek-flash`

实体/关系抽取是模板化任务，不需要推理能力。用便宜的 flash 模型（`LIGHTRAG_LLM_MODEL`）而非主问答模型 `deepseek-v4-pro`。

## 后果

### 正面

- 补齐「实体—关系」抽象层，为跨文档、多跳推理类问题提供召回能力。
- 三路融合后，向量与 BM25 都漏掉的 chunk 仍可能被图谱路径召回，提升召回上限。
- 完全可开关，关闭时对现有链路零影响；现有 Ragas 评测结论仍然有效。
- 内容匹配的映射方案不依赖 LightRAG 内部 id 实现，升级更安全。

### 负面与约束

- **入库成本上升**：每个 chunk 增加一次 LLM 抽取调用。当前 100 份文档 = 100 个 chunk，约 100 次 flash 调用。
- **查询延迟增加**：LightRAG 结果无 `embedding`，在 CoarseRank 中按中性分处理，排位可能靠后；但 RRF 用排名而非绝对分，只要图谱独有 chunk 在 RRF 中排名靠前即可保留。
- **图谱与向量的一致性依赖应用层维护**：删除文档需逐 chunk 调 `adelete_by_doc_id`（LightRAG 以传入 id 为文档单位），删除 KB 则整体删目录。
- **首次查询有冷启动**：实例化需加载图/向量/KV 存储，是同步阻塞 IO。

### 风险与回退

| 风险 | 回退 |
|:---|:---|
| 图谱检索阻塞事件循环 | 已用专属 loop + `to_thread` 隔离；仍异常则 `LIGHTRAG_ENABLED=false` 完全回退 |
| 图谱构建消耗过多 LLM 费用 | 抽取用 flash 模型；不自动重试；回填脚本默认只处理 10 份做验证 |
| 内容匹配失效（LightRAG 重新切分 chunk） | `LIGHTRAG_CHUNK_TOKEN_SIZE=2000` 远大于现有 chunk（约 460 字符），实测不触发二次切分；匹配失败的 chunk 直接跳过 |
| 双进程读写冲突（FastAPI 读 / Celery 写） | `.building` 标记文件做读写分离 |

## 相关

- [ADR-018: 向量存储抽象层](ADR-018-向量存储抽象层.md) — per-KB collection 的隔离策略先例
- [ADR-024: RRF 融合后粗排层](ADR-024-粗排层.md) — 三路融合后的下游处理
- [ADR-026: MCP Server](ADR-026-MCP-Server暴露知识库能力.md)
