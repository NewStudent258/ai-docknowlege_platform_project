"""重排消融实验 — 在统一评测集上对比 5 组检索策略。

与 eval_retrieval.py 的区别
---------------------------
`eval_retrieval.py` 只评测「向量 / BM25 / RRF」三路，**不接入粗排与精排**，
因此无法回答「Rerank 到底有没有用」。本脚本把粗排（CoarseRank）与精排
（DashScope qwen3-rerank）接进评测链路，做真正的消融对比：

    A. vector_only   — 仅向量
    B. bm25_only     — 仅 BM25
    C. rrf           — 向量 + BM25 → RRF 融合
    D. rrf_coarse    — RRF → 粗排（向量相似度过滤 + top_k 截断）
    E. rrf_rerank    — RRF → 精排（qwen3-rerank）        ← 本项目线上链路
    F. full          — RRF → 粗排 → 精排                  ← 本项目线上链路（完整）

指标（对齐业界检索评测惯例）
---------------------------
- Recall@5 / Recall@10：期望文档是否被召回
- Hit@1 / Hit@3：**最相关文档是否排在第 1 / 前 3 位**（排序质量的直接体现）
- MRR：第一个相关结果的倒数排名
- nDCG@5：考虑相关性等级的折损累计增益
- Precision@5：top-5 中来自期望文档的比例

用法：
  cd backend
  python tests/eval/eval_ablation.py --kb-id 17                    # 跑全部 6 组
  python tests/eval/eval_ablation.py --kb-id 17 --strategies rrf,rrf_rerank
  python tests/eval/eval_ablation.py --kb-id 17 --limit 20         # 快速试跑
  python tests/eval/eval_ablation.py --kb-id 17 --output md        # 导出 Markdown 报告
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select

from app.core.chroma_client import get_vector_store
from app.core.database import async_session
from app.core.redis_client import get_async_redis
from app.models.document import Document
from app.rag.bm25 import BM25Retriever
from app.rag.coarse_ranker import CoarseRanker
from app.rag.fusion import rrf_fusion
from app.rag.reranker import DashScopeReranker
from app.rag.retriever import RetrievalOutput, VectorRetriever
from tests.eval.eval_test_set_v2 import EVAL_TEST_SET_V2

logger = logging.getLogger(__name__)

# 关闭 SQLAlchemy 逐条 SQL 日志
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
# jieba 首次加载词典会打大量 DEBUG 日志，干扰评测输出
logging.getLogger("jieba").setLevel(logging.WARNING)


# ============================================================================
# 指标
# ============================================================================

def _dcg(gains: list[float]) -> float:
    """折损累计增益。"""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(ranked_relevant: list[bool], k: int) -> float:
    """nDCG@k（二值相关性：命中期望文档记 1）。

    Args:
        ranked_relevant: 按排名顺序排列的「是否相关」列表
        k: 截断位置
    """
    gains = [1.0 if r else 0.0 for r in ranked_relevant[:k]]
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


@dataclass
class CaseResult:
    """单题单策略的评估结果"""
    question_id: int
    question: str
    expected_docs: list[str]
    hit_1: float
    hit_3: float
    recall_5: float
    recall_10: float
    mrr: float
    ndcg_5: float
    precision_5: float
    precision_5_doc: float
    first_rank: int | None
    latency_ms: float


@dataclass
class StrategySummary:
    """单策略汇总"""
    name: str
    label: str
    results: list[CaseResult] = field(default_factory=list)
    hit_1: float = 0.0
    hit_3: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0
    mrr: float = 0.0
    ndcg_5: float = 0.0
    precision_5: float = 0.0
    precision_5_doc: float = 0.0
    avg_latency_ms: float = 0.0
    errors: int = 0


# ============================================================================
# 评估器
# ============================================================================

class AblationEvaluator:
    def __init__(self, kb_id: int, top_k: int = 10):
        self.kb_id = kb_id
        self.top_k = top_k
        self._vector_retriever: VectorRetriever | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._reranker: DashScopeReranker | None = None
        self._coarse_ranker: CoarseRanker | None = None
        self._name_to_id: dict[str, int] = {}

    @property
    def vector_retriever(self) -> VectorRetriever:
        if self._vector_retriever is None:
            self._vector_retriever = VectorRetriever(get_vector_store())
        return self._vector_retriever

    async def _bm25(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            redis = await get_async_redis()
            self._bm25_retriever = BM25Retriever(
                async_redis=redis, session_factory=async_session,
            )
        return self._bm25_retriever

    @property
    def reranker(self) -> DashScopeReranker:
        if self._reranker is None:
            self._reranker = DashScopeReranker()
        return self._reranker

    @property
    def coarse_ranker(self) -> CoarseRanker:
        if self._coarse_ranker is None:
            self._coarse_ranker = CoarseRanker()
        return self._coarse_ranker

    async def load_doc_map(self) -> None:
        async with async_session() as db:
            rows = (
                await db.execute(
                    select(Document.filename, Document.id).where(
                        Document.kb_id == self.kb_id
                    )
                )
            ).all()
        self._name_to_id = {f: i for f, i in rows}

    # ---------------- 各策略的检索实现 ----------------

    async def _s_vector_only(self, q: str) -> RetrievalOutput:
        return await self.vector_retriever.search(q, self.kb_id, top_k=self.top_k)

    async def _s_bm25_only(self, q: str) -> RetrievalOutput:
        bm25 = await self._bm25()
        return await bm25.search(q, self.kb_id, top_k=self.top_k)

    async def _fuse(self, q: str) -> RetrievalOutput:
        vout = await self.vector_retriever.search(q, self.kb_id, top_k=self.top_k)
        bm25 = await self._bm25()
        bout = await bm25.search(q, self.kb_id, top_k=self.top_k)
        return rrf_fusion(vout, bout)

    async def _s_rrf(self, q: str) -> RetrievalOutput:
        return await self._fuse(q)

    async def _s_rrf_coarse(self, q: str) -> RetrievalOutput:
        vout = await self.vector_retriever.search(q, self.kb_id, top_k=self.top_k)
        bm25 = await self._bm25()
        bout = await bm25.search(q, self.kb_id, top_k=self.top_k)
        fused = rrf_fusion(vout, bout)
        # query_embedding 由 VectorRetriever 产出、**不随 RRF 融合传递**，
        # 因此必须从 vout 取（与 knowledge_pipeline.py 的线上实现保持一致）
        if vout.query_embedding is None:
            return fused
        return self.coarse_ranker.rank(vout.query_embedding, fused)

    async def _s_rrf_rerank(self, q: str) -> RetrievalOutput:
        fused = await self._fuse(q)
        return await self.reranker.rerank(q, fused, top_k=self.top_k)

    async def _s_full(self, q: str) -> RetrievalOutput:
        vout = await self.vector_retriever.search(q, self.kb_id, top_k=self.top_k)
        bm25 = await self._bm25()
        bout = await bm25.search(q, self.kb_id, top_k=self.top_k)
        fused = rrf_fusion(vout, bout)

        if vout.query_embedding is not None:
            fused = self.coarse_ranker.rank(vout.query_embedding, fused)
        return await self.reranker.rerank(q, fused, top_k=self.top_k)

    # ---------------- 单题指标 ----------------

    def _compute(self, item: dict, out: RetrievalOutput, latency_ms: float) -> CaseResult:
        expected_ids = {
            self._name_to_id[n] for n in item["expected_docs"] if n in self._name_to_id
        }
        ranked = [r.doc_id for r in out.results]
        relevant = [d in expected_ids for d in ranked]

        oos = item["difficulty"] == "out-of-scope"

        if oos or not expected_ids:
            # 超范围题：无期望文档 → 召回类指标记 1.0（正确行为是不返回相关内容）
            # 但排序类指标（Hit@1）在无 ground truth 时不参与主指标
            return CaseResult(
                question_id=item["id"], question=item["question"],
                expected_docs=[], hit_1=1.0, hit_3=1.0, recall_5=1.0, recall_10=1.0,
                mrr=1.0, ndcg_5=1.0, precision_5=0.0, precision_5_doc=0.0,
                first_rank=None,
                latency_ms=latency_ms,
            )

        # Hit@1 / Hit@3
        hit_1 = 1.0 if relevant and relevant[0] else 0.0
        hit_3 = 1.0 if any(relevant[:3]) else 0.0

        top5 = set(ranked[:5])
        top10 = set(ranked[:10])
        recall_5 = len(top5 & expected_ids) / len(expected_ids)
        recall_10 = len(top10 & expected_ids) / len(expected_ids)

        first_rank = next((i + 1 for i, r in enumerate(relevant) if r), None)
        mrr = 1.0 / first_rank if first_rank else 0.0

        ndcg_5 = _ndcg_at_k(relevant, 5)

        # Precision@5：top-5 中来自期望文档的 chunk 占比
        # 注意分母用实际返回条数（而非硬编码 5），否则在候选取不满时会被系统性低估
        top5_list = ranked[:5]
        precision_5 = (
            len(top5 & expected_ids) / len(top5_list) if top5_list else 0.0
        )
        # Precision@5（文档级）：top-5 命中的期望文档数 / 期望文档总数，上限 1.0
        # 与 doc-level 的 Recall 口径一致，避免 chunk 密度差异导致的量纲错配
        docs_in_top5 = {d for d in top5 if d in expected_ids}
        precision_5_doc = (
            len(docs_in_top5) / len(expected_ids) if expected_ids else 0.0
        )

        return CaseResult(
            question_id=item["id"], question=item["question"],
            expected_docs=list(item["expected_docs"]),
            hit_1=hit_1, hit_3=hit_3, recall_5=recall_5, recall_10=recall_10,
            mrr=mrr, ndcg_5=ndcg_5, precision_5=precision_5,
            precision_5_doc=precision_5_doc,
            first_rank=first_rank, latency_ms=latency_ms,
        )

    async def evaluate_strategy(
        self, key: str, label: str, fn, test_set: list[dict],
    ) -> StrategySummary:
        summary = StrategySummary(name=key, label=label)
        for i, item in enumerate(test_set, start=1):
            t0 = time.perf_counter()
            try:
                out = await fn(item["question"])
            except Exception as e:
                logger.warning("策略 %s 在 Q%d 异常: %s: %s", key, item["id"], type(e).__name__, e)
                out = RetrievalOutput()
                summary.errors += 1
            dt = (time.perf_counter() - t0) * 1000

            summary.results.append(self._compute(item, out, dt))
            if i % 10 == 0:
                print(f"      ... {i}/{len(test_set)}", flush=True)

        # 汇总：排除 out-of-scope（无 ground truth，参与会虚高）
        scoped = [
            r for r, it in zip(summary.results, test_set)
            if it["difficulty"] != "out-of-scope"
        ]
        if scoped:
            summary.hit_1 = mean(r.hit_1 for r in scoped)
            summary.hit_3 = mean(r.hit_3 for r in scoped)
            summary.recall_5 = mean(r.recall_5 for r in scoped)
            summary.recall_10 = mean(r.recall_10 for r in scoped)
            summary.mrr = mean(r.mrr for r in scoped)
            summary.ndcg_5 = mean(r.ndcg_5 for r in scoped)
            summary.precision_5 = mean(r.precision_5 for r in scoped)
            summary.precision_5_doc = mean(r.precision_5_doc for r in scoped)
            summary.avg_latency_ms = mean(r.latency_ms for r in scoped)
        return summary


STRATEGIES: list[tuple[str, str, str]] = [
    ("vector_only", "仅向量", "_s_vector_only"),
    ("bm25_only", "仅 BM25", "_s_bm25_only"),
    ("rrf", "RRF 融合（向量+BM25）", "_s_rrf"),
    ("rrf_coarse", "RRF + 粗排", "_s_rrf_coarse"),
    ("rrf_rerank", "RRF + 精排(Rerank)", "_s_rrf_rerank"),
    ("full", "RRF + 粗排 + 精排（线上）", "_s_full"),
]


# ============================================================================
# 报告
# ============================================================================

def print_summary(summaries: list[StrategySummary], n_questions: int) -> None:
    print(f"\n{'='*104}")
    print(f"  重排消融实验汇总（{n_questions} 题，排除 out-of-scope）")
    print(f"{'='*104}\n")

    hdr = (
        f"{'策略':<30} {'Hit@1':>8} {'Hit@3':>8} {'MRR':>8} {'nDCG@5':>8} "
        f"{'R@5':>8} {'R@10':>8} {'P@5doc':>8} {'ms':>8}"
    )
    print(hdr)
    print("-" * 104)
    for s in summaries:
        print(
            f"{s.label:<30} {s.hit_1:>8.3f} {s.hit_3:>8.3f} {s.mrr:>8.3f} "
            f"{s.ndcg_5:>8.3f} {s.recall_5:>8.3f} {s.recall_10:>8.3f} "
            f"{s.precision_5_doc:>8.3f} {s.avg_latency_ms:>8.0f}"
        )
    print("-" * 104)

    # 相对提升（以 RRF 为基线）
    base = next((s for s in summaries if s.name == "rrf"), None)
    if base:
        print(f"\n  相对 RRF 基线的提升：")
        for s in summaries:
            if s.name == "rrf":
                continue
            d_h1 = (s.hit_1 - base.hit_1) * 100
            d_mrr = s.mrr - base.mrr
            d_nd = s.ndcg_5 - base.ndcg_5
            print(
                f"    {s.label:<30} Hit@1 {d_h1:+.1f}pp   MRR {d_mrr:+.3f}   nDCG@5 {d_nd:+.3f}"
            )
    print()


def build_markdown(summaries: list[StrategySummary], kb_id: int, n_questions: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    lines = [
        "# 重排消融实验报告",
        "",
        f"**生成时间**：{now}",
        f"**知识库**：kb_id={kb_id}",
        f"**评测题数**：{n_questions}（排除 out-of-scope 后参与指标计算）",
        "",
        "## 指标汇总",
        "",
        "> Precision@5 采用**文档级**口径（top-5 命中的期望文档数 / 期望文档总数）。",
        "> 单份文档约含 8 个 chunk，若用 chunk 级口径，P@5 会被分块密度系统性压低，",
        "> 且与其余文档级指标量纲不一致，无法横向比较。",
        "",
        "| 策略 | Hit@1 | Hit@3 | MRR | nDCG@5 | Recall@5 | Recall@10 | Precision@5(doc) | 平均耗时(ms) |",
        "|:---|-----:|------:|----:|-------:|---------:|----------:|-----------------:|-------------:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.hit_1:.4f} | {s.hit_3:.4f} | {s.mrr:.4f} | "
            f"{s.ndcg_5:.4f} | {s.recall_5:.4f} | {s.recall_10:.4f} | "
            f"{s.precision_5_doc:.4f} | {s.avg_latency_ms:.0f} |"
        )

    base = next((s for s in summaries if s.name == "rrf"), None)
    if base:
        lines += [
            "",
            "## 关键结论（相对 RRF 基线）",
            "",
            "| 策略 | ΔHit@1 | ΔMRR | ΔnDCG@5 |",
            "|:---|-------:|-----:|--------:|",
        ]
        for s in summaries:
            if s.name == "rrf":
                continue
            lines.append(
                f"| {s.label} | {(s.hit_1 - base.hit_1) * 100:+.1f}pp | "
                f"{s.mrr - base.mrr:+.4f} | {s.ndcg_5 - base.ndcg_5:+.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

async def main_async(
    kb_id: int, top_k: int, only: list[str] | None,
    limit: int | None, output: str | None,
) -> None:
    test_set = EVAL_TEST_SET_V2
    if limit:
        test_set = test_set[:limit]

    evaluator = AblationEvaluator(kb_id=kb_id, top_k=top_k)
    await evaluator.load_doc_map()

    n_docs = len(evaluator._name_to_id)
    scoped_n = sum(1 for it in test_set if it["difficulty"] != "out-of-scope")

    print(f"\n{'='*104}")
    print(f"  重排消融实验 — kb_id={kb_id}, top_k={top_k}")
    print(f"  知识库文档数: {n_docs}   评测题数: {len(test_set)}（参与指标 {scoped_n}）")
    print(f"{'='*104}")

    # 检查期望文档是否都在库中
    missing = set()
    for it in test_set:
        for d in it["expected_docs"]:
            if d not in evaluator._name_to_id:
                missing.add(d)
    if missing:
        print(f"\n⚠️  以下期望文档不在知识库中（将导致召回为 0）：")
        for m in sorted(missing):
            print(f"    - {m}")

    summaries: list[StrategySummary] = []
    for key, label, method_name in STRATEGIES:
        if only and key not in only:
            continue
        print(f"\n  ▶ 策略: {label} ({key})")
        fn = getattr(evaluator, method_name)
        s = await evaluator.evaluate_strategy(key, label, fn, test_set)
        summaries.append(s)
        print(
            f"    ✅ Hit@1={s.hit_1:.3f} Hit@3={s.hit_3:.3f} MRR={s.mrr:.3f} "
            f"nDCG@5={s.ndcg_5:.3f} R@5={s.recall_5:.3f} ({s.avg_latency_ms:.0f}ms)"
        )

    print_summary(summaries, scoped_n)

    if output == "md":
        md = build_markdown(summaries, kb_id, scoped_n)
        out_path = _BACKEND_ROOT / "ablation_eval.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"📄 Markdown 报告已写入: {out_path}")

    # JSON 明细（便于后续分析）
    detail = {
        "kb_id": kb_id,
        "top_k": top_k,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(test_set),
        "summaries": [
            {
                "name": s.name, "label": s.label,
                "hit_1": s.hit_1, "hit_3": s.hit_3, "mrr": s.mrr,
                "ndcg_5": s.ndcg_5, "recall_5": s.recall_5,
                "recall_10": s.recall_10, "precision_5": s.precision_5,
                "precision_5_doc": s.precision_5_doc,
                "avg_latency_ms": s.avg_latency_ms, "errors": s.errors,
                "per_case": [
                    {
                        "qid": r.question_id, "question": r.question,
                        "expected": r.expected_docs, "hit_1": r.hit_1,
                        "mrr": r.mrr, "ndcg_5": r.ndcg_5, "first_rank": r.first_rank,
                    }
                    for r in s.results
                ],
            }
            for s in summaries
        ],
    }
    json_path = _BACKEND_ROOT / "ablation_eval.json"
    json_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"📄 JSON 明细已写入: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重排消融实验")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（试跑）")
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="逗号分隔的策略 key，如 rrf,rrf_rerank",
    )
    parser.add_argument("--output", type=str, default=None, choices=["md"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    only = args.strategies.split(",") if args.strategies else None

    asyncio.run(main_async(
        kb_id=args.kb_id, top_k=args.top_k, only=only,
        limit=args.limit, output=args.output,
    ))


if __name__ == "__main__":
    main()
