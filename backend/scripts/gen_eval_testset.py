"""生成 100 题评测集 — 基于已入库的目标语料 chunk 内容出题。

设计要点
--------
1. **题目必须可从语料回答**：把每份文档的 chunk 内容喂给 LLM，让它基于真实正文出题，
   并要求给出「参考答案 + 该题证据所在的文档名」，避免出现无解题。
2. **题型配比**（对齐 TESTING.md §7.2 并强化难度梯度）：
   - 单文档精确查询 30 题：问具体数字/流程，答案单跳可得
   - 跨文档查询   25 题：答案分散在 2-3 份文档，需多路召回
   - 简称/口语化  15 题：用词与文档不一致，考 BM25 字面匹配短板
   - 蕴含推理     15 题：答案需由条款推导，非原文直述
   - 超出范围     15 题：语料中无答案，考拒答与幻觉抑制
3. **输出结构化**：id / question / type / difficulty / expected_docs / reference / reasoning

产出：backend/tests/eval/eval_test_set_v2.py（Python 字面量，供评测脚本 import）
      以及 backend/tests/eval/_gen_report.json（生成过程明细）

用法：
  cd backend
  python scripts/gen_eval_testset.py --dry-run      # 只生成 5 题试跑
  python scripts/gen_eval_testset.py                # 生成全部 100 题
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select

from app.core.database import async_session
from app.core.llm import chat_completion
from app.models.chunk import Chunk
from app.models.document import Document

CORPUS_KB_ID = 17  # DocMind评测库A(目标+干扰) —— 目标语料所在库
OUT_PY = _BACKEND_ROOT / "tests" / "eval" / "eval_test_set_v2.py"
OUT_JSON = _BACKEND_ROOT / "tests" / "eval" / "_gen_report.json"

# 各题型的目标数量
TYPE_PLAN = [
    ("单文档精确查询", "easy", 30),
    ("跨文档查询", "medium", 25),
    ("简称与口语化查询", "medium", 15),
    ("蕴含推理查询", "hard", 15),
    ("超出知识库范围", "out-of-scope", 15),
]


async def _load_docs_with_chunks() -> dict[str, str]:
    """加载目标语料的 文件名 → 全文（按 chunk 拼接）。"""
    async with async_session() as db:
        rows = (
            await db.execute(
                select(Document.filename, Chunk.chunk_index, Chunk.content)
                .join(Chunk, Chunk.doc_id == Document.id)
                .where(Document.kb_id == CORPUS_KB_ID)
                .order_by(Document.filename, Chunk.chunk_index)
            )
        ).all()

    docs: dict[str, list[tuple[int, str]]] = {}
    for filename, idx, content in rows:
        docs.setdefault(filename, []).append((idx, content))

    return {
        name: "\n\n".join(c for _, c in sorted(chunks))
        for name, chunks in docs.items()
    }


_SYS_ASK = """你是企业知识库评测集的出题专家。我会给你若干企业内部制度文档的正文，请你**严格基于这些正文**出题。

出题铁律：
1. 每道题的答案必须能在给定正文中找到依据，禁止出「文中没有答案」的题（除非我要求 out-of-scope）。
2. 问题要像一个真实员工会问的口语化问题，不要照抄条款原文当问题。
3. answer 必须是简明的参考答案（1-3 句，含关键数字）。
4. evidence_docs 列出**包含答案依据的文档名**（严格用我给的文档名列表中的字符串）。
5. 同一份文档不要出重复考点的题。
6. 只输出 JSON，不要任何解释文字、不要 Markdown 代码块围栏。

输出格式（JSON 数组）：
[
  {
    "question": "员工请病假要提前几天申请？",
    "type": "单文档精确查询",
    "difficulty": "easy",
    "expected_docs": ["考勤与请假管理制度.md"],
    "reference": "病假应提前1个工作日申请，突发疾病无法提前申请的应于当日上班后2小时内报告，返岗后3个工作日内补办手续。"
  }
]
"""

_SYS_TYPES = {
    "单文档精确查询": "出【单文档精确查询】题：答案集中在单一文档，问具体数字、时限、金额、流程步骤。问题直白，不设弯子。",
    "跨文档查询": "出【跨文档查询】题：答案需要**综合 2-3 份不同文档**才能完整回答（例如一份讲审批权限、另一份讲所需材料）。expected_docs 必须包含全部涉及的文档。",
    "简称与口语化查询": "出【简称与口语化查询】题：用员工日常口语或简称提问（如「工牌」「年假」「报销单」「走流程」），与文档中的正式表述不一致，用于考察关键词检索的字面匹配能力。",
    "蕴含推理查询": "出【蕴含推理】题：答案不能从原文直接摘抄，需要跨条款推导或把两条规定组合起来才能得出（例如「如果我 X 又 Y，会怎样」）。问题要有一定复杂度。",
    "超出知识库范围": "出【超出知识库范围】题：问题要**和企业制度场景贴近、听起来很合理**，但给定的这些文档中**确实没有**答案（例如问其他公司的政策、问文档未涉及的具体金额）。expected_docs 必须为空数组 []，reference 填写「知识库中未找到相关制度依据」。",
}


async def _ask_batch(
    type_name: str,
    difficulty: str,
    count: int,
    doc_names: list[str],
    doc_text: str,
    start_id: int,
) -> list[dict]:
    sys_prompt = _SYS_ASK
    type_rule = _SYS_TYPES[type_name]

    doc_list = "\n".join(f"- {n}" for n in doc_names)
    user_prompt = f"""可用的文档名清单：
{doc_list}

以下是文档正文：
=====================
{doc_text}
=====================

出题要求：{type_rule}

请出 {count} 道题，题型统一标为「{type_name}」，difficulty 统一标为「{difficulty}」。
只输出 JSON 数组。"""

    resp = await chat_completion(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        deep_thinking=False,
        max_tokens=8000,
    )
    raw = resp.content.strip()
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)

    # 容错：截取第一个 [ 到最后一个 ]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"LLM 未返回 JSON 数组: {raw[:200]}")
    items = json.loads(raw[start : end + 1])

    out = []
    for i, it in enumerate(items, start=start_id):
        out.append({
            "id": i,
            "question": str(it["question"]).strip(),
            "type": type_name,
            "difficulty": difficulty,
            "expected_docs": [str(d).strip() for d in it.get("expected_docs", [])],
            "reference": str(it.get("reference", "")).strip(),
        })
    return out


async def main_async(dry_run: bool) -> None:
    docs = await _load_docs_with_chunks()
    print(f"目标语料文档数: {len(docs)}")
    if not docs:
        print(f"❌ kb={CORPUS_KB_ID} 中无文档，请先运行 ingest_eval_corpus.py")
        return

    doc_names = sorted(docs.keys())
    # 全文拼给 LLM（约 4 万字，flash 模型可承受）
    full_text = "\n\n".join(f"### 文档：{n}\n{docs[n]}" for n in doc_names)
    print(f"语料总字符数: {len(full_text)}")

    plan = TYPE_PLAN
    if dry_run:
        plan = [(t, d, 2 if n > 2 else n) for t, d, n in TYPE_PLAN[:2]]

    all_items: list[dict] = []
    next_id = 1
    for type_name, difficulty, count in plan:
        print(f"\n出题中: {type_name} × {count} ...", flush=True)
        try:
            items = await _ask_batch(
                type_name, difficulty, count, doc_names, full_text, next_id,
            )
            print(f"  ✅ 得到 {len(items)} 题")
            all_items.extend(items)
            next_id += len(items)
        except Exception as e:
            print(f"  ❌ 失败: {type(e).__name__}: {e}")

    print(f"\n合计 {len(all_items)} 题")

    # 校验 expected_docs 是否为真实文件名
    valid = set(doc_names)
    bad = 0
    for it in all_items:
        filtered = [d for d in it["expected_docs"] if d in valid]
        if it["difficulty"] != "out-of-scope" and len(filtered) != len(it["expected_docs"]):
            bad += 1
        it["expected_docs"] = filtered
    if bad:
        print(f"⚠️  {bad} 题的 expected_docs 含未知名（已剔除）")

    OUT_JSON.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"明细已写入 {OUT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 100 题评测集")
    parser.add_argument("--dry-run", action="store_true", help="只生成少量题目试跑")
    args = parser.parse_args()
    asyncio.run(main_async(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
