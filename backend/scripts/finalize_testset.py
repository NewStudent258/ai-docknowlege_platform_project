"""把 gen_eval_testset.py 的原始出题结果按配比裁剪为 100 题，生成 eval_test_set_v2.py。

为什么需要这一步
----------------
LLM 出题时数量不完全受控（例如要求 30 题可能返回 70 题）。本脚本负责：
  1. 按题型目标数量裁剪（保持设计的难度梯度与题型配比）
  2. 去掉重复问题文本
  3. 剔除 expected_docs 为空的非超范围题（无解题）
  4. 强制超范围题的 expected_docs 为空数组
  5. 重新编号后写入 eval_test_set_v2.py

用法：
  cd backend
  python scripts/finalize_testset.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = _BACKEND_ROOT / "tests" / "eval"
SRC = EVAL_DIR / "_gen_report.json"
OUT = EVAL_DIR / "eval_test_set_v2.py"

# 题型配比（与 gen_eval_testset.py 的 TYPE_PLAN 一致）
TARGET = {
    "单文档精确查询": 30,
    "跨文档查询": 25,
    "简称与口语化查询": 15,
    "蕴含推理查询": 15,
    "超出知识库范围": 15,
}

OOS_TYPE = "超出知识库范围"


def main() -> None:
    if not SRC.exists():
        print(f"❌ 未找到原始出题结果: {SRC}")
        print("   请先运行: python scripts/gen_eval_testset.py")
        return

    items = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"原始题数: {len(items)}")
    for t, n in Counter(it["type"] for it in items).items():
        print(f"  {t}: {n} (目标 {TARGET.get(t, '?')})")

    seen_q: set[str] = set()
    kept: list[dict] = []
    for type_name, want in TARGET.items():
        pool = [it for it in items if it["type"] == type_name]
        got = 0
        for it in pool:
            q = it["question"].strip()
            if q in seen_q:
                continue
            if type_name == OOS_TYPE:
                it["expected_docs"] = []
            elif not it["expected_docs"]:
                # 非超范围题必须有期望文档，否则是「无解题」
                continue
            seen_q.add(q)
            kept.append(it)
            got += 1
            if got >= want:
                break
        print(f"  裁剪后 {type_name}: {got}")

    for i, it in enumerate(kept, start=1):
        it["id"] = i

    print(f"\n最终题数: {len(kept)}")
    print("难度分布:", dict(Counter(it["difficulty"] for it in kept)))

    lines = [
        f'"""评测集 v2 — {len(kept)} 题，由 scripts/gen_eval_testset.py 基于入库语料自动生成。',
        "",
        "语料来源：kb_id=17「DocMind评测库A(目标+干扰)」的 25 份目标文档。",
        "题型配比：单文档精确查询 30 / 跨文档 25 / 简称口语化 15 / 蕴含推理 15 / 超范围 15。",
        "",
        "字段说明：",
        "  - expected_docs: 包含答案依据的文档名（超范围题为空数组）",
        "  - reference:     参考答案，供 ragas ContextPrecision 等指标使用",
        "  - difficulty:    easy / medium / hard / out-of-scope",
        "",
        "注意：本文件由脚本生成，请勿手工编辑；如需调整请改 gen_eval_testset.py 后重跑",
        "      scripts/gen_eval_testset.py && python scripts/finalize_testset.py。",
        '"""',
        "",
        "from typing import Any",
        "",
        "EVAL_TEST_SET_V2: list[dict[str, Any]] = [",
    ]
    for it in kept:
        lines.append("    {")
        lines.append(f'        "id": {it["id"]},')
        lines.append(f'        "question": {json.dumps(it["question"], ensure_ascii=False)},')
        lines.append(f'        "type": {json.dumps(it["type"], ensure_ascii=False)},')
        lines.append(f'        "difficulty": {json.dumps(it["difficulty"], ensure_ascii=False)},')
        lines.append(f'        "expected_docs": {json.dumps(it["expected_docs"], ensure_ascii=False)},')
        lines.append(f'        "reference": {json.dumps(it["reference"], ensure_ascii=False)},')
        lines.append("    },")
    lines.append("]")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
