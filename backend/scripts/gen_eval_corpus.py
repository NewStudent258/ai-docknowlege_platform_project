"""DocMind 评测语料生成器 — 生成目标文档（HR/财务/IT/法务 四领域）。

背景与用途
----------
原 28 题评测集的语料（knowledge_samples/）已被 gitignore 丢失，无法复现。
本脚本用 LLM 重新生成一套**可控、可复现**的目标语料：

- 目标文档 25 份，覆盖 HR / 财务 / IT / 法务 四领域
- 每份文档结构统一（Markdown 标题 + 条款），确保 chunker 能按章节正确切分
- 文档内容含明确数字、流程、责任部门，便于编写有唯一答案的题目
- 这批文档入库到新知识库后，与既有 KB14（100 份薪酬/考勤制度）共同构成
  「目标语料 + 同领域干扰语料」的评测环境，用于跑重排消融实验

产出：backend/knowledge_samples/eval_corpus/*.md

用法：
  cd backend
  python scripts/gen_eval_corpus.py                 # 生成全部 25 份
  python scripts/gen_eval_corpus.py --only 3        # 只生成前 3 份（试跑）
  python scripts/gen_eval_corpus.py --force         # 覆盖已存在文件
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

from app.config import settings
from app.core.llm import chat_completion

OUT_DIR = _BACKEND_ROOT / "knowledge_samples" / "eval_corpus"

# ---------------------------------------------------------------------------
# 目标文档清单：25 份，四领域分布
# 每项 = (序号, 文件名, 领域, 文档主题, 必须覆盖的关键事实点)
# 关键事实点直接决定后续出题范围，也约束 LLM 不要写偏。
# ---------------------------------------------------------------------------
DOC_SPECS: list[tuple[int, str, str, str, list[str]]] = [
    # ===== HR 人力资源（7 份）=====
    (1, "员工入职办理指南.md", "HR", "新员工入职手续与账号开通", [
        "报到时间与地点（上午 9:00 人力资源部）",
        "需提交的材料清单（身份证复印件、学历学位证书、离职证明、照片、银行卡、体检报告）",
        "签署的协议（劳动合同、保密协议、竞业限制协议）",
        "领取物品（工牌、办公电脑、办公用品）",
        "IT 部门开通的账号（企业邮箱、OA、即时通讯、业务系统）",
        "入职培训内容",
    ]),
    (2, "考勤与请假管理制度.md", "HR", "考勤打卡、各类假期的申请规则", [
        "上下班时间与打卡要求（含补卡次数上限）",
        "病假：证明材料要求、提前申请天数、工资发放标准",
        "事假：每月天数上限、是否扣薪",
        "年休假：工龄对应的天数、申请提前期",
        "请假审批层级（按天数分档）",
        "紧急情况补办手续的时限",
    ]),
    (3, "员工培训与发展办法.md", "HR", "培训体系、转正要求、培训费用", [
        "新员工培训时长与考核方式",
        "培训不合格的处理方式（补考次数、影响）",
        "外部培训费用报销条件与服务期约定",
        "岗位技能认证与晋升关系",
    ]),
    (4, "绩效考核管理制度.md", "HR", "考核周期、等级、结果应用", [
        "考核周期（季度/年度）与时间节点",
        "考核等级划分（S/A/B/C/D）与各档比例",
        "C 档的后果（是否影响年终奖、绩效系数）",
        "绩效申诉流程与受理时限",
        "连续两次低绩效的处理",
    ]),
    (5, "员工离职交接流程.md", "HR", "离职申请、交接、权限回收", [
        "离职提前申请天数（试用期与正式员工区别）",
        "交接清单内容（工作、资产、文档）",
        "账号权限回收的责任部门与时限",
        "工资结算与离职证明出具时限",
        "竞业限制补偿的适用情形",
    ]),
    (6, "员工行为规范与奖惩条例.md", "HR", "行为红线、处分等级、奖励", [
        "处分等级（警告/记过/降级/解除劳动合同）",
        "各级处分的适用情形与审批权限",
        "奖励类型与评选周期",
        "申诉渠道与时限",
    ]),
    (7, "实习生管理办法.md", "HR", "实习生的招聘、待遇、转正", [
        "实习期限与工作时间要求",
        "实习补贴标准",
        "转正条件与转正后薪资调整规则",
        "实习生是否适用绩效考核",
    ]),

    # ===== 财务（6 份）=====
    (8, "费用报销管理制度.md", "财务", "报销范围、单据、审批权限", [
        "可报销费用类型与不可报销项",
        "报销单据要求（发票类型、附件）",
        "审批权限金额分档（如 2000 元以下、2000-10000 元、10000 元以上）",
        "报销时限与逾期扣减规则",
        "付款时限（财务审核通过后几个工作日）",
    ]),
    (9, "差旅费用管理规定.md", "财务", "出差申请、标准、补贴", [
        "出差申请流程与提前申请天数",
        "交通工具等级标准（按职级）",
        "住宿费上限标准（按城市级别）",
        "出差补贴标准（按天/按地区）",
        "差旅报销需附的凭证",
    ]),
    (10, "采购与付款审批流程.md", "财务", "采购申请、比价、付款", [
        "采购申请流程与所需材料",
        "比价/招标的金额门槛",
        "采购审批权限分档",
        "验收流程与付款条件",
        "供应商准入要求",
    ]),
    (11, "固定资产管理办法.md", "财务", "资产入库、编号、盘点、报废", [
        "资产编号生成规则",
        "资产入账与领用登记流程",
        "盘点周期与责任人",
        "报废条件与审批流程",
        "资产损坏/丢失的赔偿标准",
    ]),
    (12, "预算编制与执行管理规定.md", "财务", "预算编制、调整、超支", [
        "预算编制时间节点与责任部门",
        "预算审批层级",
        "预算调整流程",
        "超预算支出的处理",
        "预算执行分析与考核",
    ]),
    (13, "发票与税务管理规定.md", "财务", "发票开具、报销要求、税务合规", [
        "可报销发票类型与不可报销发票",
        "增值税专用发票的开具要求",
        "发票丢失的处理",
        "个人所得税代扣代缴的基本规则",
    ]),

    # ===== IT 信息技术（6 份）=====
    (14, "VPN 使用与配置指南.md", "IT", "VPN 申请、连接、故障处理", [
        "VPN 账号申请流程与审批人",
        "支持的客户端与系统",
        "连接步骤（分步骤）",
        "密码忘记的重置方式",
        "常见故障排查（连不上、频繁掉线）",
        "安全使用要求（禁止事项）",
    ]),
    (15, "办公设备使用说明.md", "IT", "打印机、扫描仪、投影仪使用", [
        "打印机驱动安装方式",
        "卡纸处理步骤",
        "墨盒/硒鼓更换流程与耗材申领方式",
        "设备故障报修渠道与响应时限",
    ]),
    (16, "系统权限申请流程.md", "IT", "各类系统权限的申请与审批", [
        "权限类型（生产服务器、数据库只读/读写、业务系统）",
        "各类权限的审批人",
        "申请所需材料与审批时限",
        "权限定期复核周期",
        "离职或转岗时的权限调整",
    ]),
    (17, "信息安全管理制度.md", "IT", "信息分级、设备使用、违规处理", [
        "信息密级划分（公开/内部/秘密/绝密）",
        "个人设备与私人软件的使用限制",
        "移动存储介质的使用规定",
        "工作文件能否存放于个人存储",
        "信息安全违规的处理",
    ]),
    (18, "数据备份与安全规范.md", "IT", "数据备份策略、销毁、数据安全", [
        "备份频率与保留周期",
        "备份介质与存放要求",
        "数据销毁的流程与审批",
        "用户个人数据的处理要求",
        "数据泄露的应急报告流程",
    ]),
    (19, "网络与邮箱使用规范.md", "IT", "网络使用、邮箱、即时通讯工具", [
        "办公网络使用规定（访客网络是否可用）",
        "邮箱容量与附件大小限制",
        "使用微信/钉钉等工具传输文件的限制",
        "邮件外发的审批要求",
        "违规使用的后果",
    ]),

    # ===== 法务 / 合规 / 行政（6 份）=====
    (20, "合同审批与签署流程.md", "法务", "合同起草、审核、用印", [
        "合同审核的必经环节（法务审核）",
        "合同金额分档与审批层级",
        "用印申请流程与所需材料",
        "紧急情况先用印的补办要求",
        "合同归档要求",
    ]),
    (21, "印章管理规定.md", "法务", "印章种类、保管、使用审批", [
        "印章种类（公章、合同章、财务章、法人章）",
        "各类印章的保管部门与责任人",
        "用印申请与审批流程",
        "用印登记要求",
        "违规用印的处理",
    ]),
    (22, "知识产权与保密管理规定.md", "法务", "商业秘密、竞业限制、知识产权归属", [
        "商业秘密的范围界定",
        "保密协议的签署要求与保密期限",
        "职务成果的知识产权归属",
        "违反保密义务的责任",
        "竞业限制的期限与补偿标准",
    ]),
    (23, "突发事件应急预案.md", "行政", "火灾、停电、信息安全事件的处置", [
        "火灾时的疏散路线与集合点",
        "消防器材位置与使用要点",
        "应急组织架构与联系人",
        "停电/停水的应对措施",
        "演练频率",
    ]),
    (24, "访客接待与门禁管理规定.md", "行政", "访客登记、门禁、接待", [
        "访客登记流程（前台需做的工作）",
        "访客证的发放与回收",
        "访客能否使用公司内部网络",
        "接待陪同要求",
        "门禁卡申领与遗失处理",
    ]),
    (25, "会议室与办公环境管理规定.md", "行政", "会议室预约、办公秩序、后勤", [
        "会议室预约方式与可预约时长",
        "预约后未使用的取消时限与违约处理",
        "工位与办公环境要求",
        "快递收发与办公用品申领",
    ]),
]


_SYSTEM_PROMPT = """你是一名企业行政与合规制度编写专家。请根据给定的文档主题和必须覆盖的关键事实点，撰写一份**完整、专业、内部一致**的企业管理制度文档。

写作要求：
1. 使用 Markdown 格式，一级标题为文档名，正文用「第一章/第二章」或「一、二、」分节，条款编号形如「第一条」「第二条」。
2. **必须覆盖全部给定的关键事实点**，且每个事实点都要写成具体、可执行的规定。
3. 关键数字必须明确写出（天数、金额、比例、时限、次数上限），不要用「适当」「若干」等模糊表述。
4. 明确写出归口管理部门或审批责任人。
5. 篇幅控制在 800-1500 字，条款总数 12-20 条。
6. 内容应当专业、平实，像真实的企业内部制度文档。**不要**出现「示例」「例如」「测试」「用户提问」「假设场景」这类字样。
7. 只输出文档正文本身，不要输出任何解释、前言或 Markdown 代码块围栏。
"""

_USER_TEMPLATE = """请撰写以下企业制度文档：

文档名称：{filename}
所属领域：{domain}
文档主题：{topic}

必须覆盖的关键事实点：
{facts}

请直接输出 Markdown 格式的文档正文。"""


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ``` 代码块围栏。"""
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


async def generate_one(
    index: int,
    filename: str,
    domain: str,
    topic: str,
    facts: list[str],
) -> tuple[str, str]:
    """生成单份文档，返回 (filename, content)。"""
    fact_lines = "\n".join(f"  - {f}" for f in facts)
    user_prompt = _USER_TEMPLATE.format(
        filename=filename, domain=domain, topic=topic, facts=fact_lines,
    )

    resp = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        deep_thinking=False,
        max_tokens=2500,
    )
    content = _strip_fences(resp.content)

    # 保证标题存在，便于 chunker 的 section 检测
    if not content.lstrip().startswith("#"):
        content = f"# {filename.removesuffix('.md')}\n\n{content}"

    return filename, content


async def main_async(only: int | None, force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = DOC_SPECS if not only else DOC_SPECS[:only]
    print(f"目标语料生成：共 {len(specs)} 份 → {OUT_DIR}")
    print("=" * 70)

    manifest: list[dict] = []
    for idx, (num, filename, domain, topic, facts) in enumerate(specs, start=1):
        out_path = OUT_DIR / filename
        if out_path.exists() and not force:
            print(f"[{idx}/{len(specs)}] 跳过（已存在）: {filename}")
            manifest.append({"filename": filename, "domain": domain, "topic": topic})
            continue

        print(f"[{idx}/{len(specs)}] 生成中: {filename} ({domain}) ...", flush=True)
        try:
            _, content = await generate_one(num, filename, domain, topic, facts)
        except Exception as e:
            print(f"    ❌ 失败: {type(e).__name__}: {e}")
            continue

        out_path.write_text(content, encoding="utf-8")
        print(f"    ✅ {len(content)} 字符")
        manifest.append({
            "filename": filename,
            "domain": domain,
            "topic": topic,
            "facts": facts,
            "chars": len(content),
        })

    manifest_path = OUT_DIR / "_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("=" * 70)
    print(f"完成。清单已写入 {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 DocMind 评测目标语料")
    parser.add_argument("--only", type=int, default=None, help="只生成前 N 份（试跑）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    print(f"LLM 模型: {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    asyncio.run(main_async(only=args.only, force=args.force))


if __name__ == "__main__":
    main()
