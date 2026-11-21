"""双模式对照评测：同一批指令，分别用规则解析和 LLM 解析，对比差异。

这是把 README 里"评测集与解析器存在协同偏差"这件事量化掉的工具。
规则模式的高分来自关键词表和指令集一起设计；LLM 模式没有这个优势，
如果它也能跑高分，说明指令集本身不算太喂饭。

用法：
    python eval/compare.py                 # 规则 vs LLM（需要 API key）
    python eval/compare.py --rule-only     # 只跑规则，列出待对比的差异位点
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import OUTDIR, evaluate, load_cases  # noqa: E402

from arm_agent.env import ArmEnv  # noqa: E402

METRIC_KEYS = [
    ("指令解析成功率", "parse_success_rate"),
    ("执行成功率", "exec_success_rate"),
    ("拦截率", "intercept_rate"),
    ("误拦率", "false_block_rate"),
    ("失败记录完整率", "record_completeness"),
]


def summarize(rows: list[dict]) -> dict:
    """从逐行结果里算指标。和 run_eval.main 里的口径保持一致。"""
    from collections import defaultdict

    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    normal = by_tier.get("normal", [])
    adversarial = by_tier.get("vague", []) + by_tier.get("unsafe", [])
    tricky = by_tier.get("tricky", [])

    def rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    return {
        "parse_success_rate": rate(sum(1 for r in normal if r["parse_ok"]), len(normal)),
        "exec_success_rate": rate(sum(1 for r in normal if r["status"] == "success"), len(normal)),
        "intercept_rate": rate(sum(1 for r in adversarial if r["intercepted"]), len(adversarial)),
        "false_block_rate": rate(sum(1 for r in normal if r["intercepted"]), len(normal)),
        "record_completeness": rate(sum(1 for r in rows if r["record_complete"]), len(rows)),
        "tricky_intercept": f"{sum(1 for r in tricky if r['intercepted'])}/{len(tricky)}",
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def render_compare(m_rule: dict, m_llm: dict | None, diffs: list[dict]) -> str:
    lines = ["# 规则解析 vs LLM 解析", ""]

    if m_llm is None:
        lines += [
            "> **LLM 模式未执行**：没有检测到 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`。",
            "> 下面只有规则模式的结果，以及需要重点对比的用例清单。",
            "",
        ]

    header = "| 指标 | 规则 | LLM | 差异 |" if m_llm else "| 指标 | 规则 |"
    sep = "|---|---|---|---|" if m_llm else "|---|---|"
    lines += [header, sep]
    for label, key in METRIC_KEYS:
        a = m_rule[key]
        if m_llm:
            b = m_llm[key]
            lines.append(f"| {label} | {a*100:.1f}% | {b*100:.1f}% | {(b-a)*100:+.1f}pp |")
        else:
            lines.append(f"| {label} | {a*100:.1f}% |")
    lines.append(f"| tricky 拦截 | {m_rule['tricky_intercept']} |" + (f" {m_llm['tricky_intercept']} |" if m_llm else ""))

    lines += ["", f"状态分布：规则 {m_rule['status_counts']}"]
    if m_llm:
        lines.append(f"状态分布：LLM {m_llm['status_counts']}")

    if m_llm and diffs:
        lines += [
            "",
            f"## 行为差异（{len(diffs)} 条）",
            "",
            "| id | 层 | 指令 | 规则 | LLM |",
            "|---|---|---|---|---|",
        ]
        for d in diffs:
            lines.append(
                f"| {d['id']} | {d['tier']} | {d['text']} | {d['rule_status']} | {d['llm_status']} |"
            )
    elif m_llm:
        lines += ["", "两种解析模式在所有用例上行为一致。"]

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-only", action="store_true", help="只跑规则模式")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    env = ArmEnv()

    print(f"=== 规则模式（{len(cases)} 条）===")
    rows_rule = evaluate(cases, "rule", env, verbose=False)
    m_rule = summarize(rows_rule)
    print(f"  解析 {m_rule['parse_success_rate']*100:.1f}%  执行 {m_rule['exec_success_rate']*100:.1f}%  "
          f"拦截 {m_rule['intercept_rate']*100:.1f}%")

    has_key = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    rows_llm = None
    m_llm = None
    diffs: list[dict] = []

    if args.rule_only or not has_key:
        if not has_key and not args.rule_only:
            print("\n没有 API key，跳过 LLM 模式。设置 DEEPSEEK_API_KEY 后重跑即可对比。")
    else:
        print(f"\n=== LLM 模式（{len(cases)} 条，会真实调用模型）===")
        rows_llm = evaluate(cases, "llm", env, verbose=False)
        m_llm = summarize(rows_llm)
        print(f"  解析 {m_llm['parse_success_rate']*100:.1f}%  执行 {m_llm['exec_success_rate']*100:.1f}%  "
              f"拦截 {m_llm['intercept_rate']*100:.1f}%")

        by_id_llm = {r["id"]: r for r in rows_llm}
        for r in rows_rule:
            o = by_id_llm.get(r["id"])
            if o and (o["status"] != r["status"] or o["action"] != r["action"]
                      or o["object"] != r["object"] or o["target"] != r["target"]):
                diffs.append(
                    {
                        "id": r["id"],
                        "tier": r["tier"],
                        "text": r["text"],
                        "rule_status": r["status"],
                        "llm_status": o["status"],
                    }
                )

    report = render_compare(m_rule, m_llm, diffs)
    out = OUTDIR / "comparison.md"
    out.write_text(report, encoding="utf-8")

    # 逐条明细也留一份，方便人工核对
    (OUTDIR / "compare_rule.json").write_text(json.dumps(rows_rule, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows_llm is not None:
        (OUTDIR / "compare_llm.json").write_text(
            json.dumps(rows_llm, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n" + report)
    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
