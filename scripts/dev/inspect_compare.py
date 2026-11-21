"""检查 LLM 对照结果的细节：关键用例的解析产物、置信度分布、延迟。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "eval" / "results"

WATCH = ("V07", "T01", "T02", "T03", "V10", "N01")


def main() -> None:
    rows = json.loads((RES / "compare_llm.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}

    print("=== 关键用例的 LLM 解析 ===")
    for cid in WATCH:
        r = by_id.get(cid)
        if not r:
            continue
        print(f"{cid:4s} {r['tier']:7s} status={r['status']:8s} {r['text']}")
        print(f"      -> action={r['action']} object={r['object']} target={r['target']}")
        print(f"      detail={r['detail']!r}")

    lat = [r["latency_ms"] for r in rows]
    print("\n=== 延迟 ===")
    print(f"n={len(lat)} mean={statistics.mean(lat):.0f}ms median={statistics.median(lat):.0f}ms "
          f"max={max(lat):.0f}ms total={sum(lat)/1000:.1f}s")


if __name__ == "__main__":
    main()
