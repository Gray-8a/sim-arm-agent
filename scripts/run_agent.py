"""命令行入口：给一句中文指令，走完整链路。

示例：
    python scripts/run_agent.py "把红色方块放到左边托盘"
    python scripts/run_agent.py "把红色方块放到左边托盘" --yes
    python scripts/run_agent.py "把那个东西拿过来" --mode rule
    python scripts/run_agent.py --list-scene
"""

from __future__ import annotations

import argparse
import json
import sys

from arm_agent import config as C
from arm_agent.agent import ArmAgent
from arm_agent.approval import AutoRejectApprover, CLIApprover, ScriptedApprover
from arm_agent.env import ArmEnv
from arm_agent.logger import RunLogger


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 指令 -> 仿真机械臂，带人工兜底")
    ap.add_argument("text", nargs="?", help="中文指令")
    ap.add_argument("--mode", default="auto", choices=["auto", "rule", "llm"], help="解析模式")
    ap.add_argument("--yes", action="store_true", help="跳过交互，自动批准软失败（演示用）")
    ap.add_argument("--reject", action="store_true", help="跳过交互，自动拒绝软失败")
    ap.add_argument("--log", default=None, help="JSONL 日志路径")
    ap.add_argument("--list-scene", action="store_true", help="打印场景白名单后退出")
    args = ap.parse_args()

    env = ArmEnv()

    if args.list_scene:
        print("物体:")
        for k, v in C.OBJECTS.items():
            print(f"  {k:12s} {v:8s} at {[round(float(x), 3) for x in env.body_pos(k)]}")
        print("位置:")
        for k, v in C.SITES.items():
            print(f"  {k:12s} {v:8s} at {[round(float(x), 3) for x in env.site_pos(k)]}")
        return

    if not args.text:
        ap.error("需要一条中文指令，或用 --list-scene")

    if args.yes:
        approver = ScriptedApprover(True, "demo-auto-approve")
    elif args.reject:
        approver = AutoRejectApprover()
    else:
        approver = CLIApprover()

    log_path = args.log or str(C.PROJECT_ROOT / "logs" / "run.jsonl")
    with RunLogger(log_path) as logger:
        agent = ArmAgent(env=env, logger=logger, approver=approver, parse_mode=args.mode)
        result = agent.run(args.text)

    print("\n=== 结果 ===")
    print(f"状态: {result.status}")
    if result.command is not None:
        print(f"命令: {result.command.summary()}  (置信度 {result.command.confidence:.2f}, 来源 {result.parse.source if result.parse else '?'})")
    if result.guard is not None:
        for r in result.guard.results:
            mark = "OK  " if r.passed else ("BLOCK" if r.hard_block else "ASK  ")
            print(f"  {mark} {r.layer:16s} {r.reason}")
    if result.detail:
        print(f"详情: {json.dumps(result.detail, ensure_ascii=False)}")
    print(f"日志: {log_path}")

    sys.exit(0 if result.status in ("success", "blocked", "rejected") else 1)


if __name__ == "__main__":
    main()
