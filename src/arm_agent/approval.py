"""人工审批。

两种实现：
- CLIApprover：真问人，回车/y 通过，其它拒绝。默认超时不自动放行。
- ScriptedApprover：给评测和 CI 用，按预设答复，保证结果是确定的。

审批只能放行"软失败"（低置信度、IK 不收敛）。硬阻断不会走到这里，
这是和 WAF 封禁里"可信 IP 硬阻断"完全一致的分层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .guard import GuardResult
from .schema import Command


@dataclass
class ApprovalDecision:
    approved: bool
    operator: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"approved": self.approved, "operator": self.operator, "note": self.note}


class Approver(Protocol):
    def request(self, command: Command, reasons: list[GuardResult]) -> ApprovalDecision: ...


class CLIApprover:
    """终端交互审批。默认拒绝——必须明确输入 y 才放行。"""

    def __init__(self, operator: str = "local-operator"):
        self.operator = operator

    def request(self, command: Command, reasons: list[GuardResult]) -> ApprovalDecision:
        print("\n" + "=" * 62)
        print("需要人工确认")
        print("=" * 62)
        print(f"  指令: {command.summary()}")
        print(f"  置信度: {command.confidence:.2f}")
        print(f"  理由: {command.reasoning}")
        print("  触发原因:")
        for r in reasons:
            print(f"    - [{r.layer}] {r.reason}")
        print("-" * 62)
        try:
            ans = input("执行这条指令吗？输入 y 确认，其它任意输入拒绝: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        approved = ans in ("y", "yes")
        return ApprovalDecision(approved, self.operator, "CLI 交互")


class ScriptedApprover:
    """评测用：固定答复。"""

    def __init__(self, approve: bool, operator: str = "scripted"):
        self.approve = approve
        self.operator = operator
        self.calls: list[dict] = []

    def request(self, command: Command, reasons: list[GuardResult]) -> ApprovalDecision:
        self.calls.append(
            {
                "command": command.model_dump(),
                "reasons": [r.to_dict() for r in reasons],
            }
        )
        return ApprovalDecision(self.approve, self.operator, "脚本预设")


class AutoRejectApprover:
    """非交互环境下默认拒绝，用于无人值守跑批。"""

    def __init__(self, operator: str = "auto-reject"):
        self.operator = operator

    def request(self, command: Command, reasons: list[GuardResult]) -> ApprovalDecision:
        return ApprovalDecision(False, self.operator, "非交互模式默认拒绝")
