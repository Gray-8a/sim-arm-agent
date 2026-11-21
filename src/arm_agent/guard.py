"""四层安全闸门。

设计原则（和 WAF 封禁那套逻辑同源）：
- 有些边界是"人也不能批"的，叫硬阻断（hard block）。人工审批只能放行软失败。
- 每一层都要返回可读的原因，因为它要进日志、要进评测、要能给人看。
- 闸门不负责猜"用户到底想要什么"。解析不了就拦，不替模型补全。

层次：
  L1 结构   —— Command 是否合法、枚举是否在白名单、字段组合是否自洽
  L2 置信度 —— 解析器自评置信度是否低于阈值
  L3 空间   —— 抓取点/放置点是否在允许的工作空间内
  L4 可达   —— 预演一次 IK，判断物理上能不能到
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .env import ArmEnv
from .schema import Command


@dataclass(frozen=True)
class GuardResult:
    layer: str
    passed: bool
    hard_block: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "passed": self.passed,
            "hard_block": self.hard_block,
            "reason": self.reason,
        }


@dataclass
class GuardReport:
    results: list[GuardResult] = field(default_factory=list)

    def add(self, r: GuardResult) -> None:
        self.results.append(r)

    @property
    def hard_blocked(self) -> bool:
        return any(r.hard_block for r in self.results)

    @property
    def failed(self) -> list[GuardResult]:
        return [r for r in self.results if not r.passed]

    @property
    def soft_failed(self) -> list[GuardResult]:
        return [r for r in self.results if not r.passed and not r.hard_block]

    @property
    def passed(self) -> bool:
        return not self.failed

    @property
    def needs_approval(self) -> bool:
        return bool(self.soft_failed) and not self.hard_blocked

    def to_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.results]

    def summary(self) -> str:
        if self.passed:
            return "全部通过"
        if self.hard_blocked:
            return "硬阻断: " + "；".join(r.reason for r in self.results if r.hard_block)
        return "待审批: " + "；".join(r.reason for r in self.soft_failed)


def in_workspace(p: np.ndarray) -> bool:
    p = np.asarray(p, dtype=float)
    return bool(np.all(p >= C.WORKSPACE_MIN) and np.all(p <= C.WORKSPACE_MAX))


def check_l1_structure(command: Command | None, parse_error: str | None) -> GuardResult:
    """L1：结构合法性。解析失败属于硬阻断——不猜、不补全。"""
    if command is None:
        return GuardResult("L1_structure", False, True, f"指令结构非法: {parse_error or '未解析出命令'}")
    return GuardResult("L1_structure", True, False, f"结构合法: {command.action.value}")


def check_l2_confidence(command: Command, threshold: float = C.CONFIDENCE_THRESHOLD) -> GuardResult:
    ok = command.confidence >= threshold
    return GuardResult(
        "L2_confidence",
        ok,
        False,  # 置信度低是可以人工放行的
        f"置信度 {command.confidence:.2f} {'≥' if ok else '<'} 阈值 {threshold:.2f}",
    )


def check_l3_workspace(command: Command, env: ArmEnv) -> GuardResult:
    """L3：预测抓取点与放置点是否落在允许的工作空间内。"""
    points: list[tuple[str, np.ndarray]] = []

    if command.object is not None:
        points.append(("抓取点", env.body_pos(command.object)))
    if command.target is not None:
        # 放置点取托盘上方一个方块半高，即物体落稳后的中心
        points.append(("放置点", env.site_pos(command.target) + np.array([0.0, 0.0, 0.026])))

    for label, p in points:
        if not in_workspace(p):
            return GuardResult(
                "L3_workspace",
                False,
                True,  # 出工作空间一律硬阻断，不给人批
                f"{label} {np.round(p, 3).tolist()} 超出允许范围",
            )

    if not points:
        return GuardResult("L3_workspace", True, False, "无需空间约束")
    return GuardResult("L3_workspace", True, False, f"{len(points)} 个目标点均在范围内")


def check_l4_reachability(command: Command, env: ArmEnv) -> GuardResult:
    """L4：对抓取点和放置点各预演一次 IK。不收敛则转人工。"""
    targets: list[tuple[str, np.ndarray]] = []
    if command.object is not None:
        targets.append(("抓取点", env.body_pos(command.object)))
    if command.target is not None:
        targets.append(("放置点", env.site_pos(command.target) + np.array([0.0, 0.0, 0.026])))

    if not targets:
        return GuardResult("L4_reachability", True, False, "无需可达性预演")

    for label, p in targets:
        ok, pos_err = env.preview_ik(p)
        if not ok:
            return GuardResult(
                "L4_reachability",
                False,
                False,  # 不可达可能是被挡住，允许人看一眼再决定
                f"{label} 不可达（IK 残差 {pos_err*1000:.1f}mm）",
            )
    return GuardResult("L4_reachability", True, False, f"{len(targets)} 个目标点 IK 均可解")


def run_guards(command: Command | None, env: ArmEnv, *, parse_error: str | None = None) -> GuardReport:
    """按 L1..L4 顺序执行。L1 失败就直接返回，没必要往下算。"""
    report = GuardReport()

    l1 = check_l1_structure(command, parse_error)
    report.add(l1)
    if not l1.passed:
        return report

    assert command is not None
    report.add(check_l2_confidence(command))
    report.add(check_l3_workspace(command, env))
    report.add(check_l4_reachability(command, env))
    return report
