"""编排层：把解析、闸门、审批、执行、验证串成一条可审计的链路。

执行顺序是固定的，不允许跳步：
    parse -> guard(L1..L4) -> [hard block? 结束] -> [need approval? 问人] -> execute -> verify

返回的 RunResult.status 是评测脚本唯一依赖的东西，取值：
    success  执行并验证通过
    blocked  被硬阻断，未执行
    rejected 软失败转人工后被拒绝，未执行
    failed   执行了但结果验证不通过
    error    链路自身异常
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import skills
from .approval import Approver
from .env import ArmEnv
from .guard import GuardReport, run_guards
from .logger import RunLogger
from .parser import ParseResult, parse
from .schema import Action, Command


@dataclass
class RunResult:
    status: str
    text: str
    command: Command | None = None
    guard: GuardReport | None = None
    parse: ParseResult | None = None
    steps: list[dict] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "text": self.text,
            "command": self.command.model_dump() if self.command else None,
            "guard": self.guard.to_dicts() if self.guard else None,
            "parse": self.parse.to_dict() if self.parse else None,
            "steps": self.steps,
            "detail": self.detail,
        }


class ArmAgent:
    def __init__(
        self,
        env: ArmEnv | None = None,
        *,
        logger: RunLogger | None = None,
        approver: Approver,
        parse_mode: str = "auto",
    ):
        self.env = env or ArmEnv()
        self.logger = logger
        self.approver = approver
        self.parse_mode = parse_mode

    # ---------------------------------------------------------------- 入口

    def run(self, text: str) -> RunResult:
        t0 = time.perf_counter()
        self._log("command_received", text=text, parse_mode=self.parse_mode)

        # 1. 解析
        parsed = parse(text, mode=self.parse_mode)
        self._log(
            "parsed",
            source=parsed.source,
            payload=parsed.payload,
            error=parsed.error,
            notes=parsed.notes,
        )

        # 2. 闸门
        report = run_guards(parsed.command, self.env, parse_error=parsed.error)
        self._log("guard", results=report.to_dicts(), summary=report.summary())

        if report.hard_blocked:
            res = RunResult("blocked", text, parsed.command, report, parsed, detail={"reason": report.summary()})
            self._log("blocked", reason=report.summary(), elapsed_ms=_ms(t0))
            return res

        # 3. 审批
        if report.needs_approval:
            self._log("approval_requested", reasons=[r.to_dict() for r in report.soft_failed])
            decision = self.approver.request(parsed.command, report.soft_failed)
            self._log("approval_decision", **decision.to_dict())
            if not decision.approved:
                res = RunResult(
                    "rejected",
                    text,
                    parsed.command,
                    report,
                    parsed,
                    detail={"reason": "人工拒绝", "reasons": [r.reason for r in report.soft_failed]},
                )
                self._log("rejected", elapsed_ms=_ms(t0))
                return res

        # 4. 执行
        assert parsed.command is not None
        self._log("executing", action=parsed.command.action.value)
        try:
            records = self._execute(parsed.command)
        except Exception as exc:  # noqa: BLE001
            self._log("execution_error", error=f"{type(exc).__name__}: {exc}")
            return RunResult(
                "error",
                text,
                parsed.command,
                report,
                parsed,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )

        steps = [r.to_dict() for r in records]
        steps_ok = all(r.ok for r in records)
        self._log("executed", steps=steps, all_ok=steps_ok)

        # 5. 验证
        verified, verify_detail = self._verify(parsed.command)
        self._log("verified", verified=verified, **verify_detail)

        status = "success" if (verified and steps_ok) else "failed"
        self._log("finished", status=status, elapsed_ms=_ms(t0))
        return RunResult(
            status,
            text,
            parsed.command,
            report,
            parsed,
            steps=steps,
            detail={"steps_ok": steps_ok, **verify_detail},
        )

    # ---------------------------------------------------------------- 内部

    def _execute(self, command: Command) -> list[skills.StepRecord]:
        if command.action is Action.HOME:
            return skills.home(self.env)
        if command.action is Action.PICK:
            return skills.pick(self.env, command.object)  # type: ignore[arg-type]
        return skills.pick_and_place(self.env, command.object, command.target)  # type: ignore[arg-type]

    def _verify(self, command: Command) -> tuple[bool, dict]:
        """执行后校验：物体是否真的落在目标位置。"""
        if command.action is not Action.PICK_AND_PLACE:
            return True, {"note": f"{command.action.value} 无放置校验"}

        self.env.step(200)  # 等物体落稳
        self.env.forward()
        block = self.env.body_pos(command.object)  # type: ignore[arg-type]
        site = self.env.site_pos(command.target)  # type: ignore[arg-type]
        dist = float(np.linalg.norm(block[:2] - site[:2]))
        settled = bool(block[2] < 0.10)
        ok = bool(dist < 0.10 and settled)
        return ok, {
            "block_pos": np.round(block, 4).tolist(),
            "site_pos": np.round(site, 4).tolist(),
            "xy_dist_m": round(dist, 4),
            "settled": settled,
        }

    def _log(self, kind: str, **fields) -> None:
        if self.logger is not None:
            self.logger.event(kind, **fields)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
