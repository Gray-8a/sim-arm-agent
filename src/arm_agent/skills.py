"""动作原语：把"位姿目标"包装成可审计的执行步骤。

每个原语返回 StepRecord 列表，由上层写进 JSONL。这里不做任何决策，
也不做安全校验——那是 guard.py 的事。原语只负责"照着做，并如实记录"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import config as C
from .env import ArmEnv


@dataclass
class StepRecord:
    step: str
    ok: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"step": self.step, "ok": self.ok, **self.detail}


def _approach(env: ArmEnv, target: np.ndarray, *, label: str) -> StepRecord:
    ok, info = env.solve_to_pose(target)
    return StepRecord(
        step=label,
        ok=bool(ok),
        detail={
            "target": np.round(target, 4).tolist(),
            "ik_converged": info["ik_converged"],
            "servo_converged": info["servo_converged"],
            "ik_pos_err_mm": round(info["ik_pos_err"] * 1000, 2),
            "ik_rot_err": round(info["ik_rot_err"], 5),
        },
    )


def home(env: ArmEnv) -> list[StepRecord]:
    ok = env.move_to_joints(C.HOME_QPOS)
    env.forward()
    return [StepRecord("home", ok, {"qpos": np.round(env.data.qpos[:7], 3).tolist()})]


def move_to_point(env: ArmEnv, grasp_pos: np.ndarray) -> list[StepRecord]:
    """把等效抓取点移到指定世界坐标（保持俯视姿态）。"""
    hand_target = env.hand_target_for_grasp(np.asarray(grasp_pos, dtype=float))
    return [_approach(env, hand_target, label="move_to_point")]


def pick(env: ArmEnv, obj: str) -> list[StepRecord]:
    recs: list[StepRecord] = []

    block = env.body_pos(obj)
    above = block + np.array([0.0, 0.0, C.APPROACH_HEIGHT])
    grasp = block.copy()

    recs.append(_approach(env, env.hand_target_for_grasp(above), label="pick_approach"))
    recs.append(_approach(env, env.hand_target_for_grasp(grasp), label="pick_descend"))

    env.set_gripper(closed=True)
    env.step(250)  # 等夹爪闭合到位
    env.forward()

    env.attach(obj)
    env.step(50)
    env.forward()

    recs.append(
        StepRecord(
            "gripper_close_and_attach",
            True,
            {
                "object": obj,
                "object_pos_at_attach": np.round(block, 4).tolist(),
                "attached": env.is_attached(obj),
            },
        )
    )

    lifted = block + np.array([0.0, 0.0, C.APPROACH_HEIGHT + 0.05])
    recs.append(_approach(env, env.hand_target_for_grasp(lifted), label="pick_lift"))
    return recs


def place(env: ArmEnv, obj: str, site: str) -> list[StepRecord]:
    recs: list[StepRecord] = []

    site_pos = env.site_pos(site)
    # 方块中心放在托盘表面上方一个半高
    drop = site_pos + np.array([0.0, 0.0, 0.026])
    above = drop + np.array([0.0, 0.0, C.APPROACH_HEIGHT])

    recs.append(_approach(env, env.hand_target_for_grasp(above), label="place_approach"))
    recs.append(_approach(env, env.hand_target_for_grasp(drop), label="place_descend"))

    env.detach(obj)
    env.set_gripper(closed=False)
    env.step(150)
    env.forward()

    recs.append(
        StepRecord(
            "gripper_open_and_detach",
            True,
            {"object": obj, "site": site, "attached": env.is_attached(obj)},
        )
    )

    retract = drop + np.array([0.0, 0.0, C.APPROACH_HEIGHT])
    recs.append(_approach(env, env.hand_target_for_grasp(retract), label="place_retract"))
    return recs


def pick_and_place(env: ArmEnv, obj: str, site: str) -> list[StepRecord]:
    return pick(env, obj) + place(env, obj, site)


def record_to_dicts(recs: list[StepRecord]) -> list[dict]:
    return [asdict(r) for r in recs]
