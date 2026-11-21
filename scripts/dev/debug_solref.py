"""调试：weld 约束刚度扫描，找能稳定抓取的 solref。"""

import numpy as np

from arm_agent.env import ArmEnv


def trial(solref: float, solimp: tuple[float, float, float] | None = None) -> dict:
    env = ArmEnv()
    m, d = env.model, env.data
    eq = env._eq_ids["red_block"]
    m.eq_solref[eq] = [solref, 1.0]
    if solimp is not None:
        m.eq_solimp[eq] = list(solimp)

    block = env.body_pos("red_block")
    ok, _ = env.solve_to_pose(env.hand_target_for_grasp(block))
    if not ok:
        return {"solref": solref, "ok": False, "why": "ik"}

    env.set_gripper(True)
    env.step(250)
    env.forward()

    env.attach("red_block")
    if not env.is_attached("red_block"):
        return {"solref": solref, "ok": False, "why": "attach_not_active"}

    max_qacc = 0.0
    unstable = False
    lift = block + np.array([0.0, 0.0, 0.18])
    env.solve_to_pose(env.hand_target_for_grasp(lift))

    for _ in range(400):
        env.step(1)
        if not np.all(np.isfinite(d.qpos)):
            unstable = True
            break
        max_qacc = max(max_qacc, float(np.abs(d.qacc).max()))

    env.forward()
    block_now = env.body_pos("red_block")
    lifted = block_now[2] - block[2]
    still_attached = env.is_attached("red_block")
    return {
        "solref": solref,
        "ok": (not unstable) and lifted > 0.10,
        "unstable": unstable,
        "lifted_m": round(float(lifted), 4),
        "max_qacc": f"{max_qacc:.3g}",
        "attached": still_attached,
    }


def main():
    for sr in (0.002, 0.01, 0.02, 0.05):
        try:
            print(trial(sr))
        except Exception as exc:  # noqa: BLE001
            print({"solref": sr, "ok": False, "exc": f"{type(exc).__name__}: {exc}"[:120]})


if __name__ == "__main__":
    main()
