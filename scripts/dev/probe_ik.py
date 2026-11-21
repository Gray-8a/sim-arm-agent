"""D2 探测：IK 可达性 + 离屏渲染。"""

import numpy as np

from arm_agent.env import ArmEnv


def probe_ik(env, label, pos):
    ok, info = env.solve_to_pose(np.asarray(pos, dtype=float))
    print(
        f"{label:28s} target={np.round(pos,3)} "
        f"ik_ok={info['ik_converged']} pos_err={info['ik_pos_err']*1000:.1f}mm "
        f"rot_err={info['ik_rot_err']:.4f} servo={info['servo_converged']}"
    )
    return ok


def main():
    env = ArmEnv()

    print("=== IK 可达性 ===")
    targets = [
        ("home", [0.5545, 0.0, 0.6245]),
        ("red 上方10cm", [0.5, -0.13, 0.026 + 0.10 + 0.105]),
        ("red 抓取点", [0.5, -0.13, 0.026 + 0.105]),
        ("blue 抓取点", [0.5, 0.13, 0.026 + 0.105]),
        ("tray_left 上方", [0.34, -0.30, 0.012 + 0.10 + 0.105]),
        ("tray_right 上方", [0.34, 0.30, 0.012 + 0.10 + 0.105]),
        ("远点 x=0.85", [0.85, 0.0, 0.20]),
        ("越界 x=1.2", [1.2, 0.0, 0.20]),
    ]
    for label, t in targets:
        env.reset()
        probe_ik(env, label, t)
        print(f"    grasp_point now = {np.round(env.grasp_point(),4)}")

    print("\n=== 渲染 ===")
    env.reset()
    try:
        # Renderer 需要 GL；Windows 上如果失败，改用 MUJOCO_GL=wgl / osmesa
        r = env.make_renderer(640, 480)
        img = env.render(r)
        print("render ok, shape =", img.shape, "dtype =", img.dtype, "mean =", float(img.mean()))
    except Exception as exc:  # noqa: BLE001
        print("render FAILED:", type(exc).__name__, str(exc)[:400])


if __name__ == "__main__":
    main()
