"""D2 验收：跑通 加载 -> 设任务 -> 执行 -> 记录结果。

用法：
    python scripts/demo_d2.py            # 只跑，检查结果
    python scripts/demo_d2.py --gif      # 顺便录 artifacts/demo_d2.gif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arm_agent import config as C
from arm_agent import skills
from arm_agent.env import ArmEnv


def run(obj: str = "red_block", site: str = "tray_left", make_gif: bool = False, outdir: Path | None = None):
    outdir = outdir or (C.PROJECT_ROOT / "artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    env = ArmEnv()
    frames: list[np.ndarray] = []
    renderer = None

    if make_gif:
        renderer = env.make_renderer(560, 420)
        camera = env.make_camera()

        def hook(e: ArmEnv) -> None:
            frames.append(e.render(renderer, camera))

        env.step_hook = hook
        env.hook_every = 15
        frames.append(env.render(renderer, camera))

    t0 = time.perf_counter()
    recs = skills.pick_and_place(env, obj, site)
    elapsed = time.perf_counter() - t0

    # 平滑收尾，让物体稳定下来
    env.step_hook = None
    env.step(300)
    env.forward()

    block = env.body_pos(obj)
    target = env.site_pos(site)
    xy_dist = float(np.linalg.norm(block[:2] - target[:2]))
    settled = bool(block[2] < 0.10)
    success = bool(xy_dist < 0.10 and settled)

    episode = {
        "task": f"pick_and_place({obj} -> {site})",
        "success": success,
        "final_block_pos": np.round(block, 4).tolist(),
        "target_site_pos": np.round(target, 4).tolist(),
        "xy_dist_m": round(xy_dist, 4),
        "settled": settled,
        "elapsed_s": round(elapsed, 2),
        "all_steps_ok": all(r.ok for r in recs),
        "steps": [r.to_dict() for r in recs],
    }

    log_path = outdir / f"episode_{obj}_to_{site}.json"
    log_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 执行步骤 ===")
    for r in recs:
        print(f"  {'OK ' if r.ok else 'FAIL'} {r.step:26s} {r.detail}")
    print(f"\n方块终点 = {np.round(block,4)}  目标 = {np.round(target,4)}")
    print(f"xy 距离 = {xy_dist*1000:.1f} mm   落地 = {settled}   总步数收敛 = {episode['all_steps_ok']}")
    print(f"耗时 = {elapsed:.1f}s")
    print(f"SUCCESS = {success}")
    print(f"log -> {log_path}")

    if make_gif and frames:
        try:
            from PIL import Image

            gif_path = outdir / "demo_d2.gif"
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=60, loop=0)
            print(f"gif -> {gif_path}  ({len(frames)} frames)")
        except Exception as exc:  # noqa: BLE001
            print("gif FAILED:", type(exc).__name__, exc)

    return episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="red_block", choices=list(C.OBJECTS))
    ap.add_argument("--site", default="tray_left", choices=list(C.SITES))
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    ep = run(args.obj, args.site, make_gif=args.gif)
    raise SystemExit(0 if ep["success"] else 1)


if __name__ == "__main__":
    main()
