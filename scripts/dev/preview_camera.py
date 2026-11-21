"""调相机：渲染几组候选视角拼成一张图，挑最合适的一组写进 demo。"""

import mujoco
from PIL import Image

from arm_agent import config as C
from arm_agent import skills
from arm_agent.env import ArmEnv

CANDIDATES = [
    # (azimuth, elevation, distance, lookat)
    (140, -25, 1.30, (0.42, 0.0, 0.20)),
    (150, -32, 1.10, (0.42, 0.0, 0.14)),
    (120, -30, 1.05, (0.42, 0.0, 0.16)),
    (170, -35, 1.15, (0.42, 0.0, 0.14)),
]


def cam(azimuth, elevation, distance, lookat):
    c = mujoco.MjvCamera()
    c.azimuth = azimuth
    c.elevation = elevation
    c.distance = distance
    c.lookat[:] = lookat
    return c


def main():
    env = ArmEnv()
    skills.pick(env, "red_block")
    skills.place(env, "red_block", "tray_left")
    env.forward()

    r = env.make_renderer(512, 384)
    tiles = []
    for az, el, dist, look in CANDIDATES:
        r.update_scene(env.data, camera=cam(az, el, dist, look))
        tiles.append(Image.fromarray(r.render().copy()))

    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * 2, h * 2))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * w, (i // 2) * h))
    out = C.PROJECT_ROOT / "artifacts" / "camera_candidates.png"
    sheet.save(out)
    print("saved", out)
    for i, c in enumerate(CANDIDATES):
        print(f"  tile {i} (row-major): azimuth={c[0]} elevation={c[1]} distance={c[2]}")


if __name__ == "__main__":
    main()
