"""探测场景几何：home 姿态下末端位姿、物体位置、以及简单可达性。

只读脚本，用来确定 config.py 里的常量（抓取偏移、工作空间边界）。
"""

import mujoco
import numpy as np

SCENE = "assets/scene_table.xml"
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])


def body_pose(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(name)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)

    # 复位到 home
    d.qpos[:] = m.qpos0
    d.qpos[:9] = HOME_QPOS
    mujoco.mj_forward(m, d)

    print("=== home 姿态 ===")
    p, R = body_pose(m, d, "hand")
    print("hand xpos =", np.round(p, 4))
    print("hand xmat =\n", np.round(R, 4))
    print("hand 局部 +z 轴在世界系 =", np.round(R[:, 2], 4))

    for name in ["red_block", "blue_block", "tray_left", "tray_right"]:
        pp, _ = body_pose(m, d, name)
        print(f"{name:12s} xpos = {np.round(pp, 4)}")

    print("\n=== 关节 ===")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"{i:2d} {name:16s} type={m.jnt_type[i]} qposadr={m.jnt_qposadr[i]} dofadr={m.jnt_dofadr[i]}")

    print("\n=== actuator ===")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"{i} {name:12s} ctrlrange={m.actuator_ctrlrange[i]}")

    print("\n=== 关键 body 的局部偏移 ===")
    # 手爪两个 finger 的原点，用来估算指尖位置
    for name in ["left_finger", "right_finger"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"{name}: 相对父 body 的 pos = {m.body_pos[bid]}")


if __name__ == "__main__":
    main()
