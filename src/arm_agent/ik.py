"""阻尼最小二乘（DLS）逆运动学。

只解 7 个手臂关节的静态目标，不涉及轨迹优化、动力学或力控——那是运动控制的活，
本项目明确不做。用的是标准做法：6 维误差（位置 + 姿态）乘 Jacobian 伪逆迭代。

为什么不引入 mink：本条链路只需要"给位姿、返回关节角"这一个动作，
40 行 DLS 没有外部依赖、行为完全可预测，调依赖的性价比不划算。
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF = 7


def _rot_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """返回把 R_cur 转到 R_tgt 所需的世界系旋转向量（轴 * 角）。"""
    R_err = R_tgt @ R_cur.T
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_err.flatten()))
    if quat[0] < 0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sin(angle / 2.0)
    if abs(s) < 1e-9 or angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return quat[1:] / s * angle


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    *,
    max_iters: int = 400,
    pos_tol: float = 2e-3,
    rot_tol: float = 2e-2,
    damping: float = 1e-2,
    step_scale: float = 0.5,
    rest_qpos: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, float, float]:
    """就地求解 IK，返回 (是否收敛, 7 维关节角, 位置误差, 姿态误差)。

    会修改 data.qpos 的前 7 个自由度；调用方负责把结果写回 ctrl。
    rest_qpos 给定时，从该姿态出发（默认沿用 data 当前值）。
    """
    if rest_qpos is not None:
        data.qpos[:ARM_DOF] = rest_qpos

    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    pos_err = rot_err = float("inf")

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_pos = data.xpos[body_id]
        cur_rot = data.xmat[body_id].reshape(3, 3)

        e_pos = target_pos - cur_pos
        e_rot = _rot_error(cur_rot, target_rot)
        pos_err = float(np.linalg.norm(e_pos))
        rot_err = float(np.linalg.norm(e_rot))

        if pos_err < pos_tol and rot_err < rot_tol:
            return True, data.qpos[:ARM_DOF].copy(), pos_err, rot_err

        err = np.concatenate([e_pos, e_rot])

        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])  # 6 x 7

        # dq = J^T (J J^T + lambda^2 I)^-1 e
        JJt = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)

        data.qpos[:ARM_DOF] += step_scale * dq

        # 夹到关节限位，避免解出物理上不可达的角
        lo = model.jnt_range[:ARM_DOF, 0]
        hi = model.jnt_range[:ARM_DOF, 1]
        data.qpos[:ARM_DOF] = np.clip(data.qpos[:ARM_DOF], lo, hi)

    mujoco.mj_forward(model, data)
    ok = pos_err < 5e-3 and rot_err < 5e-2
    return ok, data.qpos[:ARM_DOF].copy(), pos_err, rot_err
