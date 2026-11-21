"""MuJoCo 环境封装：只暴露"位姿 -> 运动 -> 位姿"这一层，不含任何规划逻辑。

设计约束：
- 世界坐标系 = 桌子顶面 z=0，机械臂基座在原点。
- 手爪默认保持"俯视抓取"姿态（hand 局部 +z 朝下），由 home 姿态固化而来。

关于抓取附着（重要工程妥协，面试会被问）：
先试过 equality weld 约束，实际跑不通——闭爪时手指的接触约束和 weld 的位姿约束
互相打架，激活瞬间 QACC 冲到 1e6 量级然后数值发散（solref 从 0.002 扫到 0.05 都不稳）。
正确做法是在抓取期间排除手指与物体的接触对，但那是接触参数调优，属于运动控制范畴，
本项目明确不做。因此改为运动学附着：抓住后用相对位姿把物体写到目标位置，
同时临时关闭该物体的碰撞，避免和手指穿插。代价是抓取期间物体不受重力和接触影响；
收益是行为 100% 确定、可复现、gif 稳定。这一取舍写进 README 的局限一节。

任何绕过这里的直接 mj_step 都视为破坏"单一收口"约束。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import config as C
from .ik import ARM_DOF, solve_ik


class ArmEnv:
    def __init__(self, scene_path: str | Path | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path or C.SCENE_PATH))
        self.data = mujoco.MjData(self.model)

        self._hand_id = self._body_id("hand")
        self._body_ids = {name: self._body_id(name) for name in C.OBJECTS}
        self._geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom") for name in C.OBJECTS
        }
        self._joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free") for name in C.OBJECTS
        }
        self._qpos_adr = {n: int(self.model.jnt_qposadr[j]) for n, j in self._joint_ids.items()}
        self._dof_adr = {n: int(self.model.jnt_dofadr[j]) for n, j in self._joint_ids.items()}
        self._site_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site") for name in C.SITES
        }

        # 抓取附着状态：{obj, offset(3), rot(3x3)}，offset/rot 为 hand 局部坐标系下的相对位姿
        self._attach: dict | None = None

        # 录制钩子：每 hook_every 个物理步回调一次，用于抓 GIF 帧
        self.step_hook = None
        self.hook_every = 20
        self._step_count = 0

        self.reset()
        # 俯视抓取姿态：直接用 home 时的手爪朝向固化下来
        self.grasp_rot = self.data.xmat[self._hand_id].reshape(3, 3).copy()

        # 演示画面质量：默认 shadow map 在整张桌面上不够用，边缘会碎
        self.model.vis.quality.shadowsize = 8192

    # ---------- 基础 ----------

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body not found: {name}")
        return bid

    def reset(self) -> None:
        self._attach = None
        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:ARM_DOF] = C.HOME_QPOS
        self.data.ctrl[:ARM_DOF] = C.HOME_QPOS
        self.data.ctrl[7] = C.GRIPPER_OPEN
        for name in C.OBJECTS:
            self._set_collision(name, True)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self._attach is not None:
                self._apply_attachment()
            self._step_count += 1
            if self.step_hook is not None and self._step_count % self.hook_every == 0:
                self.step_hook(self)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # ---------- 观测 ----------

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self._body_id(name)].copy()

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self._site_ids[name]].copy()

    def hand_pos(self) -> np.ndarray:
        return self.data.xpos[self._hand_id].copy()

    def hand_rot(self) -> np.ndarray:
        return self.data.xmat[self._hand_id].reshape(3, 3).copy()

    def grasp_point(self) -> np.ndarray:
        """两指中间的等效抓取点（世界坐标系）。"""
        return self.hand_pos() + self.hand_rot() @ C.GRASP_OFFSET_LOCAL

    def hand_target_for_grasp(self, grasp_pos: np.ndarray) -> np.ndarray:
        """给定想要的抓取点位置，反推 hand body 应该在哪。"""
        return np.asarray(grasp_pos, dtype=float) - self.grasp_rot @ C.GRASP_OFFSET_LOCAL

    def joint_error(self) -> float:
        return float(np.max(np.abs(self.data.qpos[:ARM_DOF] - self.data.ctrl[:ARM_DOF])))

    # ---------- 动作 ----------

    def set_gripper(self, closed: bool) -> None:
        self.data.ctrl[7] = C.GRIPPER_CLOSED if closed else C.GRIPPER_OPEN

    def move_to_joints(self, q_target: np.ndarray, *, max_steps: int = 4000, tol: float = 0.02) -> bool:
        """位置伺服到目标关节角，返回是否在步数内收敛。"""
        self.data.ctrl[:ARM_DOF] = np.asarray(q_target, dtype=float)
        for _ in range(max_steps):
            self.step()
            if self.joint_error() < tol:
                return True
        return False

    def solve_to_pose(self, pos: np.ndarray, rot: np.ndarray | None = None, **kw):
        """解 IK 并把结果伺服过去，返回 (是否收敛, 信息 dict)。"""
        rot = self.grasp_rot if rot is None else rot
        ok_ik, q, pos_err, rot_err = solve_ik(
            self.model, self.data, self._hand_id, pos, rot, rest_qpos=self.data.qpos[:ARM_DOF].copy(), **kw
        )
        ok_move = self.move_to_joints(q)
        return ok_ik and ok_move, {
            "q_target": q.tolist(),
            "ik_pos_err": pos_err,
            "ik_rot_err": rot_err,
            "ik_converged": ok_ik,
            "servo_converged": ok_move,
        }

    def preview_ik(self, grasp_pos: np.ndarray, rot: np.ndarray | None = None) -> tuple[bool, float]:
        """无副作用地预演一次 IK：解完立即恢复仿真状态，供安全闸门 L4 使用。"""
        rot = self.grasp_rot if rot is None else rot
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        target = self.hand_target_for_grasp(np.asarray(grasp_pos, dtype=float))
        ok, _q, pos_err, _rot_err = solve_ik(
            self.model,
            self.data,
            self._hand_id,
            target,
            rot,
            rest_qpos=self.data.qpos[:ARM_DOF].copy(),
        )
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        return ok, pos_err

    # ---------- 抓取附着（运动学）----------

    def _set_collision(self, obj: str, enabled: bool) -> None:
        gid = self._geom_ids[obj]
        self.model.geom_contype[gid] = 1 if enabled else 0
        self.model.geom_conaffinity[gid] = 1 if enabled else 0

    def attach(self, obj: str) -> None:
        """记录物体相对手爪的位姿，之后每个物理步把它钉在这个相对位姿上。"""
        R_hand = self.hand_rot()
        p_block = self.body_pos(obj)
        R_block = self.data.xmat[self._body_ids[obj]].reshape(3, 3)

        self._attach = {
            "obj": obj,
            "offset": R_hand.T @ (p_block - self.hand_pos()),
            "rot": R_hand.T @ R_block,
        }
        self._set_collision(obj, False)  # 抓取期间不参与碰撞，避免和手指穿插
        self._apply_attachment()

    def detach(self, obj: str) -> None:
        if self._attach is not None and self._attach["obj"] == obj:
            self._attach = None
        self._set_collision(obj, True)
        mujoco.mj_forward(self.model, self.data)

    def is_attached(self, obj: str) -> bool:
        return self._attach is not None and self._attach["obj"] == obj

    def _apply_attachment(self) -> None:
        assert self._attach is not None
        obj = self._attach["obj"]
        p = self.hand_pos() + self.hand_rot() @ self._attach["offset"]
        R = self.hand_rot() @ self._attach["rot"]

        adr = self._qpos_adr[obj]
        self.data.qpos[adr : adr + 3] = p
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R.flatten()))
        self.data.qpos[adr + 3 : adr + 7] = quat

        vadr = self._dof_adr[obj]
        self.data.qvel[vadr : vadr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    # ---------- 渲染 ----------

    def make_renderer(self, width: int = 640, height: int = 480) -> mujoco.Renderer:
        return mujoco.Renderer(self.model, height=height, width=width)

    @staticmethod
    def make_camera(
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        lookat=None,
    ) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth = C.CAMERA_AZIMUTH if azimuth is None else azimuth
        cam.elevation = C.CAMERA_ELEVATION if elevation is None else elevation
        cam.distance = C.CAMERA_DISTANCE if distance is None else distance
        cam.lookat[:] = C.CAMERA_LOOKAT if lookat is None else lookat
        return cam

    def render(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera | None = None) -> np.ndarray:
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()
