"""全局常量：场景路径、home 姿态、抓取偏移、物体/位置白名单、安全边界。

这里的每个常量都对应一个能被评测验证的假设，改动前先想清楚影响哪一层：
- HOME_QPOS / GRASP_OFFSET_LOCAL  -> 影响 IK 与执行精度
- OBJECTS / SITES                 -> 影响 LLM 输出白名单（安全闸门 L1）
- WORKSPACE                       -> 影响安全闸门 L3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENE_PATH = PROJECT_ROOT / "assets" / "scene_table.xml"

# home 关节角，取自 menagerie panda.xml 的 keyframe（上游 keyframe 已删除，
# 因为加入 freejoint 后 nq 变大导致长度不匹配，改由这里统一提供）。
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)

# 手爪张开/闭合对应的 actuator8 控制量（0..255）
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0

# 抓取点在 hand body 局部坐标系中的位置。
# hand 的局部 +z 指向指尖方向，两个手指在 +z 0.0584 处、指尖垫再延伸约 0.05，
# 取 0.105 作为"两指中间"的等效抓取点。实测校验见 scripts/probe_geometry.py。
GRASP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)

# 接近/抬升高度：抓取前先到物体上方这么多米，抓完再抬这么高
APPROACH_HEIGHT = 0.10

# ---- 白名单：LLM 只能在这两类枚举里选，坐标一律由代码查表，不由模型生成 ----
OBJECTS: dict[str, str] = {
    "red_block": "红色方块",
    "blue_block": "蓝色方块",
}

SITES: dict[str, str] = {
    "tray_left": "左侧托盘",
    "tray_right": "右侧托盘",
}

# ---- 安全闸门 L3：允许末端目标点出现的工作空间（米，世界坐标系）----
# 依据：桌面顶面 z=0，桌子范围 x∈[0,0.84], y∈[-0.42,0.42]，Panda 可达半径约 0.85。
WORKSPACE_MIN = np.array([0.15, -0.40, 0.02], dtype=float)
WORKSPACE_MAX = np.array([0.80, 0.40, 0.60], dtype=float)

# ---- 安全闸门 L2：置信度阈值 ----
CONFIDENCE_THRESHOLD = 0.75

# 仿真时间参数
TIMESTEP = 0.002
SIM_SETTLE_STEPS = 400

# ---- 演示相机（MjvCamera 自由相机模型）----
CAMERA_AZIMUTH = 140.0
CAMERA_ELEVATION = -26.0
CAMERA_DISTANCE = 1.95
CAMERA_LOOKAT = (0.40, 0.0, 0.30)
