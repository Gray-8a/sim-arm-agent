"""用 SmolVLA 跑一次真实的视觉-语言-动作推理，并和我们的枚举解析做对比。

这个脚本要回答的问题不是"能不能驱动我们的机械臂"，而是：

  把同一帧仿真画面 + 同一条中文指令喂给一个 VLA，它输出的是什么形状的东西，
  和我们"文本 -> 枚举 -> 几何查表 -> IK"这条链路比，少了哪些中间表示。

诚实边界（写进输出里，不许含糊）：
- smolvla_base 训练时的机器人是 6 维状态 / 6 维动作形态，我们的 Panda 是 7 关节 + 1 夹爪。
  动作空间不一致，所以输出**不能**直接拿去驱动本项目的仿真。
- 输入状态直接用 Panda 前 6 个关节角，没有按数据集统计做归一化，
  所以这个数值本身没有语义，只看管线通不通。
- 这是"数据流验证"，不是"策略效果验证"。

用法：
    ./.venv-vla/Scripts/python.exe scripts/vla_infer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from arm_agent.env import ArmEnv  # noqa: E402

TASK = "把红色方块放到左边托盘"
CAMERAS = [
    (140.0, -26.0),  # 主视角，和 README 里的 gif 一致
    (90.0, -20.0),   # 侧面
    (190.0, -35.0),  # 另一侧俯视
]
IMG_SIZE = 256


def render_views(env: ArmEnv) -> np.ndarray:
    """渲染 3 个视角，返回 (3, 3, 256, 256) 的 uint8 数组。"""
    renderer = env.make_renderer(IMG_SIZE, IMG_SIZE)
    views = []
    for az, el in CAMERAS:
        cam = env.make_camera(azimuth=az, elevation=el)
        img = env.render(renderer, cam)  # (H, W, 3) uint8
        views.append(img.transpose(2, 0, 1))  # -> (3, H, W)
    return np.stack(views, axis=0)


def main() -> None:
    print("=== 1. 从仿真环境取观测 ===")
    env = ArmEnv()
    env.reset()
    env.forward()

    images = render_views(env)
    state = env.data.qpos[:6].astype(np.float32)  # Panda 前 6 个关节角
    print(f"  images: {images.shape}  dtype={images.dtype}")
    print(f"  state : {state.shape}  values={np.round(state, 3).tolist()}")
    print(f"  task  : {TASK!r}")

    print("\n=== 2. 加载 SmolVLA ===")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  loaded in {load_s:.1f}s  params={n_params/1e6:.1f}M")

    print("\n=== 3. 构造 batch（lerobot 的 observation 契约）===")
    batch = {
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": [TASK],
    }
    for i in range(3):
        img = torch.from_numpy(images[i]).float().unsqueeze(0) / 255.0  # (1,3,H,W) in [0,1]
        batch[f"observation.images.camera{i + 1}"] = img

    # 语言要自己 tokenize：select_action 不做预处理，直接要 token 张量
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    enc = tokenizer(
        [TASK],
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = enc["input_ids"]
    batch["observation.language.attention_mask"] = enc["attention_mask"].bool()

    for k, v in batch.items():
        print(f"  {k:38s} {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")

    print("\n=== 4. 前向推理 ===")
    policy.reset()
    with torch.no_grad():
        t0 = time.perf_counter()
        a0 = policy.select_action(batch)
        dt_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        a1 = policy.select_action(batch)
        dt_next = time.perf_counter() - t0

    a0 = a0 if isinstance(a0, torch.Tensor) else torch.as_tensor(a0)
    a1 = a1 if isinstance(a1, torch.Tensor) else torch.as_tensor(a1)
    print(f"  select_action #1  shape={tuple(a0.shape)}  {dt_first*1000:.0f}ms   (生成整个 chunk)")
    print(f"  select_action #2  shape={tuple(a1.shape)}  {dt_next*1000:.0f}ms   (从队列取下一步)")
    print(f"  step0 action: {np.round(a0.detach().cpu().numpy().reshape(-1), 4).tolist()}")

    print("\n=== 5. 完整动作块 ===")
    chunk = None
    policy.reset()
    with torch.no_grad():
        for name in ("predict_action_chunk", "_get_action_chunk"):
            fn = getattr(policy, name, None)
            if fn is None:
                continue
            try:
                t0 = time.perf_counter()
                chunk = fn(batch)
                dt_chunk = time.perf_counter() - t0
                print(f"  via policy.{name}   shape={tuple(chunk.shape)}  {dt_chunk*1000:.0f}ms")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  policy.{name} 失败: {type(exc).__name__}: {str(exc)[:120]}")

    if chunk is not None:
        arr = chunk.detach().cpu().numpy().reshape(-1, chunk.shape[-1])
        print(f"  chunk = {arr.shape[0]} 步 x {arr.shape[1]} 维")
        print(f"  每维范围: min={np.round(arr.min(0), 3).tolist()}")
        print(f"            max={np.round(arr.max(0), 3).tolist()}")
        print("  前 3 步:")
        for i in range(min(3, arr.shape[0])):
            print(f"    step{i}: {np.round(arr[i], 4).tolist()}")

    n_action_steps = getattr(policy.config, "n_action_steps", None)
    print(f"\n  config.n_action_steps = {n_action_steps}")
    print(f"  config.max_state_dim  = {getattr(policy.config, 'max_state_dim', '?')}")
    print(f"  config.max_action_dim = {getattr(policy.config, 'max_action_dim', '?')}")

    print("\n=== 5. 结论 ===")
    print("  输入: 3 张图 + 6 维状态 + 一句话")
    print("  输出: 一次给完整动作块（50 步），没有任何可读的中间表示")
    print("  对比: 本项目的 LLM 路径输出的是 {action, object, target} 枚举，")
    print("        可以在执行前被 Schema 校验、被闸门检查、被人工审批。")


if __name__ == "__main__":
    main()
