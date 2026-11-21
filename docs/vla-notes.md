# SmolVLA 数据流实测

> 目标：不是让 VLA 控制本项目的仿真，而是搞清楚**它的输入输出长什么样**，
> 以及和本项目「文本 → 枚举 → 几何查表 → IK」这条链路比，少了哪些可以用来拦截的东西。
>
> 复现：`scripts/vla_infer.py`（需要 `.venv-vla`）

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | RTX 3050 Laptop 4GB，但 **torch 是 CPU 版**（`2.10.0+cpu`，`cuda.is_available()==False`） |
| Python | 3.10.11 |
| lerobot | 0.4.4 |
| transformers | 4.57.6 |
| 模型 | `lerobot/smolvla_base`，450 M 参数，bf16 |
| backbone | SmolVLM2-500M-Video-Instruct，裁到 16 层 |
| 隔离 | 独立 venv（`--system-site-packages`），**全局环境未改动** |

## 2. 安装过程中踩的三个坑

### 2.1 lerobot 没把 transformers 列进主依赖

`pip install lerobot` 之后 `import SmolVLAPolicy` 报 `ModuleNotFoundError: transformers`。
需要手动补。

### 2.2 transformers 5.x 直接不兼容

装最新版会拉到 5.20，然后：

```
TypeError: non-default argument 'backbone_cfg' follows default argument
```

lerobot 0.4.4 的 dataclass 定义在 transformers 5.x 下顺序错乱。同时 5.x 会把
`huggingface-hub` 拉到 2.4.0，和 lerobot 要求的 `<0.36` 冲突。

必须降到 `transformers<5` + `huggingface-hub<0.36`。

### 2.3 SmolVLM 的 processor 缺 num2words

```
ImportError: Package `num2words` is required to run SmolVLM processor.
```

这个依赖也没在 lerobot 的依赖表里。

## 3. 输入输出契约（来自 checkpoint 的 config.json）

```json
"input_features": {
  "observation.state":                {"shape": [6]},
  "observation.images.camera1":       {"shape": [3, 256, 256]},
  "observation.images.camera2":       {"shape": [3, 256, 256]},
  "observation.images.camera3":       {"shape": [3, 256, 256]}
},
"output_features": {
  "action": {"shape": [6]}
}
```

关键超参：

| 字段 | 值 |
|---|---|
| `n_action_steps` / `chunk_size` | 50 |
| `max_state_dim` / `max_action_dim` | 32 |
| `tokenizer_max_length` | 48 |
| `resize_imgs_with_padding` | [512, 512] |

注意：语言不是作为字符串传进去的，`select_action` 直接要 token 张量。
必须自己 tokenize：

```python
tokenizer = policy.model.vlm_with_expert.processor.tokenizer
enc = tokenizer([TASK], padding="max_length",
                max_length=policy.config.tokenizer_max_length,
                truncation=True, return_tensors="pt")
batch["observation.language.tokens"] = enc["input_ids"]
batch["observation.language.attention_mask"] = enc["attention_mask"].bool()
```

## 4. 实测结果

输入用我们自己仿真的三路渲染图 + Panda 前 6 个关节角 + 中文任务：

```
images: (3, 3, 256, 256)   state: (6,)   task: "把红色方块放到左边托盘"
```

输出：

| 阶段 | 结果 | 耗时 |
|---|---|---|
| `select_action` 第一次 | `(1, 6)` —— 返回 chunk 的第 0 步 | **25.6 s** |
| `select_action` 第二次 | `(1, 6)` —— 从队列取下一步 | **3 ms** |
| `predict_action_chunk` | `(1, 50, 6)` —— 完整 chunk | **25.7 s** |

也就是说：**一次前向要 25 秒，生成 50 步，之后 50 步内的取用几乎零成本。**
25 秒的开销来自 flow matching 的多步去噪（每步都要跑一次完整前向），
在 CPU 上无法实时控制。

chunk 的数值范围：

```
dim0 [0.034, 0.420]   dim1 [-0.828, 0.026]   dim2 [-0.682, 0.076]
dim3 [-2.174, -1.700] dim4 [0.148, 0.412]    dim5 [0.222, 2.897]
```

看起来像 6 个关节的绝对角度目标（SO-100 一类的 5 臂 + 1 夹爪形态）。

## 5. 为什么接不上本项目的仿真

不是"没跑通"，是动作空间不匹配：

- `smolvla_base` 输出 6 维
- 我们的 Panda 是 7 关节 + 1 个 tendon 夹爪（`nu=8`）
- 而且输出是给另一个机器人形态的绝对关节目标，硬映射过来没有物理意义

要真接上，需要：动作空间对齐（pad / 投影）+ 用本项目的数据做微调 + 每个控制周期重规划。
这是另一个项目的工作量，不在 7 天 MVP 范围内。

## 6. 最有价值的产出：两种范式的对比

| | 本项目（LLM → 枚举） | SmolVLA（VLA → 动作块） |
|---|---|---|
| 输入 | 文本 | 3 路图像 + 6 维状态 + 文本 |
| 中间表示 | `{action, object, target}`，可读、可校验 | **无** |
| 输出 | 位姿目标，几何参数由代码查表 | 50 × 6 的浮点数组 |
| 执行前能否审批 | 能 —— 四层闸门看的正是这个结构 | **不能** —— 给你一串数，没法问"你确定要抓红的吗" |
| 单次延迟 | 826 ms（DeepSeek API，46 次均值） | 25.7 s（CPU） |
| 出错时的表现 | 结构非法 → 硬阻断，可归因 | 输出一串数，无错误信号 |
| 泛化能力 | 受限于枚举空间（只有 2 物体 2 位置） | 理论上更强，但本项目没验证 |

这张表就是本项目选"LLM 出枚举"而不是端到端 VLA 的理由：

**不是 VLA 不好，是它不给可以拦的东西。**

一个 50×6 的浮点数组，没有任何字段可以问"目标是什么""为什么是这个"。
而 `{action: pick_and_place, object: red_block, target: tray_left}` 能被 Schema 校验、
能被工作空间闸门检查、能被人看一眼说不行。

如果以后要把 VLA 用进来，合理的分工是：**VLA 做底层技能（怎么动），LLM 做任务分解和闸门（该不该动）**。
本项目只做了后者。
