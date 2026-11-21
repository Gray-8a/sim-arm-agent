# v0.1.0 — 结项

7 天 MVP 完成。这个项目的核心不是机械臂动得多准，而是**「不可回滚的执行」怎么做到可拦截、可审批、可审计**。

## 做了什么

- **仿真**：MuJoCo 桌面场景 + Franka Panda（menagerie），阻尼最小二乘 IK，pick / place / home 原语，放置误差 1.3 mm
- **解析**：双路径 —— 中文规则解析 + DeepSeek JSON 模式。模型只输出白名单枚举（`action` / `object` / `target`），坐标一律由代码查表，越界意图在数据结构上无法表达
- **闸门**：四层 —— 结构合法性 / 置信度阈值 / 工作空间包围盒 / IK 可达性预演。前两层中 L1、L3 是硬阻断（人也不能批），L2、L4 可转人工
- **审批**：CLI 交互 + 脚本化两种实现，每次决策写入结构化 JSONL
- **评测**：46 条分层指令集（25 正常 / 10 模糊 / 5 危险 / 6 tricky），双模式对照，`eval/compare.py` 可复现
- **VLA**：SmolVLA 450M 推理实测，验证视觉-语言-动作数据流，并与本项目范式做对比

## 关键数据

| 指标 | 值 |
|---|---|
| 单元测试 | 34 个，CI 31 秒 |
| 规则解析 | 解析 100% / 执行 100% / 拦截 100% / 误拦 0% |
| LLM 对照（deepseek-chat，46 次真实调用） | 解析 100% / 执行 100% / **拦截 93.3%** / 误拦 0% |
| 放置误差 | 1.3 mm |
| SmolVLA 推理 | 450M / CPU / chunk 生成 25.7 s / 输出 (1, 50, 6) |

## 最值得看的两个发现

1. **LLM 在否定句上也会翻车。**「别把红色方块放到左边托盘」被解析成肯定动作，最后拦住它的是低置信度，不是理解力。同一句话如果模型给了高置信度，闸门就会放行 —— 所以 LLM 的分数只能决定"要不要拦"，不能决定"能不能执行"。

2. **VLA 不给可以拦的东西。** SmolVLA 输出 50×6 浮点数组，没有中间表示，执行前无法问"你确定要抓红的吗"。这是本项目选择 LLM 出枚举而不是端到端 VLA 的真正理由。

## 已知局限

- 抓取是运动学附着（weld 方案实测数值发散），抓取期间物体不受重力
- 只有 2 个物体、2 个目标位置，白名单硬编码
- 评测集自建，与解析器存在协同偏差，100% 不能外推
- ROS2 只有概念笔记，未安装未实践
- VLA 只验证数据流，未做动作空间对齐和微调
- 没有真机，不涉及标定、延迟、硬件急停

## 复现

```bash
pip install -r requirements.txt
git clone --filter=blob:none --sparse --depth 1 \
  https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
cd third_party/mujoco_menagerie && git sparse-checkout set franka_emika_panda && cd ../..
python scripts/vendor_panda.py
PYTHONPATH=src python scripts/run_agent.py "把红色方块放到左边托盘" --yes
```

详见 [README](../README.md)。
