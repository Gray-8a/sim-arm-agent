# ROS2 概念笔记

> 状态：**概念整理，未在本机安装验证**。本机的 ROS2 环节没有进入主链路，
> 主链路是直接调用 MuJoCo Python API。这份笔记记录的是"如果要把这套东西
> 接进真实机器人栈，需要理解的三件事"。

## 三种通信原语

| 原语 | 语义 | 典型用途 | 有反馈 | 可取消 |
|---|---|---|---|---|
| **topic** | 单向广播，发布/订阅，无应答 | 关节状态、传感器流、TF | 否 | — |
| **service** | 请求-响应，同步阻塞 | 查当前位姿、切换模式、标定 | 单次响应 | 否 |
| **action** | 目标-反馈-结果，长时间运行 | 机械臂移动到指定位姿、抓取 | 是（持续 feedback） | 是 |

## 为什么"移动到 (x,y,z)"必须是 action

如果做成 service：

1. **阻塞**。调用方在整个运动过程中卡住，没法同时处理别的事（比如监控碰撞）。
2. **没有中间状态**。上层想知道"现在走到哪了""还剩多久"完全拿不到。
3. **不能取消**。发现要撞了，只能等它走完，或者粗暴断电。

action 的三段式正好对上：

```text
goal     (x, y, z, 姿态, 速度上限)
  ↓
feedback 每秒若干次：当前末端位姿、剩余距离、进度百分比
  ↓
result   succeeded / aborted / canceled + 错误码
```

对应到本项目的代码结构：`skills.pick_and_place()` 就是那个"长任务"，
`StepRecord` 就是简化的 feedback，`RunResult.status` 就是 result。

## 在真实栈里这套东西该落在哪

```text
LLM 解析        →  本项目 src/arm_agent/parser.py
四层闸门        →  本项目 src/arm_agent/guard.py（可原样搬到 ROS2 节点的服务端）
人工审批        →  本项目 src/arm_agent/approval.py（改成 Web 审批页即可）
────────────── 进程边界 ──────────────
action client   →  rclpy ActionClient.send_goal_async(...)
action server   →  MoveIt2 / 厂商控制器
```

也就是说：**闸门和审批属于上层应用，不需要理解 DDS、QoS、自定义 msg**。
真正需要 ROS2 底层知识的是 action server 那一侧，而那通常是机器人团队的工作。

## 本项目没做的

- 没装 ROS2，没跑 `ros2 run demo_nodes_cpp talker`
- 没写自定义 msg/srv/action
- 没接 MoveIt2、没做 Nav2、没碰 `ros2_control`
- 没做 DDS 配置和 QoS 调优

理由：7 天 MVP 里，装 ROS2 在 Windows 上的环境成本（WSL2 + Humble + 依赖）
会吃掉整整一天，而这套东西对"证明 Agent 编排与安全设计能力"没有增量。
