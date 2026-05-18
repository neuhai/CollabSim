# ShapeFactory Time-Mode 实验运行指南

本文档基于当前仓库实现，总结一套可复现、可分析的实验运行框架，覆盖两部分：
1. 可配置参数：你可以在 YAML 中手动控制哪些实验变量。
2. Controller 运行逻辑：time-mode 下事件如何入队、触发、执行、落状态。

适用入口命令示例：
- `python -m src.cli configs/shapefactory_time_mode_example.yml --run-id shapefactory_time_v1 --output-dir experiments/shapefactory_time_v1 --print-actions`

---

## 1. 配置层：你可以手动控制什么

配置文件主分区：
- `experiment`
- `prompts`
- `agents`
- `action_space`
- `controls`
- `task`
- `protocol`
- `probe`
- `logging`

下面按实验设计视角说明关键参数。

### 1.1 experiment：实验元信息与时长

常用参数：
- `id`: 实验 ID。
- `seed`: 随机种子（影响 think-time 随机、部分顺序行为）。
- `duration_sec` 或 `duration_ms`: time-mode 的仿真总时长。

说明：
- time-mode 下以仿真时间推进，不是 wall-clock 睡眠推进。
- `duration_sec` 越大，可能 wall-clock 运行越久（取决于模型调用耗时与 probing 负载）。

### 1.2 agents：agent 数量与模型配置

每个 agent 可配置：
- `id`: 参与者标识（如 A/B/C/D）。
- `model.provider`: 如 azure/openai。
- `model.name`: 模型名。
- `model.temperature`: 采样温度。

可配实验变量：
- agent 总数（增减 agents 列表）。
- 混合模型（A 用一个模型，B 用另一个模型）。
- 同模型基线（所有人相同模型，观察路径分化）。

### 1.3 action_space：允许动作集合

常见开启：
- `message`
- `produce_shape`
- `propose_trade_offer`
- `trade_response`
- `cancel_trade_offer`
- `fulfill_order`
- `do_nothing`

说明：
- 这里只决定动作是否可用，不决定策略倾向。
- 策略倾向由 prompt、状态、成本收益共同决定。

### 1.4 controls：通信模式与可见性分配

常用参数：
- `controls.communication.mode`: `direct` 或 `broadcast`。

当前推荐（你们讨论后的主线）：
- `direct` 私信模式。
- 私信仅收件人可见，非收件人看不到该消息内容。

### 1.5 task（shapefactory）：博弈经济规则

核心参数：
- `starting_money`: 初始资金。
- `regular_cost`: 非擅长形状生产成本。
- `specialty_cost`: 擅长形状生产成本。
- `min_trade_price` / `max_trade_price`: 交易报价边界。
- `incentive_money`: 每 fulfill 一个订单项奖励。
- `max_production_num`: 每 agent 最大生产次数（重要约束）。
- `shape_options`: 形状集合。
- `specialties`: 每个 agent 擅长形状映射。
- `shapes_order`: 初始任务长度（每人多少个待 fulfill 的目标）。

你关心的两个关键点：
1. `inventory` 是库存（持有形状）。
2. `fulfill_order` 会消耗库存、推进 `order_progress`、增加 money。

### 1.6 protocol：time-mode 调度行为

核心参数：
- `turn_taking`: `simultaneous` 或 `sequential`。
- `step_mode`: `time`。
- `proactive_wakeup_interval_ms`: 心跳唤醒间隔。
- `action_durations_ms`: 各动作执行耗时（仿真时间）。
- `max_concurrent_requests`: 同时并发请求上限。

最重要的可控项：
- 每个动作耗时（你提到的“每个生产 shape 时间”就在这里）。
- 心跳频率（太高会更像全员轮询，太低更依赖消息触发）。

### 1.7 probe：访谈/探针策略

支持的 cadence：
- `per_action`: 每个动作后触发。
- `per_turn`: 每轮触发。
- `on_event`: 指定事件触发。
- `per_agent_n_actions`: 每个 agent 完成 N 个动作触发（新增，用于降负载）。

新增参数：
- `every_n_actions`: 与 `per_agent_n_actions` 配套，比如 5 表示每个 agent 每完成 5 个动作触发一次。

负载估算公式（4 agent 场景）：
- `requests = 问题数 * 目标数 * (agent总数 - 1)`
- 例如 3 个问题 * 4 个目标 * 3 个回答者 = 36。

### 1.8 logging：输出追踪

常用参数：
- `output_dir`: 运行输出目录。
- `observation_events`: 是否写 observation 事件。

典型输出文件：
- `events.jsonl`
- `probes.jsonl`
- `metrics.json`（完整结束时写）
- `run_manifest.json`（完整结束时写）

注意：
- 中途 `KeyboardInterrupt` 时，常只保留部分日志（尤其可能缺 metrics/manifest）。

---

## 2. Controller 运行逻辑（time-mode）

下面是当前实现的时序化流程。

### 2.1 初始化阶段

1. 读取配置并初始化任务状态。
2. `sim_time_ms = 0`。
3. 给每个 agent 入队 `agent_wakeup(reason=start, t=0)`。
4. 入队首个 `heartbeat(t=proactive_wakeup_interval_ms)`。

### 2.2 事件队列主循环

控制器不断从最小时间戳事件出队：
- 将 `sim_time_ms` 跳到该事件时间。
- 执行对应 handler。
- 根据 handler 结果继续 enqueue 新事件。

这是一种离散事件模拟（DES），不是按 wall-clock 逐毫秒循环。

### 2.3 核心事件与触发器

#### A) agent_wakeup
触发来源：
- 启动时 start
- 心跳 heartbeat
- 私信 message trigger
- 收件箱 inbox trigger

行为：
- 仅当 agent 状态为 `idle` 才触发。
- 立即置 `busy` 并入队 `agent_decide(t)`（不再引入预思考延迟）。

#### B) agent_decide
行为：
1. 构建 observation。
2. 调模型拿 action proposal（此阶段状态为 `busy`）。
3. 校验 action（schema/precondition）。
4. 根据动作类型计算执行耗时：
   - `duration_ms <= 0` 则立即完成。
  - 否则置 `executing`，并入队 `action_complete(t + duration_ms)`。

并发语义：
- 同一 `sim_time_ms` 的一批 decide 可并发调用模型。
- 提交落账按确定顺序处理，保证可复现性。

#### C) action_complete
行为：
1. 打印/通知 `starting`（动作落地点）。
2. 真正 apply action 到 task_state。
3. 写 `state_updated`。
4. 触发 probe（按 cadence 规则）。
5. agent 置回 `idle`。
6. 如有 inbox/message，可立即触发下一次 wakeup。

#### D) heartbeat
行为：
- 扫描所有 `idle` agent，入队 wakeup(reason=heartbeat)。
- 继续入队下一次 heartbeat。

#### E) message trigger（通信动作副作用）
行为：
- 发送 direct 消息后，写 `message_delivered`。
- 收件人加入 inbox，并在当前 sim_time 触发 wakeup(reason=message)。

这就是你们讨论的“被找的人更忙”的机制来源。

---

## 3. 动作落状态语义（你关心的一致性）

### 3.1 produce_shape
- 通过后会：
  - 扣 money（按 specialty_cost 或 regular_cost）
  - 增 inventory
  - 增 production_number
- 事件：`shape_produced`，带 `money_after`。

### 3.2 propose_trade_offer / trade_response / cancel
- 创建报价：进入 pending_offers。
- 接受：执行资金与库存转移。
- 拒绝或取消：offer 状态写入 completed_trades。

### 3.3 fulfill_order
- 前提：inventory 中必须有对应 shapes。
- 成功后：
  - inventory 扣减
  - tasks 删除对应项
  - order_progress 增加
  - money 增加 `incentive_money * fulfilled_count`

---

## 4. 为什么会出现“ABCDABCD”或“看起来不均衡”

两种现象都可能出现，取决于触发主导权：

1. 更像轮询（ABCDABCD）时：
- 心跳触发占主导。
- 大家都 idle，然后同批被唤醒。

2. 更像不均衡负载时：
- 私信/交易触发占主导。
- 某个 agent 被频繁点名，会更高频被唤醒。

调参建议：
- 降低 heartbeat 频率，增强消息触发主导。
- 在 prompt 中强化“收到消息要尽快回应”的策略约束。

---

## 5. 本框架下推荐实验设计模板

### 5.1 基线（同模型）
- 全员同 provider/model，观察自发分化。
- 作用：剥离模型差异，先看机制。

### 5.2 对比（混合模型）
- A/B/C/D 绑定不同模型。
- 评估：最终 money、order_progress、成交率、沟通效率、拒单率。

### 5.3 去偏置检查
- 固定模型，轮换 agent id 与模型映射（A<->B<->C<->D）。
- 看结果是否随位置显著变化。

### 5.4 运行时长控制
- 先短跑（如 15-60 秒仿真）调参数。
- 再长跑（如 300-600 秒仿真）做正式统计。
- 中断时优先保留 events/probes，避免只看终局遗漏过程证据。

---

## 6. 一页总结（操作手册）

1. 先定任务经济学：成本、激励、生产上限、订单长度。
2. 再定调度动力学：think-time、action-duration、heartbeat。
3. 再定探针预算：cadence 与每轮请求量。
4. 跑短实验看日志行为是否符合预期。
5. 调参后再做长实验与多 seed 对比。

当前你们最实用的组合：
- `step_mode: time`
- direct 私信
- `max_production_num: 3`
- probe 用 `per_agent_n_actions` + `every_n_actions: 5`

这个组合已经能显著降低探针负载，并保持可解释的行为轨迹。
