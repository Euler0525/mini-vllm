# mini_llm 动态调度与连续批处理手册

本手册交付可处理动态到达请求的 mini_llm 引擎，包含确定性调度器、真实模型多请求执行、取消/回收与三策略基准。对应主计划的调度主题和 examples X03，预计 20–25 小时。前置是单请求生成与 oracle 已通过，以及 [在线测量手册](vLLM在线服务与流式测量手册.md) 的事件定义已固定。

## 实施安排

| 工作 | 时间 | 可审查的变更 |
| --- | --- | --- |
| 状态与调度契约 | 3–4 小时 | 状态图、三请求手算表 |
| 确定性 scheduler | 4–5 小时 | 假 runner、预算与资源原子性测试 |
| 真实模型 batch runner | 5–6 小时 | 每请求长度、mask、位置和批次映射 |
| 到达、取消与恢复 | 3 小时 | 生命周期事件与资源回收验证 |
| 三策略基准和 X03 | 3–4 小时 | 串行、静态、动态的同负载对照 |
| 回归与报告 | 2–3 小时 | 独立重放、架构说明、限制 |

## 先确认当前实现的限制

```bash
cd ~/Desktop/mini-vllm
rg -n 'class Sequence|transition_to|num_cached_tokens' mini_llm/engine/sequence.py
rg -n 'allocate|append|deallocate|can_append' mini_llm/engine/block_manager.py
rg -n 'length|prefill|decode|positions' mini_llm/engine/kv_cache.py mini_llm/models/qwen2p5.py
.venv/bin/python -m pytest tests/test_sequence.py tests/test_block_manager.py tests/test_generation.py \
  -q -p no:cacheprovider
```

现有 batch Cache 只有一个 `length`，attention 使用同一个 `cache_position`，不能直接支持不同上下文长度。`BlockManager.append()` 根据已增加的逻辑 token 数决定块需求；如果先改请求再发现无空间，状态可能已部分改变。新设计先计算本轮所需容量、完成预留，再提交 token/长度更新。保留旧单请求路径作为 oracle。

## 定义调度与状态契约

| 待开发位置 | 责任与最小接口 |
| --- | --- |
| `mini_llm/engine/scheduler.py` | `add_request`、`schedule`、`update`、`cancel`；只决定工作与状态，不调用模型 |
| `mini_llm/engine/batch.py` | 记录请求顺序、token 数、每请求长度/位置、有效 mask 和 KV 槽位 |
| `mini_llm/engine/model_runner.py` | 消费 batch、执行模型、返回按 request ID 对齐的 logits/采样结果 |
| `mini_llm/engine/llm_engine.py` | 驱动调度、runner、状态提交和输出事件 |
| `benchmarks/inference/replay_requests.py` | 固定输入与到达时间，在 serial/static/continuous 模式重放 |
| `tests/test_scheduler.py`、`tests/test_batch_runner.py` | 假模型生命周期与真实小配置数值回归 |

先实现 FCFS waiting/running 队列、最大活跃请求数和每轮 token budget。第一版 prefill 可以逐请求执行，decode 必须把多个活跃请求合成一次真实模型调用；仅在 Python 循环中依次调用 `decode()`，还不能算完成连续批处理的 GPU 执行。

为终态增加独立的 `finish_reason`，或审慎扩展枚举。EOS、输出上限、上下文上限、取消和失败应能区分；取消 waiting 请求也能进入终态，不强行经过实际 running。重复取消采用幂等返回，未知 ID 给出明确结果。同步第一版在 step 边界处理取消，禁止释放仍被 GPU 使用的存储。

每轮维持以下不变量：请求只能在一个队列或终态集合中；总调度 token 不超过预算；每个活跃位置有合法资源；结束资源恰好释放一次；空转必须有可说明的等待原因；失败提交前后的可见状态一致。不可容纳的单个 prompt 直接拒绝或明确等待可恢复资源，不能永久占据队首导致所有请求饥饿。

### 手推三请求

先用 budget=4、最多两条活跃请求、固定假模型输出长度。A 在第 0 轮到达，prompt=4、输出=4；B 在第 1 轮到达，prompt=2、输出=2；C 在第 2 轮到达，prompt=1、输出=2。假模型约定 prefill 最后一个 token 的 logits 产生第一个输出，后续每次 decode 产生一个输出。

下面是“优先活跃 decode，再准入完整 prefill”的一种合法预期；它是自有策略，不宣称等同 vLLM。

| 轮次 | 到达 | 调度工作与 token 数 | 轮后结果 |
| --- | --- | --- | --- |
| 0 | A | A prefill=4 | A 输出 1 |
| 1 | B | A decode=1，B prefill=2 | A 输出 2，B 输出 1 |
| 2 | C | A decode=1，B decode=1 | A 输出 3，B 完成；C 等待 |
| 3 | 无 | A decode=1，C prefill=1 | A 完成，C 输出 1 |
| 4 | 无 | C decode=1 | C 完成，资源全部可复用 |

此表采用“本轮开始时准入、结束时回收”；若选择同轮回收再准入，应调整预期并写清顺序。新生成 token 不一定已经写入 KV，需区分已生成数与已计算/已缓存数，避免多分配或遗漏最后一次追加。

## 用假 runner 验证状态机

假 runner 根据 request ID 与步数生成确定的 token；允许注入 EOS、执行异常和延迟完成，不依赖 GPU。

| 场景 | 应验证的行为 |
| --- | --- |
| 长请求运行时短请求加入 | 新请求在适当 step 获准，旧请求继续推进 |
| 同轮多个请求结束 | 每个只输出一次完成事件、各自释放资源 |
| 取消 waiting/running | 队列移除、终态原因正确、资源不重放 |
| 预算不足 | 未执行工作不改变缓存长度或 token 列表 |
| 物理容量不足 | 准入延后或明确拒绝，已运行请求状态完整 |
| 重复 ID、空输入、非法长度 | 在分配前拒绝，已有请求不受影响 |
| runner 抛异常 | 释放或隔离本轮资源，后续请求仍可执行 |
| 全部请求结束 | 引擎退出，无 busy loop、无资源泄漏 |

先提交能在 CPU 上运行的调度器和这些行为测试，再接模型。不要靠真实模型“恰好生成 EOS”覆盖生命周期分支。

## 接入不同长度的真实 batch

第一步逐请求 prefill，保存独立长度。第二步把 running 请求的一个待计算 token 打包，维护 `request_id → batch row → cache row` 映射。请求退出后压缩 batch 时，同步移动或重建映射；不允许把已退出请求的 KV 交给新请求读取。

连续 KV 过渡方案可以使用每请求容量相同的槽位池和长度向量。attention 按各请求真实长度屏蔽无效 KV；positions 来自该请求已计算长度。prefill 的 causal mask 同时考虑已有上下文、padding 与有效 query。无效 query 不得参与采样；GQA 的 KV head 数保持不变。

先验证同长度 batch，再验证 3/5/9 token 的不同长度组合。对每条请求，将同一已生成前缀喂入单请求 oracle，比较每步 logits；再做 greedy 完整输出对照。只比较最终文本可能掩盖请求串线或 padding 错误。

完整 batch 前向至少展示一轮张量形状、positions、有效 KV 长度、batch row 映射及输出 ID。通过小配置 CPU FP32 测试后，运行真实 Qwen BF16。容差沿用相邻 oracle 测试的依据，记录最大/平均误差及首个分歧，不随意放宽阈值。

## 实现同负载三策略回放

待开发回放工具先支持规范 JSONL，并复用既有 prompt IDs。下列是数据格式示例，不是已经生成的输入文件。

```json
{"id":"long","arrival_ms":0,"prompt_token_ids":[101,102,103],"max_tokens":128}
{"id":"short","arrival_ms":50,"prompt_token_ids":[201,202],"max_tokens":16}
{"id":"cancelled","arrival_ms":100,"prompt_token_ids":[301],"max_tokens":64,"cancel_after_ms":80}
```

真实实验由 tokenizer 生成合法且满足目标长度的 IDs，不直接使用示意数字。取消时间明确相对到达或发送时刻。结果保存计划到达、实际准入、首次输出、逐 token、完成/取消与排队等待；模拟轮次与真实毫秒两种模式分开。

| 策略 | 定义 | 对照要求 |
| --- | --- | --- |
| serial | 按到达顺序，前一条完成后执行下一条 | 共享同一模型加载，不反复初始化 |
| static | 固定准入窗口收集 batch，整批完成后再启动下一批 | 窗口大小固定，等待时间计入请求体验 |
| continuous | 每轮移除终态并允许新请求加入 | 相同到达序列，真实 decode batch |

实验至少包含同时到达、短请求延迟到达、20% 长 prompt 混合和取消四组。报告 makespan、成功输出 tokens/s、各长度组 TTFT/完成时间、padding token 数和峰值 KV。测量窗口从首个计划到达到最后一个终态，另列服务内部执行窗口。

不要要求动态策略对所有请求都更快；静态大 batch 可能有更高吞吐，动态策略也可能让长请求完成更晚。应展示具体权衡，并解释当 prefill 仍串行时短请求 TTFT 的限制。chunked prefill 是后续独立变更，不在第一版同时实现抢占、APC 与复杂优先级。

## 对照 vLLM 与 examples

```bash
cd ~/Desktop/vllm
rg -n 'def schedule|num_computed_tokens|token_budget|waiting|running' vllm/v1/core/sched/scheduler.py
rg --files examples/deployment | rg 'engine|async'
rg -n 'abort|add_request|step' examples/deployment
```

按 X03 独立改写最小异步引擎例，注入 A/B/C 到达与取消，记录 `SchedulerOutput`。vLLM 用统一 token 进度调度，chunking 和异步更新可能改变轮数；只比较职责与不变量，不要求与自有五轮表逐行相同。

## 交付与验收

实现完成后，为新增入口补 `--help` 和一条最小用法，更新架构文档及 `docs/reports/continuous-batching.md`。适用的 pytest 命令在文件创建后登记，运行已有全量回归并分别报告 CPU、GPU 和跳过数。

- [ ] 三请求五轮表与确定性测试一致。
- [ ] 取消、预算不足、同轮结束、容量耗尽和执行异常均可复现。
- [ ] 真实多请求 decode 经过一次 batch 模型调用，支持不同上下文长度。
- [ ] 每请求 logits/输出与单请求 oracle 对照，batch 压缩不串线。
- [ ] 三策略共享输入与到达时序，包含短请求中途到达案例。
- [ ] 完成后全部分配可复用，报告明确尚未实现的 chunking/抢占能力。

下一步用 [容量评测手册](vLLM监控与容量评测手册.md) 建服务曲线，再按 [分页 KV 手册](mini_llm分页KV与前缀缓存手册.md) 替换连续槽位。
