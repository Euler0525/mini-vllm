# vLLM examples 实践计划

这份计划把 vLLM examples 转化为开源项目的最小实例、回归样本和性能案例。目标是理解 API 合约、内部实现、性能行为与工程边界，最终能独立改写、测量、排错和讲解。它与 [vLLM 学习与求职计划](vLLM学习与求职计划.md) 共用 12 个能力里程碑，每模块约 **5–6 小时**，已计入对应里程碑 20–25 小时的总投入。项目交付与全部操作手册见 [开源推理项目实施总览](开源推理项目实施总览.md)。

硬件按 RTX 4060 Laptop 8GB、单卡规划。源码固定为 `bfdbfa1632ef638db7d201f366c6f46cc26ca2b6`；下文链接指向并排的本地 vLLM 仓库。当前 [examples 官方索引](https://docs.vllm.ai/en/latest/examples/) 与本地目录按 `basic/`、`features/` 等分类，一些旧教程中的 `examples/offline_inference/`、`examples/online_serving/` 路径已经不适用于此快照。

本文件是待执行的学习计划，没有将任何实例标记为已运行。示例中的默认大模型、远端资产、额外依赖和外部服务需要逐项核对，不能直接批量执行所有脚本。

## 优先级与掌握标准

| 优先级 | 学习目标 | 处理深度 |
| --- | --- | --- |
| P0，必做 | 离线/在线请求、动态请求、指标、prefix caching、profiling | 运行、改写、追到源码、测量、解释失败场景 |
| P1，按岗位深入 | 结构化输出、LoRA、pooling、多模态、投机解码、单机部署 | 至少选三项完成实测；其余保留阅读与最小实验设计 |
| P2，进阶 | 多卡 Ray、EP、P/D 网络传输、RL 权重更新和特殊模型 | 完成架构追踪；资源允许再运行，注明依赖与未验证内容 |

每个实例用五级记录进度：**已读 → 已运行 → 已改造 → 已测量 → 可独立解释**。P0 达到第五级才算完成。P1/P2 可以明确记录阅读深度和资源限制。一个脚本打印了文本，仅表示已运行。

每个实例都采用同一个方法：先写清输入、输出和外部依赖，运行最小 case；随后脱离原脚本独立实现一个小变体；从 example 的入口追到引擎或扩展模块；设计一个正例与反例；最后记录正确性、性能、失败边界和面试问题。同目录的多个模型变体使用差异表归纳，避免把复制运行当作掌握新机制。

## examples 全目录地图

| 目录 | 要理解的场景 | 代表实例与阅读方式 | 本机安排 |
| --- | --- | --- | --- |
| [basic](../../vllm/examples/basic) | `LLM.generate/chat`、OpenAI 兼容客户端、任务类型 | 从 generate/chat 到 completion/chat client；embed/score/classify 对照各自输出 | P0；生成和服务全部实测 |
| [features](../../vllm/examples/features) | APC、LoRA、结构化输出、投机、profiling、DP、KV events 等 | 按 X05–X10 选机制；同机制变体做对比 | P0/P1；分布式子项 P2 |
| [observability](../../vllm/examples/observability) | 离线 metrics、Prometheus/Grafana、OTel、dashboard | 先读指标，再建面板，再选 trace | P0；OTel/Perses 可选 |
| [deployment](../../vllm/examples/deployment) | 低层引擎、异步流、Helm、部署入口 | 先 engine/async 示例；Helm 读资源与探针配置 | 引擎 P0；Kubernetes 按岗位选做 |
| [generate](../../vllm/examples/generate) | 多模态、长上下文、批量 Chat、trace replay | 文本批量对照、多模态选一个模型家族；读 replay 的输入格式 | P1；百万上下文只读设计，不在 8GB 上照默认规模运行 |
| [pooling](../../vllm/examples/pooling) | embedding、classification、score/rerank、reward、token 级任务 | 先 embedding/rerank；其他任务比较输出形状和模型要求 | P1；使用对应 pooling 模型 |
| [reasoning](../../vllm/examples/reasoning) | 推理内容与最终回答、流式解析、Responses | 选一个支持的模型与 parser，其余比较协议 | P1；先阅读，再按资源实测 |
| [tool_calling](../../vllm/examples/tool_calling) | 工具描述、选择、参数解析、流式工具调用 | 先本地无副作用函数；对照 required/auto/streaming | P1；MCP 示例只读或使用自有测试服务 |
| [applications](../../vllm/examples/applications) | API server、聊天 UI、RAG 集成 | 重点识别引擎、网络、检索和 UI 的延迟边界 | P1；选一个薄集成，不扩展成大型应用 |
| [disaggregated](../../vllm/examples/disaggregated) | KV 交接、P/D、offload、connector、故障恢复 | 从顺序的 example_connector 到 recovery，再读实际连接器 | 顺序示例 P1；网络与多卡吞吐 P2 |
| [ray_serving](../../vllm/examples/ray_serving) | Ray 资源、集群、批推理、弹性 EP | 从 batch inference 到 run_cluster，区分 Ray 调度与引擎调度 | P2；单卡运行不用于证明扩展效率 |
| [rl](../../vllm/examples/rl) | pause/resume、batch invariance、权重传递、rollout | 本机优先前两项；其他按训练/rollout 的时间线读 | pause/invariance P1；分布式训练 P2 |
| [scale_out](../../vllm/examples/scale_out) | token-in/token-out 与多模态服务拆分 | 跟 client 与 server，解释 tokenization 所在位置 | P2；作为前端与引擎解耦的案例 |
| [speech_to_text](../../vllm/examples/speech_to_text) | 转录、翻译、实时音频 | 对比文件请求、实时流、麦克风客户端的协议与延迟口径 | P1/P2；模型另选，不复用文本 Qwen 权重 |
| 根目录 `*.jinja` | 普通/工具 Chat template | 对照当前 tokenizer 模板与一个工具模板，检查特殊 token | X01/X07 中学习；模板属于资产，不是独立 Python demo |

这张表覆盖当前 14 个一级场景目录。掌握机制后，剩余文件按模型、输入模态、协议和依赖做差异归纳；特殊模型、平台或协议若改变执行链，就单独建实例卡。不要为覆盖率强行下载所有权重。

## 实例卡与运行约定

实例按机制组织：可复用的变体使用 `examples/vllm/<feature>/`，配置使用 `configs/inference/`，运行数据使用 `artifacts/examples/<feature>/`，经过验证的结论整理到 `docs/reports/`。例如共享前缀实例使用 `prefix_caching`，投机解码实例使用 `speculative_decoding`；X01 等编号只用于计划索引，不用作工程目录名。下面的实例卡记录源码、配置、证据和结论。

```yaml
id: X05
source_commit: bfdbfa1632ef638db7d201f366c6f46cc26ca2b6
entrypoint: examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py
status: planned
question: 相同 token 前缀被复用时，TTFT 的变化能否由命中 token 数解释？
prerequisites: []
model_snapshot: 待填写完整 revision 和本地路径
commands: []
changes: []
source_trace: []
controls: []
correctness: []
measurements: []
failure_case: 待填写
conclusion: 待实测
limitations: []
artifacts: []
```

Python 示例在独立 vLLM 环境中运行。下面的起始命令以 vLLM 仓库根目录为工作目录，使用 `.venv/bin/python`；先按 [基线验证手册](vLLM基线验证与性能评测手册.md) 准备工程内的固定模型链接。若变体放在学习工程中，用 vLLM 环境的绝对 Python 路径运行，避免调用到 mini_llm 的另一个环境。

```bash
.venv/bin/python examples/basic/offline_inference/generate.py \
  --model ../mini-vllm/artifacts/models/qwen2.5-0.5b --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 --enforce-eager \
  --temperature 0 --max-tokens 32

.venv/bin/python examples/basic/offline_inference/chat.py \
  --model ../mini-vllm/artifacts/models/qwen2.5-0.5b --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 --enforce-eager \
  --temperature 0 --max-tokens 32
```

这两个脚本支持模型与 engine 参数。其他脚本未必支持：APC 示例硬编码了长上下文 13B 模型，profiling/metrics 示例使用 OPT，LoRA 示例使用特定 Llama 与 adapter，投机示例的默认方法是 EAGLE。先读参数和构造器，再在自己的变体中缩小配置。LoRA、视觉、音频、embedding 和 draft 模型还具有结构兼容要求，不能统一替换为 Qwen2.5-0.5B。

## 操作手册与工程落点

实例变体进入 `examples/vllm/<feature>/`，可复用测量逻辑进入 `benchmarks/inference/`，不要只积累独立脚本。下表与项目里程碑共用验收，不增加另一套时间预算。

| 实例 | 操作手册 | 项目交付落点 |
| --- | --- | --- |
| X01 | [基线验证](vLLM基线验证与性能评测手册.md) | 固定输入、三后端对照、最小离线例 |
| X02 | [在线服务与测量](vLLM在线服务与流式测量手册.md) | 流式客户端、协议与取消 |
| X03 | [动态调度](mini_llm动态调度与连续批处理手册.md) | 引擎生命周期、动态到达回放 |
| X04 | [监控与容量](vLLM监控与容量评测手册.md) | 指标采集、容量图与汇总 |
| X05 | [分页 KV 与前缀缓存](mini_llm分页KV与前缀缓存手册.md) | APC 输入、冷热对照与边界 |
| X06 | [性能分析](vLLM执行链追踪与性能分析手册.md) | 可复用 profiling 变体 |
| X07 | [算子优化与协议实例](mini_llmTriton算子优化手册.md) | 结构化输出、工具协议和失败样本 |
| X08 | [编译与图执行](vLLM编译与CUDAGraph验证手册.md) | adapter 身份与 graph 交互 |
| X09 | [SLO 调优与任务边界](vLLM面向SLO的服务调优手册.md) | pooling/多模态的独立输入输出契约 |
| X10 | [投机解码](vLLM投机解码验证手册.md) | 参数化 n-gram 实例与接受统计 |
| X11 | [KV 交接与扩展](vLLMKV交接与并行扩展手册.md) | 顺序 connector 与故障恢复 |
| X12 | [发布与独立复现](开源推理项目发布与复现手册.md) | 生命周期、实例索引与综合演示 |

## 核心实例模块

### X01：离线生成与模型任务

在基线验证里程碑中投入 5–6 小时，优先级 P0。入口为 [basic.py](../../vllm/examples/basic/offline_inference/basic.py)、[generate.py](../../vllm/examples/basic/offline_inference/generate.py)、[chat.py](../../vllm/examples/basic/offline_inference/chat.py)。

- 先运行 generate/chat，打印模板处理后的 token IDs、实际输入/输出长度、停止原因和 sampling 配置。比较裸 prompt、多轮消息、相同内容经 chat template 处理后的结果。
- 独立写一个 20–30 行的最小脚本，支持固定 snapshot、greedy、seed、`max_tokens` 和批量请求。增加输出上限、EOS、上下文边界三个 case。
- 源码从 `LLM.generate/chat` 追到输入处理、`LLMEngine`、sampler 与 output processor；与 mini_llm 的生成循环画在同一张图上。
- 对同一组 token IDs 做正确性对照。首轮计时仅用于检查实验口径，不将脚本中的打印、加载和生成总耗时混为模型吞吐。
- 阅读同目录 embed/score/classify，写出所需模型类型和输出差异，实际运行并入 X09。

验收时能脱离上游脚本写出最小示例，并解释“相同文字为什么不一定对应相同模型输入”。

### X02：在线协议与流式输出

在在线服务里程碑中投入 5–6 小时，优先级 P0。入口为 [completion client](../../vllm/examples/basic/online_serving/openai_completion_client.py)、[chat client](../../vllm/examples/basic/online_serving/openai_chat_completion_client.py)，服务配置使用主计划中的单卡模板。

- 对比 completion/chat、streaming/non-streaming。修改模型名、token 上限和一项不合法参数，记录 HTTP 状态与错误内容。
- 自己写支持连接复用的异步客户端，记录请求起点、首个有效生成片段、每个片段到达时间和结束时间。区分角色/空片段与首个内容输出。
- 从客户端追到 Chat serving、`AsyncLLM` 和结果流，说明客户端 tokenization 与服务端 tokenization 的差异。
- 用同一负载比较离线、HTTP 单并发和 HTTP 多并发，解释网络/序列化/前处理的存在，但不把所有差值简单归为网络。
- 流式输出中途断连，观察引擎是否收到取消、running 数是否回落，相关资源是否可重新分配。已缓存前缀可以继续保留，不要求 Cache 使用率归零。

验收产物是一张客户端事件时间线、一份协议差异表和一个可复现的流式取消 case。

### X03：低层引擎、异步与请求调度

在动态调度里程碑中投入 5–6 小时，优先级 P0。入口为 [llm_engine_example.py](../../vllm/examples/deployment/llm_engine_example.py)、[async_llm_streaming.py](../../vllm/examples/deployment/async_llm_streaming.py)。

- 将硬编码模型与资源配置改为自己的单卡配置。构造 A 长输入、B 短输入、C 延迟到达的请求集合，保存 request ID 和完成顺序。
- 在不修改调度策略的前提下记录每轮新加入、运行、完成、取消的请求数与调度 token 数。用小样本完成五轮手工推演。
- 从 `add_request/step` 或 `AsyncLLM` 追到 `EngineCore`、`Scheduler.schedule`、`SchedulerOutput`、`update_from_output`。核对是否走异步调度或 batch queue。
- 将这种生命周期迁移到 mini_llm 的简化调度器，比较串行、静态 batch 与动态 batch。先验证资源回收，再测性能。
- 区分 Python async、HTTP 并发、continuous batching 和异步调度；分别解释它们在哪一层起作用。

验收产物为调度表、动态到达对照与一段五分钟讲解。不能只根据 `async def` 断言 GPU 已实现并行执行。

### X04：metrics、监控与容量曲线

在监控与容量里程碑中投入 5–6 小时，优先级 P0。入口为 [离线 metrics](../../vllm/examples/observability/metrics/offline.py)、[Prometheus/Grafana](../../vllm/examples/observability/prometheus_grafana/README.md)、[dashboards](../../vllm/examples/observability/dashboards/README.md)。

- 先用离线示例理解 Counter、Gauge、Histogram，再抓一次服务 `/metrics`，查看 `HELP`、单位、labels 与实际序列名。
- 按 README 建本地监控，检查容器中 Prometheus 到宿主机 vLLM 的地址可达；示例默认地址不一定符合当前网络布局。已有 GPU 服务继续单独运行，不额外启动第二个模型容器。
- 最少展示 running、waiting、KV 使用率、抢占速率、prefix 命中率、P95 TTFT、P95 E2E 和生成 token/s。面板时间窗与压测时间窗对齐。
- 对并发 1/2/4/8 和由基线能力推算的有限到达率做两组扫描，记录实际发送速率、客户端排队和服务排队。
- 追到 `v1/metrics/loggers.py` 与 metrics 埋点，解释每个数在哪个事件上更新。OTel 另作为可选请求追踪练习。

验收时能用监控解释一个尾延迟上升 case，并交付吞吐–延迟曲线。当前版本常用的指标如下，仍需以本机 `/metrics` 为准：

| 用途 | 指标或查询要点 |
| --- | --- |
| 活跃与排队 | `vllm:num_requests_running`、`vllm:num_requests_waiting` |
| KV 使用率 | `vllm:kv_cache_usage_perc`；值为 0–1，显示百分数需乘 100 |
| 抢占 | `rate(vllm:num_preemptions_total[5m])`；Counter 导出名核对 `_total` |
| Prefix token 命中率 | `sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))`；无查询时不解读为有效命中率 |
| P95 TTFT | `histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` |
| 生成 token/s | `sum(rate(vllm:generation_tokens_total[5m]))` |

短实验若只有几十秒，5 分钟 rate 窗口会混入其他阶段；延长稳定压测或调整分析窗口。Histogram 聚合先按 bucket 汇总，不直接平均多个 rank 的 P95。服务端 histogram 的分位数是桶估计，不能要求与客户端逐请求 JSON 的分位数完全一致。

### X05：自动前缀缓存与 KV 事件

在分页 KV 与前缀缓存里程碑中投入 5–6 小时，优先级 P0。入口为 [automatic_prefix_caching_offline.py](../../vllm/examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)、[prefix_caching_offline.py](../../vllm/examples/features/automatic_prefix_caching/prefix_caching_offline.py)，可选 [kv_events_subscriber.py](../../vllm/examples/features/kv_events/kv_events_subscriber.py)。

- APC 原例使用 13B 模型，改为自己的 Qwen 小模型，缩短共享文本，显式设置上下文、显存与并发。共享长度按 tokenizer 实际 token 数构造。
- 做 APC 开/关 × 冷/热 × 同前缀/不同前缀对照。先独立预热 kernel，再重置实验 Cache 或重启服务，避免把第二次更快全归因于 APC。
- 构造共同前缀 256/1024 token，改变后缀问题；再改变前缀开头和块边界附近的一个 token，观察共享范围与命中量变化。
- 从示例追到 `get_computed_blocks`、block hash、block pool 和释放/驱逐。可选订阅 KV events，区分事件流与实际 KV 数据搬运。
- 同时测 TTFT、prefill 时间、E2E 和命中 token 数；增加输出较长的 case，解释为何 TTFT 改善但 E2E 改善有限。

验收产物必须含一个无收益反例，并用 token 级前缀与命中指标解释，而非只比较原例的两次总耗时。

### X06：profile 一个真实执行窗口

在执行链与性能分析里程碑中投入 5–6 小时，优先级 P0。入口为 [simple_profiling_offline.py](../../vllm/examples/features/profiling/simple_profiling_offline.py)、[run_one_batch_offline.py](../../vllm/examples/features/profiling/run_one_batch_offline.py)。

- 先运行简单示例理解 start/stop，再用自己的模型与独立预热。不要把第一次 `generate` 同时当作 APC、JIT、graph 与稳态基准。
- `run_one_batch_offline.py` 支持 `--profile none/prefill/decode/both`、`--profile-dir`、`--batch-size`、`--prompt-size` 等参数。注意这里 `_build_prompt` 按**字符数**构造文本，`prompt-size` 不是保证的 token 数；记录实际 token 长度。
- 在小 batch、关闭 prefix caching 的稳定对照下，分别采 prefill、decode 和混合负载。该示例按引擎 iteration 延迟采集，chunked prefill 时一个 prefill 可能跨多轮，必须检查 trace 后判断窗口是否纯净。
- 保存 PyTorch CPU/CUDA 关联 trace，再用 Nsight Systems 检查跨进程空隙、launch、memcpy 和同步。最后仅对一个热点 kernel 做 Nsight Compute 分析。
- 追到实际 runner 的 `execute_model`、backend 选择与 Qwen2 forward，标出 attention、投影、MLP、norm、sampling；不能只凭 kernel 名猜调用点。

验收产物为三张有注释的时间线截图或导出、一份热点表，以及“观察—候选原因—验证实验”的闭环。profile 采集开销与正式吞吐基准分开报告。

### X07：结构化输出、logits 与工具协议

在算子优化与协议实例里程碑中投入 5–6 小时，优先级 P1；做服务开发岗位时提高优先级。入口为 [structured outputs](../../vllm/examples/features/structured_outputs/README.md)、[logits processor](../../vllm/examples/features/logits_processor/README.md)、[tools client](../../vllm/examples/tool_calling/openai_chat_completion_client_with_tools.py)，配合 [reasoning streaming](../../vllm/examples/reasoning/openai_chat_completion_with_reasoning_streaming.py) 阅读。

- 必做一个 JSON schema 约束输出例，比较无约束/有约束；独立校验 JSON 语法、schema 和业务语义，三者分别统计。
- 选一个 logits processor，解释它在采样前怎样改变候选 token；不要直接把生成质量变化归因于模型权重。
- 工具调用使用一个本地纯函数，例如温度单位换算。写出模型输出工具名与参数、应用执行函数、工具结果回填、模型继续回答的闭环。
- 阅读 required/auto/streaming 变体，核对模型对应的 chat template、tool parser 和服务参数。小 Qwen 不支持或表现差时保留限制，不用随机输出伪装成功工具调用。
- reasoning 例重点区分思考片段、最终回答、流式 parser 和预算；运行需要对应支持的模型。追踪 structured output manager、logits processor、parser 的不同职责。

验收时能解释“格式合法”与“内容正确”的区别，并给出不受支持模型、过短输出预算或无效 schema 的失败记录。大型 reasoning/tool 模型的实测作为可选扩展。

### X08：LoRA 与引擎配置的交互

在编译与图执行里程碑中投入 5–6 小时，优先级 P1。入口为 [multilora_offline.py](../../vllm/examples/features/lora/multilora_offline.py)、[lora_with_quantization_offline.py](../../vllm/examples/features/lora/lora_with_quantization_offline.py)。同一里程碑的 CUDA Graph 主实验使用 X01/X06 的变体，避免再增加一套不相关负载。

- 先核对 base model、adapter 架构、目标模块、rank 和 revision。原例使用特定 Llama-3.2-3B 与 SQL adapter，不能只换 base model 为 Qwen 而沿用 adapter。
- 能找到小模型兼容 adapter 时，构造 base/A/B 三组请求，比较 `max_loras=1/2`、rank 和 CPU adapter 缓存的影响。没有适配权重时先阅读，保留时间完成 graph 实验。
- 原例用同一 adapter 路径创建两个不同 LoRA ID，适合观察资源与调度，不代表两个不同任务的语义能力。自己的实验明确区分 ID、缓存与真实权重差异。
- 从 `LoRARequest` 追到 LoRA manager、模型 runner 和相关算子；记录同 batch 支持的 adapter 数量与请求等待的联系。
- 可选量化变体先确认支持的 backend，比较基础与 adapter 输出、显存和服务延迟；不要把 LoRA 多租户与 TP/DP 混为一谈。

验收产物是一张兼容性与配置表、一份请求分组结果；实际未运行时明确标记仅阅读。graph 实验要同时报告冷启动与稳态，`--enforce-eager` 的组合效果不能全归为 CUDA Graph。

### X09：pooling、多模态与应用边界

在SLO 调优与任务边界里程碑中投入 5–6 小时，优先级 P1。主选 [embedding client](../../vllm/examples/pooling/embed/openai_embedding_client.py)、[rerank client](../../vllm/examples/pooling/score/rerank_api_online.py)；多模态选 [vision_language_offline.py](../../vllm/examples/generate/multimodal/vision_language_offline.py) 或 [multimodal client](../../vllm/examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)。

- 先用一个小 embedding 模型完成请求、输出维度、归一化和相似度排序验证，再用匹配的 reranker 验证 query-document 打分。两者顺序加载，避免 8GB 中同时驻留多个服务。
- 不把 embedding 吞吐记成生成 tokens/s；记录输入 token/s、请求/s、batch、长度、E2E 与输出维度，校验 padding/truncation。
- 多模态先阅读所有支持模型的分支，再选择适合显存的一个；用固定本地图片、分辨率和数量。若无法留出运行空间，完成输入处理与显存估算，标记未实测。
- 测试图片数量或分辨率变化时，记录 preprocessing、vision encoder、文本 prefill 与 decode 的边界；相同文本长度不代表相同总计算量。
- 从 pooling runner/模型与多模态 processor/encoder 跟到共享引擎路径。选一个 [RAG example](../../vllm/examples/applications/rag)，画出检索、embedding/rerank、生成三段延迟；UI 和集成不扩展成独立大项目。

验收至少完成 embedding 实测，并解释 pooling 与 autoregressive generation 的输入、输出和性能差异。语音目录进一步比较转录、翻译、实时音频的协议与实时性指标，有资源再扩展。

### X10：投机解码

在投机解码里程碑中投入 5–6 小时，优先级 P1。入口为 [spec_decode_offline.py](../../vllm/examples/features/speculative_decoding/spec_decode_offline.py)；其默认方法为 EAGLE，需要额外模型，本机先显式选择 n-gram。

- 阅读 `--method`、`--model-dir`、`--num-spec-tokens` 等参数，先构造不投机的普通生成基线，再构造 n-gram 变体。保持 target model、token IDs、dtype、采样与输出长度一致。
- 设计重复文本/代码续写与非重复问答两类负载；扫描 spec token 数 1/2/4 和并发 1/4/8，不把一次高接受率当作普遍收益。
- 记录 proposed/accepted token、每步接受长度、verify 成本、总输出吞吐、ITL 和显存。先看总时间，再用接受率解释，接受率高也可能总体变慢。
- 从 speculative 配置追到 drafter、target verification、rejection sampler、Scheduler 的 token 数更新；解释与普通逐 token decode 的差异。
- EAGLE、MTP、draft model 作为第二阶段，先确认 target/drafter/词表与 backend 兼容。8GB 中新增 drafter 的代价要计入，不默认优于 n-gram。

验收产物必须包含一个无收益或负收益 case。greedy 输出可逐 token 对照；随机采样下结合算法性质与任务指标解释，不把两次随机文本不同直接视为错误。

### X11：KV 交接、恢复与可选并行

在KV 交接与扩展里程碑中投入 5–6 小时，P1/P2。本机入口为 [example_connector](../../vllm/examples/disaggregated/example_connector/README.md)、[KV load failure recovery](../../vllm/examples/disaggregated/kv_load_failure_recovery_offline/README.md)。

- 阅读 prefill/decode 脚本和 `run.sh` 的工作目录、环境、模型、文件路径和清理行为，再准备自己的小模型配置。该示例先执行 prefill 保存 KV，再执行 decode 加载，适合单卡顺序学习，不是两个 GPU 同时服务的 P/D 部署。
- 记录 prompt、token IDs、KV 文件、模型配置和完整 revision，解释何种不匹配会使 KV 无法复用。测量保存/加载开销与重新 prefill 的成本。
- 运行正常、模拟同步加载失败、模拟异步加载失败三种模式，检查恢复后的输出、重算和请求完成。以对应 README 和 `--help` 为准，不自行猜测故障注入参数。
- 跟到 KV connector 接口与 scheduler/worker 的交接点，区分 metadata 控制流与真正的 KV 数据流。
- 有两张真实 GPU 时，再选 [data_parallel](../../vllm/examples/features/data_parallel/data_parallel_offline.py)、[torchrun](../../vllm/examples/features/torchrun) 或网络 connector。比较 TP 与两个副本时记录总 GPU 数、模型分片约束、通信和 SLO；单卡多进程不作为扩展效率证据。

验收产物为 KV 生命周期图和恢复对照。继续阅读 LMCache、Mooncake、Ray/EP，比较传输介质、额外依赖和一致性/故障责任；不需要在笔记本搭齐所有服务。

### X12：生命周期、综合服务与作品整理

在开源版本交付里程碑中投入 5–6 小时。入口包括 [pause_resume](../../vllm/examples/rl/pause_resume/pause_resume_offline.py)、[batch invariance](../../vllm/examples/rl/batch_invariance/reproducibility_offline.py)，并重用此前通过验证的客户端、监控与 profiling 变体。

- 阅读并选择资源适合的 pause/resume 或 batch invariance 例，解释暂停/恢复、请求状态和 batch 改变时的输出差异；相同 seed 并不自动保证所有执行路径逐位一致。
- RL 中权重传递、rollout 与训练交替的其他例用时序图归纳，说明什么时候更新、怎样同步、哪些进程持有权重；没有训练系统时标记仅阅读。
- 做一个最小服务演示：固定模型、流式客户端、并发负载、metrics、一次 profile、最终调优配置，并保留异常或取消请求的观察。
- 从空结果目录重新执行一个核心实验，输出正确性、吞吐/P95/goodput、显存/功耗和关键 trace；与未参与调参的负载对照。
- 给所有已研究的 examples 建索引，标明原文件、变体、状态、完整命令、结论、局限和面试问题，录制或现场演示 10 分钟。

验收时能不看原例实现最小服务，并在源码中定位两个实际行为。完成的 examples 数量仅作为索引信息，求职展示以独立理解与证据为主。

## 其余实例的补全方法

核心模块完成后，再处理尚未覆盖的文件。每次只选一个新机制，其他模型或协议变体通过差异表归纳。

| 剩余实例族 | 需要单独确认的问题 | 推荐产物 |
| --- | --- | --- |
| `features/openai_batch` | JSONL 批任务与实时 continuous batching 有什么区别？ | 一组合法/非法 JSONL 与任务输出记录 |
| `features/prompt_embed` | 直接 embedding 输入与 token 输入的长度、dtype、结构约束是什么？ | 张量输入契约；确认启用参数与模型要求 |
| `features/reset_kv`、`context_extension` | Cache 重置边界是什么？长上下文改变哪些容量与位置编码条件？ | 重置前后行为与显存估算；长上下文不按默认极限实跑 |
| `features/sharded_state`、`tensorize_vllm_model.py` | 权重保存/加载格式解决的是启动、分片还是运行时吞吐？ | 格式/依赖表，冷启动测量设计 |
| `features/structured_diffusion` | 所用模型和解码机制与普通 autoregressive 路径有什么不同？ | 独立机制卡，不能直接套用普通 decode 结论 |
| pooling 的 reward、token_classify、token_embed、plugin | 分数对应 sequence、token 还是多个向量？特殊处理器做什么？ | 输出形状与语义对照；每种新合约至少一个小样本 |
| generate 的 trace replay、batched Chat | 到达时序、原始输入和批量接口怎样影响实验口径？ | 同一负载的回放对照；区分批量 API 与引擎批处理 |
| reasoning、tool_calling 的 Responses/streaming 变体 | 事件类型、parser、工具选择和回填怎样改变？ | 协议事件差异表与失败样本 |
| speech_to_text/realtime | 输入音频时长、首个转录延迟和 real-time factor 怎样测量？ | 固定音频资产与延迟口径；不将 ASR 指标混入文本生成 |
| Ray、scale_out、Helm、SageMaker | 哪一层负责资源、路由、tokenization、探针和扩容？ | 部署图与依赖表；Helm 可先本地渲染模板检查 |
| RL 权重传递、EP、特殊 MoE | 状态一致性、通信拓扑和额外设备要求是什么？ | 权重/请求时序图，标明实测资源缺口 |
| 根目录 Jinja 和 logging configuration | 模板怎样改变 token 前缀？日志与 profile 如何对应请求？ | 模板 token 差异、日志配置与小样本 |

浏览时用 `rg --files examples` 建立当前快照清单；升级后对清单和 commit 做 diff。新文件先归入机制，再决定是否运行。这样目录扩张时仍能维护自己的学习地图。

## 结业验收与求职展示

随机抽一个 P0 example，给自己 30–45 分钟完成：说明输入输出；独立写出最小版本；指出至少两个关键源码入口；设计一个性能对照；给出一个失败或无收益场景。能完成这组动作，才具备将 example 迁移到工程任务的能力。

将实例复用到运行时和评测工具，最终至少交付六类材料：最小离线生成、流式在线客户端、动态请求生命周期、监控容量曲线、APC 对照、带注释的 profile。P1 中优先完成投机解码与顺序 KV 交接/恢复案例；结构化输出、embedding/rerank 按兼容资源运行或记录阅读与限制，LoRA 与多模态按资源扩展。examples 状态需与实际运行证据一致，首个公开版本的必需能力以实施总览为准。

讲解时围绕具体问题展开，例如“为什么同前缀第二次请求更快”“为什么提高并发后 TPOT 变差”“为什么 spec 接受率高仍未提速”。将原始 JSON、配置、源码调用链和 trace 放在一起，避免只展示运行截图或 examples 数量。

协议、指标和采集方式以 [官方 examples](https://docs.vllm.ai/en/latest/examples/)、[Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)、[Profiling](https://docs.vllm.ai/en/latest/contributing/profiling/)、[Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) 和固定版本源码为依据。性能变化与学习时间安排是本计划的实验设计，需要由自己的实际结果验证。
