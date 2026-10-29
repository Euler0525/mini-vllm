# vLLM 学习与求职计划

本计划面向 AI Infra 中的推理引擎、推理性能优化和模型服务岗位，以现有 `mini-vllm` 为基础，最终交付可维护的 **mini-vllm 与推理评测工具集**。vLLM 源码、examples 和性能研究服务于项目实现与验证，求职材料来自已完成的工程成果。主线包含 12 个能力里程碑，总投入约 **240–300 小时**，硬件是 **RTX 4060 Laptop / Max-Q，8188 MiB 显存，单卡，当前功耗上限 80 W**。examples 计入同一时间预算，具体见 [vLLM examples 实践计划](vLLM-examples实践计划.md)。

逐项执行入口见 [开源推理项目实施总览](开源推理项目实施总览.md)，其中关联全部详细手册、前置条件、待开发模块、实验报告和版本验收。代码、配置与报告按功能命名，以可运行的行为推进进度。

编制日期为 2026-10-15。源码基准是本机兄弟目录 `../../vllm`，commit 为 `bfdbfa1632ef638db7d201f366c6f46cc26ca2b6`，提交日期 2026-10-14；可通过 [固定版本源码](https://github.com/vllm-project/vllm/tree/bfdbfa1632ef638db7d201f366c6f46cc26ca2b6) 在线查阅。下文的源码链接使用本地相对路径，假定两个仓库保持当前并排布局。在线 `latest` 文档用于查阅，实际实验以安装版本、启动日志和该版本 `--help` 为准。

这些文档是项目实施方案，待开发功能与实验结论需要逐项验证。编写手册没有在 GPU 上运行基准，也没有预设加速倍数。宿主机驱动状态以实际 `nvidia-smi` 信息为准，代理环境的访问限制不用于判断宿主机配置。

## 学习起点与目标

现有工程已经具备可利用的基础，先建立对应关系，再扩展缺失的系统能力。

| 现有内容 | 已具备的基础 | 与 vLLM 衔接的下一步 |
| --- | --- | --- |
| `mini_llm/models/qwen2p5.py`、`layers/`、`utils/weights.py` | Qwen2.5 前向、GQA、RoPE、SwiGLU、权重加载 | 对照融合 QKV、并行 Linear、Attention backend、权重映射 |
| `engine/kv_cache.py`、`generation.py` | 连续 KV Cache、单请求 prefill/decode、采样和停止条件 | 多请求生命周期、动态调度、分页存储与回收 |
| `engine/block_manager.py`、`sequence.py` | 逻辑 block 分配、追加与释放 | 将 block table 接入真实 KV 存储和 attention；现有管理器本身不等于已实现 PagedAttention |
| `mha_mqa_lab/` | KV 容量、逻辑带宽、微基准 | 对接真实硬件计数器，分析 decode 与 batch 的关系 |
| `tests/`、Hugging Face oracle | 算子、Cache 和生成正确性基线 | 扩展多请求、边界长度、取消、回收和优化后回归 |

完成主线后，需要能够独立展示五种能力：追踪一次请求的完整执行链；实现和验证一个简化的调度或分页机制；设计合理的服务压测；用 trace 和指标定位瓶颈；解释一项优化在何种负载下有效、何时失效。AI Infra 范围很广，本计划优先覆盖推理方向；集群运维、训练框架和 Kubernetes 深度开发放在岗位要求明确后补充。

## 时间与硬件安排

每个里程碑约 20–25 小时，按当前问题在源码阅读、实现、examples 和分析之间分配；examples 约 5–6 小时，已计入总量。源码阅读与 example 追踪可以共享笔记，避免重复计时。每完成一个里程碑，留下可演示产物、验证记录和下一项实现入口；连续读源码的时间尽量控制在两小时以内，随后用实验或手工推演检验理解。

| 条件 | 推荐工作量 | 本阶段边界 |
| --- | --- | --- |
| 当前 8GB 单卡 | 主模型用已有 `Qwen/Qwen2.5-0.5B-Instruct@7ae5576`；BF16；初始上下文上限 2048；并发从 1、2、4、8 扩展 | 固定完整 snapshot revision；同一时刻只运行一个 GPU 模型服务 |
| 单卡进阶 | 稳定后试 4096 上下文、并发 16/32；另选一个能留出运行空间的模型做迁移验证 | 1.5B 级模型作为可选项；先算权重、KV、workspace 和图捕获开销，不默认 7B BF16 能装入 |
| 多卡可选 | 具备资源时做 TP = 1/2 与两副本 DP 对照 | 记录 GPU 拓扑和总 GPU 数，不能把单卡多进程当作真正的多卡扩展验证 |
| 暂无多卡 | 读通信、分片和 connector，画执行图；先运行磁盘 KV 交接的顺序示例 | 可以展示机制理解，简历中注明尚未完成多卡实测 |

笔记本实验需固定插电状态、性能模式、驱动和后台负载，记录温度、功耗、时钟与显存占用。开始前已有约 545 MiB 显存占用，这只是一次快照；每次实验重新记录。先让温度进入稳定区间，交替执行 A/B 配置，避免把热降频误判为参数效果。80 W 笔记本的绝对性能不能直接外推到数据中心 GPU。

建议建立独立的 vLLM 虚拟环境，保留现有 mini_llm 的环境。当前工程在 `pyproject.toml` 固定了自己的 PyTorch 来源，vLLM 又有自己的二进制依赖，强行共用会使正确性与性能结果难以归因。先使用官方安装路径跑通，再按 [GPU 安装说明](../../vllm/docs/getting_started/installation/gpu.md) 和仓库 `AGENTS.md` 建立与固定 commit 匹配的 editable 环境。使用 `uv` 管理环境；在 vLLM 仓库内使用 `.venv/bin/python`、`.venv/bin/vllm`，修改 CUDA/C++ 时再遵循 [增量构建说明](../../vllm/docs/contributing/incremental_build.md)。

## 源码阅读地图

按请求路径组织阅读，每一轮只回答一个问题。先读数据结构、关键函数、调用点和对应测试，再深入异常分支；无需逐行通读整个仓库。

| 顺序 | 问题与入口 | 要追踪的内容 | 阅读产物 |
| --- | --- | --- | --- |
| A | [架构说明](../../vllm/docs/design/arch_overview.md)、[离线入口](../../vllm/vllm/entrypoints/llm.py) | `LLM.generate` 如何构造请求和驱动引擎 | 离线请求时序图，标注进程边界 |
| B | [HTTP 入口](../../vllm/vllm/entrypoints/openai/api_server.py)、[Chat serving](../../vllm/vllm/entrypoints/openai/chat_completion/serving.py)、[AsyncLLM](../../vllm/vllm/v1/engine/async_llm.py) | 参数校验、模板与输入处理、异步输出、流式返回、取消 | 在线与离线共用哪些路径、额外开销在哪里 |
| C | [输入处理](../../vllm/vllm/v1/engine/input_processor.py)、[请求结构](../../vllm/vllm/v1/request.py)、[Core client](../../vllm/vllm/v1/engine/core_client.py) | 请求 ID、token IDs、采样配置、跨进程传递 | 一份请求字段表，说明字段由谁写入 |
| D | [EngineCore](../../vllm/vllm/v1/engine/core.py) 的 `step`、`step_with_batch_queue` | `schedule → execute_model → sample_tokens → update_from_output` 及异步执行 | 一轮 engine step 的输入、输出和状态变化 |
| E | [Scheduler](../../vllm/vllm/v1/core/sched/scheduler.py) 的 `schedule`、`update_from_output`，以及 [调度输出](../../vllm/vllm/v1/core/sched/output.py) | token budget、waiting/running、抢占、chunking、完成与释放 | 三请求、五轮调度的手算表，与日志核对 |
| F | [KVCacheManager](../../vllm/vllm/v1/core/kv_cache_manager.py) 的 `get_computed_blocks`、`allocate_slots`；[BlockPool](../../vllm/vllm/v1/core/block_pool.py) | 分配、引用、空闲队列、prefix hash、命中与驱逐 | block 生命周期图，包含命中、追加、取消和内存不足 |
| G | [GPUWorker](../../vllm/vllm/v1/worker/gpu_worker.py) 的 `_make_model_runner`、`execute_model` | 模型加载、显存规划、runner 选择和执行 | 运行时配置与实际 runner 类的对应记录 |
| H | [GPUModelRunner](../../vllm/vllm/v1/worker/gpu_model_runner.py) 或 [新版 GPUModelRunner](../../vllm/vllm/v1/worker/gpu/model_runner.py) | batch 打包、positions、slot mapping、attention metadata、forward 和采样 | 标注真实形状的张量流图 |
| I | [Qwen2 模型实现](../../vllm/vllm/model_executor/models/qwen2.py)、[backend 选择](../../vllm/vllm/v1/attention/selector.py)、[FlashAttention backend](../../vllm/vllm/v1/attention/backends/flash_attn.py) | Qwen2.5 的模型注册映射、融合投影、权重装载、KV 写入与读取 | mini_llm 与 vLLM 的模块对照，以及实际 kernel 调用路径 |
| J | [Sampler](../../vllm/vllm/v1/sample/sampler.py)、[输出处理](../../vllm/vllm/v1/engine/output_processor.py) | greedy/top-k/top-p、停止条件、输出汇集和 detokenize | 从 logits 到客户端文本的路径说明 |
| K | [CUDA Graph 设计](../../vllm/docs/design/cuda_graphs.md)、[编译代码](../../vllm/vllm/compilation)、[指标实现](../../vllm/vllm/v1/metrics/loggers.py) | 图捕获、运行时调度、编译开销、计时边界 | 一张性能指标与源码埋点对应表 |
| L | [并行状态](../../vllm/vllm/distributed/parallel_state.py)、[通信操作](../../vllm/vllm/distributed/communication_op.py)、[KV transfer](../../vllm/vllm/distributed/kv_transfer) | TP/DP/PP/EP 的进程组、通信与跨实例 KV 交接 | 一份可选多卡实验设计和通信成本估算 |

这个版本同时存在两套 GPU runner，先看 `_make_model_runner` 和运行时配置，再选实际执行的一套阅读。`v1/worker/gpu/` 的 README 仍标为实验性，不应仅凭文件名认定默认使用它。V1 引擎与 Model Runner V2 是不同层次的概念。

Scheduler 的源码用 `num_computed_tokens` 与待处理 token 数统一调度，未按两套独立 prefill/decode 循环实现。可以先用两阶段心智模型入门，随后必须把它映射到真实的 token budget 与请求状态。`EngineCore.step` 便于理解基本闭环，实际部署也可能走 batch queue 或异步调度路径，需要从调用点确认。

对应测试优先阅读 `tests/v1/core/test_scheduler.py`、`test_async_scheduler.py`、`test_prefix_caching.py`、`test_single_type_kv_cache_manager.py`，以及 `tests/v1/sample/test_sampler.py`。不同测试的 GPU 与依赖要求不同，先阅读 fixture，再选择能在本机运行的子集。

每次源码阅读按四步执行：先用 `rg -n` 找类、函数及调用点；写下输入/输出字段与状态不变量；用一个最小请求在实际执行路径记录值；最后用测试或自己构造的边界 case 检验理解。跨进程追踪时标明 PID、request ID 与 engine step，断点或日志放在真正执行的 worker/engine 中。调试日志与逐 token 打印只在小样本运行中开启，正式压测恢复干净配置。

例如两个请求的 prompt 分别为 64 和 128 token，观察 prefill 如何把输入组织成 token batch，positions 与请求边界如何表达，KV slot 如何映射；随后观察两个活跃请求各生成一个 token 的 decode batch。启用 chunking、APC 或投机后，实际调度 token 数会变化，必须重新从 `SchedulerOutput` 核对，不能一直假定“decode 的 token 数等于请求数”。

## 项目推进路线

| 详细手册 | 主问题 | 源码与实现 | 实验和 examples | 验收产物 |
| --- | --- | --- | --- | --- |
| [基线验证](vLLM基线验证与性能评测手册.md) | 请求如何从 API 到生成结果？ | 路线 A–C；核对已有能力；固定环境与模型 | X01；离线生成；20 个固定 prompt 对照 | 环境清单、调用图、差异记录 |
| [在线测量](vLLM在线服务与流式测量手册.md) | 一轮引擎执行做什么？ | 路线 D、J；单请求分阶段计时、流式客户端 | X02；HTTP、离线与 mini_llm 基线 | 事件 schema、取消记录、三路径比较边界 |
| [动态调度](mini_llm动态调度与连续批处理手册.md) | 多请求如何共享 GPU？ | 路线 E；状态机、确定性调度器、真实 batch runner | X03；serial/static/continuous 对照 | 调度表、不同长度、取消/耗尽回归 |
| [监控与容量](vLLM监控与容量评测手册.md) | 为什么尾延迟随负载上升？ | 调度与指标埋点；监控与汇总工具 | X04；并发扫描和到达率扫描 | 吞吐–P95 曲线、饱和分析、首次展示 |
| [分页 KV](mini_llm分页KV与前缀缓存手册.md) | 分页与 prefix caching 怎样工作？ | 路线 F；物理存储、block table、参考 attention | X05；冷/热前缀、边界、复用与驱逐 | oracle、显存估算、缓存收益和反例 |
| [执行链与 profile](vLLM执行链追踪与性能分析手册.md) | CPU、GPU 各在等待什么？ | 路线 G–I；实际 backend 与 runner | X06；PyTorch Profiler、Systems，分离 prefill/decode | trace、前三项开销、两个瓶颈闭环 |
| [Triton 算子](mini_llmTriton算子优化手册.md) | 哪个底层操作值得优化？ | RMSNorm 或 KV 写入热点；oracle、集成开关与回退 | X07；形状扫描、数值验证、端到端回测 | kernel、适用范围与无收益说明 |
| [编译与 graph](vLLM编译与CUDAGraph验证手册.md) | 编译与图执行改变了什么？ | 路线 K；capture、dispatch 和 shape 限制 | X08；eager/优化、启动/稳态对照 | 消融、trace、显存、第二次展示 |
| [SLO 调优](vLLM面向SLO的服务调优手册.md) | 服务参数怎样围绕 SLO 调整？ | token budget、并发、Cache；可选量化 | X09；分阶段扫参、保留负载、质量门槛 | Pareto 图、最终配置、退化场景 |
| [投机解码](vLLM投机解码验证手册.md) | 多 token 验证何时有收益？ | draft/verify、rejection sampling；先 n-gram | X10；重复/非重复负载，低/高并发 | 接受量、ITL、吞吐与显存联合分析 |
| [KV 交接与扩展](vLLMKV交接与并行扩展手册.md) | 并行与 P/D 的成本在哪里？ | 路线 L；单卡顺序 connector、失败恢复 | X11；正常/同步失败/异步失败，可选 TP/DP | 生命周期、恢复与 IO/重算对照 |
| [开源版本交付](开源推理项目发布与复现手册.md) | 新读者能否安装、使用并复现？ | 包、CI、贡献文档、许可证、关键回归 | X12；新结果目录重现报告、综合演示 | 可维护版本、实例索引与证据化简历 |

容量报告完成后开始展示项目，性能分析完成后可以尝试投递并用反馈调整深度。进度较慢时先打通单请求、动态调度、分页和评测，再完成有正确性与回退的算子实验及必需验证案例；多卡、量化和额外模型按资源选做。公开开发中状态不等于版本验收通过。进度较快时，把节余时间用于重复测量、反例和独立复现。

## mini_llm 的三个衔接任务

### 动态调度器

先用假模型构造可确定的请求生命周期，完成 FCFS waiting/running 队列、每轮预算、请求结束移除、新请求加入、取消和显存不足处理。然后在真实小模型上接入多请求 decode，支持不同 prompt 长度。第一版允许用逐请求 prefill 和简单 batch packing，明确它与 vLLM 的差距，随后再做 chunked prefill。

必须覆盖“短请求在长请求运行中到达”“不同请求同时结束”“取消后资源可复用”“预算不足时状态不被部分破坏”。用同一请求集合比较串行、静态 batch 和动态 batch 的 makespan、各请求 TTFT、完成时间与填充开销。连续批处理的价值需要由动态到达实验体现，单次把列表交给 `generate` 不足以验证。

### 分页 KV 参考实现

保留当前连续 Cache 作为 oracle，增加物理 block 存储与每请求 block table。先用 PyTorch gather/scatter 拼出参考 attention，验证逻辑 token 到物理槽位的映射，再考虑直接读取分页布局的 Triton kernel。参考版的目标是机制与正确性，它可能更慢。

重点测试长度为 `block_size-1`、`block_size`、`block_size+1` 的 prefill/decode，跨块追加，多请求交错分配，释放后重新分配，容量耗尽和取消。prefix 共享先作为第二阶段，加入引用计数和共享块写入语义；不要在第一版同时实现驱逐、copy-on-write 和分布式传输。

### 一个有证据的 kernel 优化

从现有 `RMSNorm` 或 trace 里的其他小热点选一个。建立 PyTorch oracle，覆盖本模型真实维度、不同 token/batch 数、连续与支持的非连续布局，以及 BF16 数值误差。计时用 CUDA events 或合适的基准工具，预热并正确同步；不要把异步 launch 的 CPU 时间当成 GPU 执行时间。

先报告微基准，再报告 mini_llm 端到端变化。若计划修改 vLLM，先确认它的实际路径尚有同一瓶颈，避免用自己的 kernel 去比较框架已经融合的另一种操作。解释 Amdahl 定律：即使某算子加快一倍，端到端收益仍受该算子原先占比约束。

## 实验规范与指标

每次实验先写“假设—控制变量—测量项—判断条件”，再运行。工程按功能组织，时间安排只保留在计划中。评测代码和配置可以跨阶段复用；生成数据与运行结果放在 Git 忽略的 `artifacts/`，经过验证的报告再整理进文档。

```text
benchmarks/inference/        # 输入准备、跨后端对照与推理评测工具
configs/inference/           # 按硬件和运行场景命名的服务配置
docs/reports/                # 经过验证的分析报告，按主题逐步补充
artifacts/                   # 本地运行产物，由 Git 忽略
  models/                    # 指向固定 snapshot 的本地模型链接
  inference/
    inputs/                  # 固定 token IDs、长度分布、seed 与校验值
    logs/                    # 环境、安装、启动与诊断记录
    results/<scenario>/<run>/ # 原始输出、bench JSON 和实际配置
    traces/                  # 原始 profile，仅整理后的小样本进入报告
```

manifest 还需记录操作系统、Python/PyTorch/vLLM 版本，CUDA runtime、GPU 驱动、模型与 tokenizer 的完整 revision、dtype、sampling、chat template、prefix caching、chunked prefill、编译和 CUDA Graph 配置。`nvidia-smi` 显示的 CUDA UMD 版本、`nvcc` 的 Toolkit 版本、PyTorch 使用的 CUDA runtime 分别记录，不能视为同一个版本。

| 指标 | 口径 | 使用方式与陷阱 |
| --- | --- | --- |
| 客户端 TTFT | 从客户端开始请求到首个有效生成输出；具体起点检查客户端实现 | 包含服务前处理、排队和传输；不等于 GPU prefill 耗时 |
| TPOT | 通常按请求计算 `(E2E - TTFT)/(输出 token 数 - 1)`；只对输出超过一个 token 的请求定义 | 说明排除规则，再汇总分布；不等于逐间隔 ITL 的相同统计 |
| ITL | 客户端观察到的连续生成输出间隔，注意 chunk 与 token 的映射 | 同一网络 chunk 可能带多个 token；与 GPU decode step 时长分开解释 |
| E2E | 一次请求从开始到完成的时长 | 流式与非流式结果分别报告，说明是否含客户端等待队列 |
| 输出吞吐 | 成功请求的生成 token 总数 / 测试窗口 | 与输入+输出总 token/s、请求/s 分开，记录失败和超时 |
| Goodput | 同时满足预先设定 SLO 的成功请求数 / 测试窗口 | 先定义 TTFT、TPOT 等门槛；不能只追求最大 tokens/s |
| 排队与抢占 | waiting/running 数、queue time、preemption 增量 | 用来解释负载超过容量、KV 紧张和重算成本 |
| KV 指标 | Cache 使用比例、可分配容量、prefix token 命中率 | 使用比例高不必然等于当前有大量活跃请求；缓存块可能可驱逐复用 |
| GPU 指标 | 显存、功耗、温度、SM 活跃、DRAM 吞吐、kernel 时间 | `nvidia-smi` 的 GPU-Util 不是 SM 吞吐或带宽利用率，也不反映工作是否有用 |
| 冷启动 | 进程启动到可服务时间，区分权重加载、编译、capture 和 warmup | 与稳态性能分别报告；实验工具首次编译时间也单列 |

客户端和服务端指标的测量边界不同，不能直接相减就称为“网络耗时”。可通过请求 ID、时间线和埋点逐步核对。服务指标以实际 `/metrics` 的 `HELP`、单位和标签为准，详见 [官方 Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)。

测量时先预热，每个筛选配置重复 3 次，最终候选至少 5 次，并交替运行基线与候选。200 个请求用于初筛；P99 需要更充分样本，最终结果建议至少 1000 个成功请求并覆盖多次运行，必要时延长稳定测试窗口。保留各次结果和中位数、离散范围，不只保留最好的一次。性能结论需在一个未参与调参的长度分布上再验证。

正确性分为两类。基础生成在固定 token IDs、revision、dtype、采样和停止条件后比较 logits 误差、top-k 与输出；greedy 仍可能因数值细节产生分歧，需要定位第一个不同 token。量化、投机解码等改变数值或执行策略的实验，另设任务集与质量指标，不要求所有情况下文本逐字相同。基础集合先用 20 条，最终扩到 50–100 条，包含长短输入、中文、代码、EOS 和上下文边界。

## 第一份可运行基准

以下命令是 **未来在宿主机执行的起始模板**。先完成独立环境安装，并确保 `vllm`、bench 依赖与所读源码版本对应；本次没有执行安装或压测。运行目录为 vLLM 仓库根目录，先检查 `.venv/bin/vllm --version`、`.venv/bin/vllm serve --help`、`.venv/bin/vllm bench serve --help`，用 `.venv/bin/python -c 'import vllm; print(vllm.__file__)'` 确认导入位置。editable 源码与预编译二进制必须按官方流程匹配，不能仅修改导入路径就认为运行版本一致。

先按 [基线验证手册](vLLM基线验证与性能评测手册.md) 将固定 snapshot 关联到 `artifacts/models/qwen2.5-0.5b`，登记完整 revision。以下命令仍从 vLLM 仓库根目录运行，直接使用该模型链接。先用人为简化的 eager、关闭 APC 配置建立容易解释的基线，随后另测官方默认优化配置。

```bash
.venv/bin/vllm serve ../mini-vllm/artifacts/models/qwen2.5-0.5b \
  --served-model-name qwen2.5-0.5b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.70 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enforce-eager
```

服务就绪后在另一终端运行，结果保存到学习工程中。这里的 SLO 仅作为学习用门槛，后续按实际交互需求修改，不作为行业标准。

```bash
.venv/bin/vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen2.5-0.5b --tokenizer ../mini-vllm/artifacts/models/qwen2.5-0.5b \
  --dataset-name random \
  --input-len 128 --output-len 128 --random-range-ratio 0.0 \
  --num-prompts 200 --num-warmups 10 \
  --request-rate inf --max-concurrency 4 \
  --temperature 0 --ignore-eos --seed 42 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --goodput ttft:1000 tpot:80 \
  --save-result --save-detailed \
  --result-dir ../mini-vllm/artifacts/inference/results/baseline \
  --result-filename qwen05b-isl128-osl128-c4-run1.json
```

这个配置测量固定并发下的生成能力。要模拟外部流量，单独扫描有限 `--request-rate`，检查客户端实际发送速率与客户端排队；过低的 `--max-concurrency` 会限制流量，不能据此宣称服务器能够承受设定 QPS。固定长度使用 `ignore_eos` 便于比较算力消耗，但还需一组正常 EOS 的真实请求测试。

若初始配置无法启动，保留启动日志，先降 `max_num_seqs`、长度和 token budget，检查可用显存与实际 backend；不要只反复提高 `gpu_memory_utilization`。后续所有长上下文 case 都需确保输入和输出之和不超过 `max_model_len`。

## 核心实验矩阵

以下是分阶段实验集，无需做完整笛卡尔积。先固定模型和 dtype，测一个变量；找到主要瓶颈后再测交互。所有 case 均保留正确性检查、失败率和完整配置。

| ID | 假设与对照 | 4060 8GB 起始范围 | 重点测量与预期产物 |
| --- | --- | --- | --- |
| L01 | mini_llm 与 vLLM 在相同输入下有可解释差异 | batch = 1；20 条固定输入；输出 16/64 token | 首个不同 token、logits/top-k、prefill/decode；先解决输入和停止口径 |
| L02 | batch 增加可摊薄部分权重读取与调度开销 | ISL/OSL = 128/128；并发 1/2/4/8，再试 16 | 输出吞吐、P95 TTFT/TPOT、温度和显存；识别饱和点 |
| L03 | 长 prefill 与短 decode 混合会改变尾延迟 | 128/128、1024/64、128/512；另构造 20% 长 prompt 的混合到达 | TTFT/ITL 分组分布、排队和调度 trace；长度不能只看均值 |
| L04 | chunked prefill 的 token budget 存在权衡 | 启用 chunking；budget = 256/512/1024/2048；固定并发 | 长请求 TTFT、短请求 ITL、总吞吐；关闭 chunking 的对照需满足版本的 budget/长度校验 |
| L05 | 共享前缀主要节省重复 prefill | 共享前缀 0/256/1024 token；冷/热；APC 开/关 | TTFT、token 命中率、prefill 时间；不同前缀和 decode 密集负载作为反例 |
| L06 | KV 紧张引发抢占和重算，吞吐可能下降 | 正常 Cache 对照 128/256 MiB 的显式 KV 预算；在可启动范围逐步增加并发与总长度 | preemption、queue time、E2E、完成率；不要把可恢复抢占等同于 OOM |
| L07 | eager 与编译/graph 优化对小模型的影响不同 | 相同形状下 eager、版本默认优化；再用受支持配置隔离 graph 和 compile | 启动耗时、capture 显存、launch 间隙、稳态 ITL；不能把组合开关收益全归因于 graph |
| L08 | 一个算子热点的优化能否传递到服务端 | 实际模型维度；小/中 token batch；PyTorch 与 Triton | kernel 时间、误差、DRAM/SM 证据、端到端变化和 Amdahl 分析 |
| L09 | 参数最佳点随负载变化 | `max_num_seqs` = 4/8/16；budget = 512/1024/2048；显存比例先固定，之后试 0.60/0.70/0.80 | 先一维筛选，再挑少量二维组合；输出吞吐–P95 曲线与 SLO 内 goodput |
| L10 | 权重量化、KV 量化解决不同问题 | 可选；只选当前模型、Ada GPU 与实际 backend 支持的方案 | 权重/KV 显存分别记、任务质量、TTFT/ITL；不预设 FP8/INT4 一定更快 |
| L11 | 投机解码收益取决于接受率和验证开销 | 先 n-gram；spec token 数 1/2/4；重复与非重复输入；并发 1/4/8 | 接受率、每步接受长度、总输出吞吐、ITL、额外显存；必要时解释负收益 |
| L12 | 多卡/外部 KV 交接的收益可能被通信吞没 | 本机先做顺序 connector；多卡时 TP = 1/2、两副本 DP，或 P/D | KV 字节数、传输与重算时间、TTFT、总 GPU 成本；不能把演示 connector 当作低延迟网络基准 |

L06 可以使用该版本的 `--kv-cache-memory-bytes` 限制 KV 池；它显式设定 KV 容量，与基于 `gpu_memory_utilization` 的自动推算需要分别解释，并记录实际生效值。先保证单请求长度可容纳，再制造并发压力。

容量推导与现有 MHA/MQA 实验连接。对普通全注意力 decoder、相同 KV dtype、无 TP 分片的近似容量为

$$
M_{KV}= 2L H_{KV}D_h s\sum_i T_i.
$$

Qwen2.5-0.5B 的 `L=24`、`H_KV=2`、`D_h=64`，BF16 的 `s=2` 字节，因此每 token 约 12 KiB，单请求 4096 个缓存 token 约 48 MiB。分页后用 `block_size × Σceil(T_i/block_size)` 代替总 token 数估算分配量；再加 block metadata、权重、激活、workspace 和 graph 内存。这个公式描述存储量，不能直接当作 Nsight 测得的 DRAM 流量，也不直接适用于 MLA、混合状态模型或所有并行布局。

## 性能工具使用顺序

先从服务指标找现象，再用系统时间线定位等待，最后用 kernel 工具解释计算与访存。不要从单个 kernel 的 occupancy 推断整个服务的瓶颈。

| 工具 | 用途 | 要保存的证据 |
| --- | --- | --- |
| `vllm bench serve` | 客户端吞吐、延迟、goodput、实际负载 | 完整参数、详细 JSON、成功/失败数、实际 token 长度 |
| Prometheus + Grafana | 排队、KV、抢占、服务阶段延迟 | dashboard、测试窗口、PromQL 与原始指标样本 |
| `nvidia-smi` | 温度、功耗、时钟、显存和粗粒度利用率 | 与压测窗口对齐的时间序列；不能用它替代硬件带宽计数器 |
| PyTorch Profiler | CPU/PyTorch 算子与 CUDA kernel 对应 | 预热后少量请求的 trace、关键算子表、prefill/decode 标注 |
| Nsight Systems | 多进程 CPU/GPU 时间线、launch、memcpy、同步与空隙 | 指明 PID/rank、关键等待链和对应请求区间 |
| Nsight Compute | 一个或少量热点 kernel 的 DRAM、SM、occupancy、stall 和 roofline | kernel 名称、形状、采集 section、资源与吞吐数据 |
| OpenTelemetry，可选 | 请求在前端、引擎及其他服务间的追踪 | request/trace ID 与时间范围；它不能替代 GPU kernel profiling |

当前版本用 `--profiler-config` 配置内置 profiler，再通过 `bench --profile` 或 profiling API 控制窗口。先读 [官方 Profiling vLLM](https://docs.vllm.ai/en/latest/contributing/profiling/)；旧教程中的环境变量不默认适用于此 commit。可在起始服务配置中增加以下参数，使用独立结果目录，并将采集请求缩小到 2–10 个：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"../mini-vllm/artifacts/inference/traces/torch"}'
```

启动后先完成普通预热，再用带 `--profile` 的小批量 bench 采集；确认 trace 已刷新再打开 Perfetto 或兼容查看器。采集期间的性能数字用于诊断，不与未开启 profiler 的基准混合。

Nsight Systems 优先采用该版本官方文档中的 CUDA profiler 控制方式，把采集窗口限制在稳态请求内；确认引擎和 GPU worker 子进程已被捕获，否则只看到 API server 容易误判。Nsight Compute 则先从 Systems 中找到热点，按 kernel 名称和 launch 范围只采少量实例。使用 `--help` 核对本机版本的子进程和 graph 支持；若硬件计数器权限不足，保留限制说明，继续做已有时间线分析，不把缺少计数器伪装成测量结果。参考 [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 和 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

安装好兼容版本的 Nsight Systems 后，可按下面的起始模板采集。先结束此前自己的测试服务，保持端口和 GPU 上只有这一份模型实例；模型路径沿用前文。这里关闭 CPU sampling，先采 CUDA、NVTX 和 OS runtime 时间线。

```bash
mkdir -p ../mini-vllm/artifacts/inference/traces

VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
  --trace=cuda,nvtx,osrt --sample=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output=../mini-vllm/artifacts/inference/traces/nsys-qwen \
  .venv/bin/vllm serve ../mini-vllm/artifacts/models/qwen2.5-0.5b \
  --served-model-name qwen2.5-0.5b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 2048 \
  --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.70 --enable-chunked-prefill \
  --no-enable-prefix-caching --enforce-eager \
  --profiler-config '{"profiler":"cuda"}'
```

在另一终端先做不带 `--profile` 的普通预热，再沿用前面的 bench 命令，改为 `--num-prompts 4 --num-warmups 0 --profile` 并使用新的结果文件名。完成采集后正常结束自己的服务，让报告落盘；查看实际生成的 `.nsys-rep`，通过 `nsys stats <报告路径>` 和 GUI 对照分析。后续测 graph 时另外保存配置与报告，不能覆盖这份 eager 基线。

每份 profile 报告回答五个问题：时间花在哪个进程/阶段；GPU 是否连续执行；热点是 launch、计算、访存还是同步；哪项指标支持判断；下一项实验如何推翻这个判断。降低 profiler 的采集范围，优先于打开所有事件。

## 调优决策

| 观察现象 | 候选解释 | 下一步验证 | 优先尝试的改变 |
| --- | --- | --- | --- |
| 并发增加后 TTFT 急升、waiting 持续增长 | 到达率超出服务能力，或长请求阻塞 | 实际到达率、queue time、长度分组结果 | 限制准入、调整 token budget、chunking；给出容量上限 |
| 长 prefill 出现时短请求 ITL 恶化 | 混合 batch 变长、共享执行资源 | 相关时间线和实际调度 token 数 | 缩小 prefill chunk；对照总吞吐是否下降 |
| GPU 有明显空隙且 CPU 忙 | tokenization、调度、Python 开销或客户端供给不足 | 分别观察客户端、前端、engine core 的 CPU 与 IPC | 复用客户端连接、修正压测供给、排查 CPU 热点，再研究编译/graph |
| GPU 忙但生成吞吐低 | 小 batch 的权重读取、长 KV 读取、低效 kernel 等 | 实际 backend、kernel 时间与硬件计数器 | batch、上下文/Cache、支持的 backend 或量化；不只看 GPU-Util |
| Cache 高且抢占频繁 | 并发与长度超过 KV 容量 | Cache 预算、活跃 token 数、重算与抢占增量 | 降低并发/长度、调整 Cache 预算，必要时评估 KV 量化 |
| 热前缀几乎无收益 | token 前缀不一致、对照未预热、驱逐或负载由 decode 主导 | token IDs、命中 token 数、prefill 占比 | 固定模板、构造冷/热对照、区分 TTFT 与 E2E |
| eager 比默认配置更快 | 编译/capture 未摊薄、shape 不稳定或路径不同 | 启动与稳态分开、核对 backend/runner | 预热后复测；用受支持配置逐项隔离，保留负收益 case |
| 结果跨次明显漂移 | 热降频、功耗、后台进程、请求集合不同 | 功耗/时钟/温度、负载校验值 | 固定条件、A/B 交替、重复测量 |

`max_num_seqs` 是引擎侧可调度序列数，客户端并发是外部在途请求数，两者分别控制并记录。显存比例也不是当前 KV 使用率。量化、增加并发、换 attention backend 和开启投机解码都需要工作负载与支持条件，不能作为无条件的优化口诀。机制说明可结合 [官方 Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) 阅读，最终配置由本机实验选择。

## 求职作品与验收

最终在同一开源仓库保留两个完整交付部分，examples 作为可检索材料。结果允许无显著加速，但必须能安装、复现和解释；各部分以 [实施总览](开源推理项目实施总览.md) 的验收标准为准。

| 作品 | 内容 | 验收标准 |
| --- | --- | --- |
| mini_llm 推理运行时 | 动态调度、真实多请求执行、分页 KV 参考实现、一个带回退的 Triton kernel | 关键边界与 oracle 通过；可安装和演示；有前后对照与支持范围；能解释与 vLLM 的差距 |
| 推理评测工具集 | 负载、回放、流式测量、监控、profile、SLO 调优及机制案例 | 原始结果可追溯；至少两个瓶颈分析；默认/最终配置对照；含一个无收益案例与保留负载验证 |
| examples 实验索引 | 场景分类、入口、依赖、源码追踪、改造和运行结果 | 标明已运行、已改造、已测量或仅阅读；挑任意核心实例可从配置复现 |

在容量基线、编译/graph 验证、公开版本候选完成后分别演示。首次展示一次请求、一个监控面板和容量曲线；随后解释一个瓶颈及其优化；最终从新结果目录复现一个主结论，并在 10 分钟内讲清问题、证据、改动、收益、代价和适用边界。

面试自查问题至少包括：为什么 prefill 与 decode 的瓶颈可能不同；GQA 怎样改变 KV 容量；PagedAttention 与 FlashAttention 分别解决什么问题；动态 batch 为什么影响吞吐和尾延迟；APC 为什么不能消除后续 decode；V1 如何处理 token budget 和重算；CUDA Graph 与编译各解决什么开销；为什么 GPU-Util 高仍可能低效；量化与投机解码何时变慢；单卡优化怎样迁移到 TP/DP，以及哪些结论不能迁移。

简历在实测后填写，例如：“基于固定版本 vLLM，在 RTX 4060 8GB 上建立覆盖三类长度分布的推理基准；通过 〔 证据 〕 定位 〔 瓶颈 〕，在 〔TTFT/TPOT SLO〕 内将 goodput 从 〔 实测 〕 调整至 〔 实测 〕，同时记录 〔 质量/显存/启动代价 〕。”只有真正运行和验证过的内容进入经验描述。

若后续做开源贡献，先选已复现且能独立解释的问题，查重并遵循 vLLM 仓库 `AGENTS.md` 的人工 review、测试和 AI 辅助披露要求；本计划不以提交低价值 PR 作为毕业条件。

## 基线阶段行动

逐日执行步骤、完整命令和统一对照脚本见 [vLLM 基线验证与性能评测手册](vLLM基线验证与性能评测手册.md)。

| 时间 | 任务 | 完成标准 |
| --- | --- | --- |
| 第 1 天，3–4 小时 | 整理宿主机、两个虚拟环境、源码与模型 revision；运行现有 preflight 和适用测试 | 环境可复现；清楚哪些测试需要 GPU；保留已有正确性结果 |
| 第 2 天，3 小时 | 阅读 `LLM.generate`、输入处理和 `LLMEngine`；运行 X01 的 generate/chat | 相同内容经模板处理前后的 token IDs 可解释 |
| 第 3 天，3–4 小时 | 准备 20 条固定输入，比较 HF、mini_llm、vLLM | 区分模板/采样/停止差异与数值差异，定位至少一个实际差异或确认一致范围 |
| 第 4 天，3 小时 | 阅读 `EngineCore.step`、请求与调度输出数据结构 | 画出一次 engine step，并标明 scheduler、executor、worker 的职责 |
| 第 5 天，3–4 小时 | 跑一次无 profiler 的离线小基准；准备在线基准配置 | 保存原始数据、实际 token 长度和显存，而不是只截图 tokens/s |
| 第 6 天，3–4 小时 | 阅读一个真实请求从 logits 到停止与输出的路径；核对 source import | 能解释 EOS、max_tokens、上下文上限和输出汇集 |
| 第 7 天，2–3 小时 | 整理调用图、问题清单和复现说明；口头演示 10 分钟 | 另一位读者可以复现首个 example；在线测量的前置条件已满足 |

基线之后逐项执行 [全部详细手册](开源推理项目实施总览.md)，每项将通用实现提交到功能目录，实测后再补 `docs/reports/`，最终按发布手册完成独立安装与复现。

## 资料入口

阅读优先级为固定版本源码与测试、对应官方文档、原始论文，再到第三方讲解。旧版 V0 的 block manager 和 engine 路径可帮助理解历史，但不要直接套用到当前 V1 调用图。

- [vLLM 官方 examples 索引](https://docs.vllm.ai/en/latest/examples/) 与 [当前版本 examples](../../vllm/examples)。
- [Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) 与 [本地 bench 实现](../../vllm/vllm/benchmarks/serve.py)，核对工作负载和计时口径。
- [架构](../../vllm/docs/design/arch_overview.md)、[APC](../../vllm/docs/design/prefix_caching.md)、[CUDA Graph](../../vllm/docs/design/cuda_graphs.md)、[Model Runner V2](../../vllm/docs/design/model_runner_v2.md)。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)，重点读显存管理与调度设计，再核对当前实现。
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)，重点理解 IO-aware attention，与 KV 分页区分。
- [DistServe 论文](https://arxiv.org/abs/2401.09670)，用于理解 goodput、SLO 与 P/D 分离的动机；不把论文硬件结果当作本机预期。
