# vLLM 编译与 CUDA Graph 验证手册

本手册分离模型启动、编译/捕获和稳态执行，验证减少 launch 开销是否改善真实服务。对应主计划的编译与图执行主题，包含 X08 LoRA 配置交互，预计 20–25 小时。前置是 [profile 报告](vLLM执行链追踪与性能分析手册.md) 和无 profiler 基线，不能把首次请求同时当作冷启动与稳态性能。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 配置与 dispatch 阅读 | 3 小时 | compilation/graph/shape 对照 |
| eager 与默认优化 | 3–4 小时 | 冷启动、热启动、稳态对照 |
| 隔离编译与 graph | 4–5 小时 | 支持范围内的最小消融 |
| trace、显存与边界 | 3–4 小时 | replay 证据、fallback 与内存增量 |
| X08 adapter 交互 | 5–6 小时 | 兼容性表、可运行的小例子或限制记录 |
| 报告与第二次演示 | 2–3 小时 | 收益、代价、适用形状 |

## 先核对固定版本语义

```bash
cd ~/Desktop/vllm
rg -n 'class CompilationMode|class CUDAGraphMode|cudagraph_mode|capture_sizes' vllm/config/compilation.py
rg -n 'BatchDescriptor|dispatch|initialize_cudagraph_keys' vllm/v1/cudagraph_dispatcher.py
rg -n 'enforce_eager|cudagraph_mode' vllm/config/model.py vllm/config/vllm.py
sed -n '1,110p' docs/design/cuda_graphs.md
```

编译与 CUDA Graph 是不同机制。固定快照中 compilation mode 0 表示不编译，3 表示 vLLM compile；graph mode 包含 NONE、FULL_DECODE_ONLY、FULL_AND_PIECEWISE 等，实际支持由 backend 与运行时配置决定。`enforce_eager` 会影响整套优化，单独比较 eager 与默认只能得到组合收益。

记录启动日志中最终生效配置、实际 capture sizes、attention backend 与任何自动降级。不能仅依据启动参数宣称某种 graph 已被执行。

## 准备独立配置

以下均为待开发配置，先从现有 YAML 复制公共参数，再逐项修改。不同实验臂不同时运行。

| 配置路径 | 目标差异 |
| --- | --- |
| `configs/inference/vllm-eager.yaml` | `enforce-eager: true`，作为已有基线 |
| `configs/inference/vllm-optimized.yaml` | `enforce-eager: false`，其余保持一致，观察默认优化 |
| `configs/inference/vllm-compile-only.yaml` | 非 eager、compile mode 3、graph NONE |
| `configs/inference/vllm-compile-graph.yaml` | 相同 compile 配置、受支持的 graph mode |

编译但关闭 graph 的配置片段如下，在固定版本校验后写入新配置。

```yaml
enforce-eager: false
compilation-config:
  mode: 3
  cudagraph_mode: NONE
```

使用相同 mode，再将 graph 改为受支持的 `FULL_DECODE_ONLY`，可检验同一编译路径下 decode replay 的影响。若无法独立禁用编译但保留所需 graph，记录消融边界，不强行构造无效组合。保留 full/piecewise 对 prefill、mixed batch 的差异说明。

## 分开测量启动与稳态

待开发 `benchmarks/inference/measure_startup.py` 启动配置指定的子进程，按单调时钟记录启动、HTTP ready、首次成功请求、退出与失败。轮询 ready 的间隔和误差要写入结果。正常结束由工具管理的进程组，不扫描并终止其他进程。

| 测量 | 定义 | 必须固定 |
| --- | --- | --- |
| 进程到 ready | 启动进程至健康检查通过 | 缓存状态、轮询间隔、后台负载 |
| ready 到首次输出 | 服务就绪后的第一个真实请求 | prompt、sampling、输出长度 |
| 冷编译启动 | 使用新的专用编译缓存目录 | 不把 OS 文件缓存冷/热混为一谈 |
| 热编译启动 | 同配置与专用缓存复用后再启动 | 缓存身份、源码、GPU/依赖 |
| 稳态 | 所需形状已预热后的正式请求窗口 | 到达时序、实际 graph dispatch |

使用新的 `artifacts/inference/graph/<run_id>/`，不要清空用户原有编译缓存。固定版本支持 `VLLM_CACHE_ROOT` 时，可用单命令作用域指向专用目录；先从源码确认，不依赖跨终端导出的变量。

```bash
cd ~/Desktop/mini-vllm
rg -n 'VLLM_CACHE_ROOT|VLLM_COMPILE_CACHE' ../vllm/vllm/envs.py
mkdir -p artifacts/inference/graph
../vllm/.venv/bin/vllm serve --help \
  > artifacts/inference/graph/serve-help.txt
```

配置创建并校验后，逐个启动各臂，沿用在线手册的 bench 做预热和正式运行。每个配置至少三次独立进程，交替 A/B 顺序；先 128/128、并发 1/4，再补 1024/32、混合到达。不要一开始同时扫描全部 capture sizes。

## 验证 replay、fallback 与内存

复用 profiling 手册采极短窗口，观察 replay 调用、kernel launch 数与间隙，标出 eager/graph 区间。采集数据用于解释机制，正式性能使用未开启 profiler 的同配置。

| 场景 | 应检查的行为 |
| --- | --- |
| 命中已捕获形状 | replay 证据、CPU launch 时间与稳态 ITL |
| batch 大小变化 | padding 到 bucket、dispatch key 或 eager fallback |
| prefill/decode 混合 | 是否符合 graph mode 与 backend 支持 |
| 取消后 batch 压缩 | 元数据更新、输出对应关系与 Cache 正确性 |
| 长度或形状越界 | 有效 fallback 或明确拒绝，不能复用错误 graph |
| 显存不足 | 记录 capture 阶段失败与实际配置，逐步缩小捕获范围 |

graph 可能预留静态 buffer 与内存池。比较整卡峰值、实际 KV 容量和启动日志；同一显存比例下 KV 块数可能因 graph 开销不同而变化。先报告真实服务配置的总体变化；需要隔离 launch 影响时再固定可启动的 KV 字节预算。

性能报告包含 TTFT、TPOT/ITL、吞吐、启动时间、首次请求与显存。对于短生命周期作业，用累计完成相同请求数的总耗时估算启动成本何时被摊薄；不要只给稳态百分比。

mini_llm 自有 graph runner 是可选项。若实现，先限定固定 decode batch 和预分配 KV，保证内存地址稳定，输入在 capture 外原位更新，禁止 replay 中出现数据依赖的 Python 分支。动态 shape、取消与槽位复用必须有回归，不能只捕获一个 matmul 就宣称完成整引擎 graph。

## 完成 X08 LoRA 交互实例

```bash
cd ~/Desktop/vllm
sed -n '1,180p' examples/features/lora/multilora_offline.py
rg -n 'has_lora|lora|capture' vllm/v1/cudagraph_dispatcher.py vllm/config/compilation.py
```

先填 base model revision、adapter revision、训练底座、target modules、rank、dtype 和许可证表。示例的 Llama adapter 不能替换成 Qwen 小模型后继续使用。已有兼容小 adapter 时，在 `examples/vllm/lora/` 运行 base/adapter、冷/热装载、单/多 adapter 最小对照。

记录 adapter ID、是否命中正确权重、额外显存、load 时间与 graph 的 `has_lora` dispatch 差异。验证两个 adapter 请求不串线，切换后输出与各自独立运行一致。没有兼容资产时先完成配置与机制阅读卡，标明未实测，不下载不适配的大模型硬凑结果。

LoRA 结果单独报告；改变权重后的输出质量不能与无 adapter 的固定 greedy 文本作“必须相同”检查。

## 交付与验收

报告为 `docs/reports/compile-and-graph.md`。第二次演示解释一个真实开销、一个可控优化和其启动/内存代价。

- [ ] eager 与默认优化结果完整，组合收益没有全部归给 graph。
- [ ] 支持范围内完成编译/graph 消融，实际生效模式有证据。
- [ ] 冷/热编译、启动/首次请求/稳态分开测量。
- [ ] replay、shape 变化、fallback 和取消正确性经过验证。
- [ ] 显存与 KV 块数变化被记录，包含一个无收益或不支持的范围。
- [ ] X08 有兼容性表与真实运行状态。

下一步按 [SLO 调优手册](vLLM面向SLO的服务调优手册.md) 选择服务配置。
