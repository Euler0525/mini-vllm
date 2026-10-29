# vLLM 执行链追踪与性能分析手册

本手册交付实际 runner/backend 的执行链、带注释 trace 和可证伪的瓶颈分析。对应主计划执行路径与性能工具主题、X06，预计 20–25 小时。前置是固定负载、无 profiler 的 [容量基线](vLLM监控与容量评测手册.md) 以及可对照的 mini_llm 路径。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 确认 runner、backend 与模型路径 | 3–4 小时 | 启动证据、调用图、张量形状 |
| 分离 prefill/decode 窗口 | 3 小时 | 小负载与窗口校验 |
| PyTorch Profiler 与 X06 | 4–5 小时 | CPU/算子/kernel 对应 trace |
| Nsight Systems | 4–5 小时 | 多进程等待与 GPU 空隙分析 |
| 热点验证与可选 Compute | 3–4 小时 | 三项开销表、反证实验 |
| 归档与优化决策 | 3–4 小时 | 两个瓶颈分析及一个待优化目标 |

## 先回答实际执行的是哪条路径

```bash
cd ~/Desktop/vllm
rg -n '_make_model_runner|use_v2_model_runner|execute_model' vllm/v1/worker/gpu_worker.py
rg -n 'execute_model|slot_mapping|positions|attn_metadata' vllm/v1/worker/gpu_model_runner.py
rg -n 'execute_model|slot_mapping|positions' vllm/v1/worker/gpu/model_runner.py
rg -n 'class Qwen2|qkv_proj|load_weights|forward' vllm/model_executor/models/qwen2.py
rg -n 'backend|FLASH|selector' vllm/v1/attention/selector.py
```

结合启动日志记录实际 runner 类、attention backend、dtype、head_dim、KV 布局、graph/compile 状态。目录中存在两套 runner，不等于都在运行；V1 引擎与 Model Runner V2 也不是同一个开关。

张量流图至少标出 token IDs、positions、Q/K/V、slot mapping、block table、logits 和采样结果，写一轮的实际形状。对照 mini_llm 的融合程度、GQA、RoPE 和权重装载，指出哪个 tensor 属于当前 token batch，哪个容量随历史上下文增长。

## 固定小负载与采集窗口

先测无 profiler 的 128/128、1024/32、128/512 三种负载，各取并发 1 与 4 的少量代表 case。主诊断先 eager、APC 关，后续 graph 单独采集。模型加载、kernel 预热、编译和测量窗口分别标记。

X06 使用的 `run_one_batch_offline.py` 支持按 prefill/decode 选择窗口，但 prompt size 可能按字符构造。先保存 `--help`，记录真实 token 数；chunked prefill 会跨多个 engine iteration，不能只按“第 1 轮等于全部 prefill”标注 trace。

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/profiling/{torch,nsys,logs}
~/Desktop/vllm/.venv/bin/python ../vllm/examples/features/profiling/run_one_batch_offline.py --help \
  > artifacts/inference/profiling/logs/offline-help.txt
rg -n 'profiler|profile|prompt-size|prompt_size|iteration' \
  ../vllm/examples/features/profiling/run_one_batch_offline.py
```

在 `examples/vllm/profiling/` 写自己的最小变体，使用固定 token IDs 和显式采集起止。mini_llm 增加可关闭的 `record_function` 或 NVTX 区间，标记 prefill/decode、batch packing、KV gather/scatter、forward、sampling。正式吞吐运行关闭额外诊断日志。

## 采集 PyTorch Profiler

先停止自己的普通服务，以下命令启动诊断专用实例。

```bash
cd ~/Desktop/mini-vllm
../vllm/.venv/bin/vllm serve --config configs/inference/vllm-single-gpu.yaml \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"artifacts/inference/profiling/torch"}' \
  2>&1 | tee artifacts/inference/profiling/logs/torch-server.log
```

在另一终端先用在线手册的普通 bench 完成预热，再执行短采集。`--num-warmups 0` 避免把 bench 自身预热混入关注窗口。

```bash
cd ~/Desktop/mini-vllm
../vllm/.venv/bin/vllm bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --model qwen2.5-0.5b --tokenizer artifacts/models/qwen2.5-0.5b \
  --dataset-name random --input-len 128 --output-len 32 --random-range-ratio 0 \
  --num-prompts 4 --num-warmups 0 --request-rate inf --max-concurrency 1 \
  --temperature 0 --ignore-eos --seed 42 --profile \
  --save-result --save-detailed --result-dir artifacts/inference/profiling \
  --result-filename torch-window.json
```

正常停止服务让 trace 落盘，确认 trace 中包含 GPU worker。用 Perfetto 或兼容查看器标出首个 prefill、连续 decode、CPU batch 准备、kernel launch 与输出处理。开启 stack/shape/memory 会增加开销，仅在需要解释调用点时另开短窗口。

热点表同时给 self CPU、CUDA 时间、调用次数和对应源码。异步/重叠事件的总和可能超过墙钟时间；不能把各行 CUDA 总时间直接相加当端到端耗时。GPU 空隙需沿依赖链解释，不能仅看 API server 的 CPU 事件。

## 采集 Nsight Systems

先保存 `nsys --version` 和 `nsys profile --help`，确认本机支持相关开关。下列沿用固定快照的 CUDA profiler API 控制方式，限制到稳态窗口。

```bash
cd ~/Desktop/mini-vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
  --trace=cuda,nvtx,osrt --sample=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output=artifacts/inference/profiling/nsys/eager \
  ../vllm/.venv/bin/vllm serve --config configs/inference/vllm-single-gpu.yaml \
  --profiler-config '{"profiler":"cuda"}'
```

另一终端先普通预热，再复用上一段短 bench，加 `--profile`，换新的结果名。正常结束本次服务后，列出生成的 `.nsys-rep`，对实际文件调用 `nsys stats`。同时保存命令、PID/rank、捕获开始结束与相关请求窗口。

判断 GPU 是否等待 CPU 准备、IPC、同步、内存拷贝，或客户端供给不足。观察到间隙后，改变一个变量验证，例如增加在途请求、缩短 Python packing 或减少输出转换；不要看到 CPU 忙就直接把问题归为调度器。

## 选择热点与反证实验

| 观察 | 候选解释 | 可以推翻解释的实验 |
| --- | --- | --- |
| 小 batch decode 存在大量短 kernel 与间隙 | launch/CPU 开销占比较高 | 同形状 graph 对照后间隙是否减少，端到端是否改变 |
| 长上下文 decode attention 变长 | 读取 KV 的工作量增长 | 固定 batch 扫历史长度，核对 kernel 与硬件流量 |
| prefill 大矩阵占据主要窗口 | 投影/MLP 计算占比高 | 固定输出扫输入长度，比较 GEMM 时间和 SM 指标 |
| 分页参考版额外 memcpy/分配 | gather 与临时张量成本 | 预分配/重用缓冲后耗时是否下降 |
| GPU 空闲且队列为空 | 客户端未提供足够工作 | 提高实际发送率并观察 engine queue |

至少完成两个“现象 → 假设 → 单变量实验 → 结论”闭环。瓶颈可因负载不同而变化，不能用 prefill 的热点决定所有 decode 优化。

Nsight Compute 只对 Systems 找到的一个热点采少量 launch。先确认 `ncu --help`、kernel 名、形状和权限，选择实际需要的 DRAM、SM、occupancy、stall section。记录 profiler replay 与缓存行为，不能把被重放的采集时间当服务延迟。

没有计数器权限时记录限制，继续用时间线和形状变化做因果验证；没有实测 DRAM 字节就不声称带宽已饱和。有效带宽与理论字节估算分开标注，roofline 需要明确运算量与数据量口径。

## 交付与验收

报告为 `docs/reports/inference-profiling.md`，保留三张标注后的 prefill/decode/mixed 时间线、前三项开销、两个瓶颈闭环和下一项优化选择。大 trace 存在 `artifacts/`，文档附精简图、完整参数与来源索引。

- [ ] 实际 runner/backend 有日志与源码证据，张量形状可解释。
- [ ] 预热、prefill、decode 与混合窗口没有凭轮次猜测。
- [ ] PyTorch 与 Systems 捕获到真实 GPU worker。
- [ ] 诊断性能与无 profiler 的正式基准分开。
- [ ] 至少两个假设经过反证实验；缺失的硬件计数器明确记录。
- [ ] 选择一个可独立实现且有 oracle 的热点进入 [Triton 手册](mini_llmTriton算子优化手册.md)。
