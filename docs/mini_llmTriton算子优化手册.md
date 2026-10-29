# mini_llm Triton 算子优化手册

本手册交付一个有 oracle、支持范围、显式回退和端到端数据的 Triton 算子。优先从 RMSNorm 开始；若 [性能分析](vLLM执行链追踪与性能分析手册.md) 显示它占比极低，可以改选 KV 写入等可独立验证的热点。对应主计划算子主题，包含 X07 的协议实例，预计 20–25 小时。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 热点选择与误差契约 | 2–3 小时 | 选择依据、形状清单、oracle |
| Triton 参考实现 | 4–5 小时 | 一个算子、输入检查与回退 |
| 正确性和边界 | 3–4 小时 | FP32/BF16、尾部与布局验证 |
| 微基准和端到端 | 4–5 小时 | 时间分布、集成开关、Amdahl 分析 |
| X07 结构化输出与工具协议 | 5–6 小时 | 一个最小协议实例与失败样本 |
| 报告与回归 | 2 小时 | 支持矩阵、负收益范围与复现 |

## 从现有算子建立 oracle

```bash
cd ~/Desktop/mini-vllm
cat mini_llm/layers/rms_norm.py
cat tests/test_rms_norm.py
rg -n 'hidden_size|rms_norm_eps' mini_llm/config/model.py
.venv/bin/python -m pytest tests/test_rms_norm.py -q -p no:cacheprovider
.venv/bin/python - <<'PY'
import torch
import triton
print({"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda})
PY
```

先确认 mini_llm 环境已具备与当前 PyTorch 配套的 Triton。若缺失，按该环境兼容依赖增加可选 extra 并更新锁文件，不从 vLLM 环境复制包目录，也不安装任意最新编译器。

RMSNorm 计算为 `y=x×rsqrt(mean(x²)+eps)×weight`。保留当前实现作为默认 oracle；明确平方、归约、归一化与乘权重的 dtype/转换顺序。BF16 中“全部用 FP32 最后再转回”与“中间回到 BF16”可能有差异，先约定语义再写 kernel。

## 实现一个可维护的内核

| 待开发位置 | 要求 |
| --- | --- |
| `mini_llm/kernels/rms_norm.py` | Triton kernel 与 Python wrapper；按行归约、尾部 mask、dtype/shape/device 校验 |
| `mini_llm/layers/rms_norm.py` | 保留 PyTorch 路径，增加显式 backend 选择；默认策略由实测决定 |
| `benchmarks/kernels/bench_rms_norm.py` | 形状扫描、预热、计时、重复、元数据与误差表 |
| `tests/test_rms_norm.py` | 扩展有意义的数值、布局、尾部与回退行为 |

第一版每行处理一个 hidden vector，对归约范围使用正确 mask。真实 hidden size 896 不是二次幂，mask 必须同时保护 load/store 与归约。先只支持最后一维连续的二维视图；其他布局明确回退或拒绝，不能静默按连续内存解释。

wrapper 检查输入与 weight 在同一 CUDA device、最后一维匹配、输出 dtype 正确、eps 合法。空 token batch 返回约定形状；对过大 hidden size 明确支持范围。记录别名和 inplace 契约，第一版建议返回新 tensor。

不要捕获所有异常后静默改走 PyTorch，这会掩盖 kernel 错误。仅对已知的不支持布局/设备选择回退；运行时编译或执行错误保留异常。实现一个内核后再研究 fusion，不把 norm、残差与量化同时引入。

## 数值与行为验证

| 维度 | 最小测试范围 | 通过依据 |
| --- | --- | --- |
| 真实形状 | hidden=896；token rows=1/2/4/8/32/128/512 | oracle 输出、最大与平均误差 |
| 尾部 | hidden=63/65/895/897 | masked lane 不参与归约，不越界 |
| dtype | FP32、BF16；FP16 若声明支持 | 预先固定绝对/相对容差 |
| 输入分布 | 全零、普通随机、较大/较小有限幅度 | eps 与归约稳定性 |
| 布局 | 连续、步长切片、转置视图 | 支持布局正确，不支持布局按契约处理 |
| 参数 | 不同 eps、weight、空 batch | 输出形状与错误行为一致 |
| 端到端 | 固定 20 条 prompt、相同已生成前缀 | logits 误差、首个 token 分歧及停止原因 |

容差从 oracle 精度与模型数值范围确定，记录选择理由。先用 FP32/小尺寸定位错误，再跑 BF16。随机 token 序列中出现分歧时用相同条件前缀做单步比较，不能用后续整段文本差异判断误差不断扩大。

## 建立可靠微基准

预先分配输入并完成 JIT，计时窗口只包住算子。选择 CUDA events 或 Triton 基准工具，预热后做多组重复，报告中位数、分位范围与单位。若两种实现都有输出分配，保持相同契约；另测预分配输出时单列，不混合比较。

至少扫描 rows=1/8/32/128/512、hidden=896，再补周边维度解释尾部成本。交替 PyTorch/Triton 次序并记录功耗/温度。小 kernel 的 CPU launch、GPU 时间和整次调用时间分别测量；禁止用未同步的 `perf_counter()` 代表 GPU 执行时间。

有效带宽可用已声明的数据读写字节除以时间估计，但这是算法口径，不能称为硬件实测 DRAM 带宽。真实优化主张结合 Systems/Compute 证据；记录 `num_warps`、block size、编译器与 GPU 型号，避免隐藏 autotune 试探成本。

## 接入 mini_llm 并端到端回测

给运行时和 benchmark 增加显式 `torch`/`triton` backend 参数，写入结果元数据。先在单请求模型通过 oracle，再跑动态 batch 与分页路径。公平对照固定输入、输出 token 数、调度策略、KV 配置、预热和 GPU 状态；不启用 profiler 记录正式吞吐。

| 层级 | 对照 | 应交付结果 |
| --- | --- | --- |
| 单算子 | PyTorch 与 Triton | 每形状延迟、误差和支持范围 |
| 单请求 | prefill 重、decode 重两种负载 | prefill/decode/总时延与首个分歧 |
| 动态请求 | 相同到达序列与并发 | makespan、TTFT、吞吐和失败率 |

用 profile 中算子原占比 p 和局部加速比 s 估算端到端上限 `1/((1-p)+p/s)`。若局部快两倍但端到端无变化，检查占比、launch、同步与测量噪声。无收益时仍交付实验性 backend，默认继续使用可靠路径。

若计划迁移到 vLLM，必须先确认它实际执行的 norm 是否已经融合，比较的是同一语义。mini_llm 上有效的独立 RMSNorm 不能直接证明可替换 vLLM 的融合算子。

## 完成 X07 协议实例

此处按 examples 计划补充结构化输出与工具调用，复用在线客户端，独立于 kernel 性能对照。先阅读固定版本示例与 parser 要求：

```bash
cd ~/Desktop/vllm
rg --files examples/features/structured_outputs examples/tool_calling
sed -n '1,180p' examples/features/structured_outputs/README.md
```

在 `examples/vllm/structured_outputs/` 编写最小 JSON schema 请求，验证成功对象、字段类型、非法 schema 和截断输出。schema 合法不等于业务结果正确，另写语义断言。

工具例只选本地无副作用函数，例如加法；记录工具选择、参数 JSON、校验、调用和结果回填。模型提出调用与客户端真正执行分开。Qwen2.5-0.5B 的调用质量及 parser 兼容性需实测；若不支持，交付协议假响应测试和兼容性结论，记录额外模型实验的资源需求。

logits processor 或约束解码会改变生成语义和开销，不能把其结果混进 kernel 加速结论。为每个实例记录上游入口、固定 commit、变体、运行状态、失败样本和源码定位。

## 交付与验收

报告为 `docs/reports/rmsnorm-kernel.md`，若改选热点使用功能名称；X07 结论进入 examples 索引。

- [ ] 热点选择有 trace 占比依据，oracle 与误差契约明确。
- [ ] 896 和非二次幂尾部、BF16、布局与回退行为经过验证。
- [ ] 微基准预热、同步、输出分配与重复口径一致。
- [ ] 端到端开关可复现，数值与停止语义未被性能测量掩盖。
- [ ] 报告包含 Amdahl 估算、无收益范围和默认 backend 决策。
- [ ] X07 至少有一个有效协议实例或明确的兼容性阻塞证据。

下一步进入 [编译与 CUDA Graph 手册](vLLM编译与CUDAGraph验证手册.md)。
