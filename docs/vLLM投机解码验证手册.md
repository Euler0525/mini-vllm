# vLLM 投机解码验证手册

本手册从不额外加载 drafter 的 n-gram 方法开始，验证多 token 提议/验证的正确性与服务收益。对应主计划投机主题和 X10，预计 20–25 小时。前置是统一输入与 sampling、[SLO 基线](vLLM面向SLO的服务调优手册.md) 和可解释的 trace。

交付是 vLLM 实验变体、接受统计与开销分析，首个公开版本不要求 mini_llm 自行实现 rejection sampling。额外 drafter、EAGLE/MTP 根据模型兼容性与显存再扩展。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| draft/verify 与调度阅读 | 3–4 小时 | token 状态图、采样语义 |
| n-gram 最小运行与 X10 | 5–6 小时 | 参数化实例、普通生成对照 |
| 正确性与停止边界 | 3 小时 | 相同条件前缀、EOS 和预算测试 |
| 接受统计与性能矩阵 | 4–5 小时 | 接受率、接受长度、ITL 与吞吐 |
| 成本反证与报告 | 5–7 小时 | verify/proposal trace、负收益案例 |

## 阅读实际接口

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/speculative/discovery
~/Desktop/vllm/.venv/bin/python ../vllm/examples/features/speculative_decoding/spec_decode_offline.py --help \
  > artifacts/inference/speculative/discovery/example-help.txt
rg -n 'method|ngram|num_spec|prompt_lookup|model-dir' \
  ../vllm/examples/features/speculative_decoding/spec_decode_offline.py
rg -n 'prompt_lookup|num_speculative_tokens|ngram' ../vllm/vllm/config/speculative.py
rg --files ../vllm/vllm/v1/spec_decode ../vllm/vllm/v1/sample | rg 'ngram|reject|metric'
```

上游示例默认可能选择 EAGLE，不直接运行默认大模型。先用当前 target Qwen 与 n-gram，记录 proposal 来源、target verification、拒绝后的修正 token、额外 bonus token 和 KV 提交/回滚边界。

画三种路径：全部接受、部分接受、全部拒绝。明确接受的 draft token 与最终输出 token 不同；每步可能还输出 target token。scheduler 的 token budget、已计算数与最终提交进度需要随验证结果更新。

## 跑通最小 n-gram 服务

以下是固定快照配置字段的起始组合，启动时核对实际兼容性与日志。先停止自己的其他模型服务。

```bash
cd ~/Desktop/mini-vllm
../vllm/.venv/bin/vllm serve --config configs/inference/vllm-single-gpu.yaml \
  --speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  2>&1 | tee artifacts/inference/speculative/discovery/server.log
```

若不支持该 backend/eager 组合，保存具体错误，按固定版本验证逻辑调整双方共同配置。不能只为投机臂开启 graph，却把普通臂留在 eager 后归因于投机。

在 `examples/vllm/speculative_decoding/` 写待开发参数化离线实例，支持 target、method、spec token 数、输入文件、输出、seed、EOS。普通臂不传 speculative config，其他引擎参数一致。结果输出 proposed、accepted、验证步数、真实生成 token 数和缺失字段说明。

## 准备两类负载

待开发生成器构造固定文本并保存 IDs 与 hash。先固定输入 256/1024、输出 128，确保总长度合法；真实语义负载另测正常 EOS。

| 负载 | 内容 | 目的 |
| --- | --- | --- |
| 重复型 | prompt 中含可复用的短模式、重复段落或代码续写 | 增加 n-gram 提议机会，验证机制 |
| 非重复型 | 不同主题短问答、低重复续写 | 检验提议不足或接受低的反例 |
| 短输出 | 只请求 8/16 token | 检验额外准备成本能否摊薄 |
| 真实交互 | 固定任务集，允许 EOS | 验证实际输出长度与质量 |

重复型不保证高接受，先实测 prompt lookup 与 target 行为。随机 token 性能负载无法代表 n-gram 的真实机会，不能只用它得出通用结论。

## 正确性与停止边界

先 greedy、相同 IDs、target 权重、dtype、EOS 和输出上限，比较普通/投机 token IDs。发现差异后在相同条件前缀下做单步 target 对照，并记录 backend/计算顺序与候选间距；不能直接将任何分歧都认作算法错误或一律归为浮点噪声。

| 场景 | 验证点 |
| --- | --- |
| 提议为空 | 能退回正常 target 生成并完成 |
| 全部/部分拒绝 | 已拒绝的 KV 不污染后续输出 |
| 输出上限小于 spec 数 | 不超出预算，不把未提交 token 返回客户端 |
| EOS 出现在候选内部 | 不提交 EOS 后的内容，finish reason 正确 |
| 上下文接近上限 | proposal/verify 预算与实际位置不越界 |
| 取消或容量不足 | 请求最终结束、资源恢复、下一请求正常 |

若真实模型不稳定触发某些边界，先阅读或运行固定版本已有确定性单测，再为自己的 wrapper 补输入与协议测试。不要为收集所有分支而修改 target 权重。

随机采样下理解 rejection sampling 的目标分布保持条件；相同 seed 不保证普通与投机逐字相同。需要任务质量或分布性检查时另设实验，不凭两条随机文本下结论，也不把 greedy 验证外推到全部采样配置。

## 执行性能矩阵

先普通生成与 spec=2、并发 1 对照，确认结果和统计；再扫描 spec=1/2/4、并发=1/4/8。每类负载只选代表长度，不做全组合。每臂三次重复，普通/投机交替运行，保存相同到达序列。

使用在线手册与回放器的测量方法：吞吐、goodput、TTFT、TPOT/ITL、E2E、失败率与显存。verify 一次可能返回多个 token，流式片段到达间隔不能直接当成逐 token ITL；说明客户端数据如何形成指标。

```bash
cd ~/Desktop/mini-vllm
rg -n 'name=|draft|accept|iteration' ../vllm/vllm/v1/spec_decode/metrics.py
curl --fail-with-body http://127.0.0.1:8000/metrics \
  > artifacts/inference/speculative/discovery/metrics.txt
rg 'spec|draft|accept' artifacts/inference/speculative/discovery/metrics.txt
```

| 指标 | 定义与解释 |
| --- | --- |
| 提议 token 数 | 本窗口实际交给 target 验证的 draft 数，不含没有提出的候选 |
| 接受 token 数 | 被接受的 draft 数，不能直接等同全部输出 |
| 接受率 | accepted/proposed；proposed=0 时记为不适用 |
| 每步接受长度 | accepted/验证步数，说明 bonus token 是否另外统计 |
| 每步输出长度 | 最终输出 token/验证步数，与接受长度分开 |
| 总体收益 | 完成相同成功 token 工作量的时间及 SLO 内 goodput |

从 `/metrics` 读取计数器增量，先核对名称和单位。服务重启后清零；不要把 baseline 和投机臂的累计统计相减。统计不可获取时标缺失，不从吞吐反推接受率。

## 用开销解释反例

选一组收益、一组无收益或负收益配置，用短 trace 分别看 proposal、target verify、采样、KV 更新与 CPU 调度成本。高接受率也可能被更大的 verify batch、CPU proposer 或更高并发下的资源竞争抵消。

若需隔离 graph 与投机的交互，第二轮在双方都已验证的 graph 配置下重复；将它标为交互实验。额外 drafter 必须包含权重加载、词表/结构兼容性、额外 KV 和显存成本；8GB 下不默认可行。

## 交付与验收

报告为 `docs/reports/speculative-decoding.md`，实例卡记录 n-gram 和额外方法各自的运行状态。

- [ ] 普通与投机臂的 target、输入、sampling 和共同执行配置一致。
- [ ] greedy、输出上限、EOS、上下文与取消边界有验证依据。
- [ ] 重复与非重复负载均测量，至少三次重复。
- [ ] 接受率、接受长度、每步输出与总体吞吐的分母清楚。
- [ ] 至少一个负收益案例由提议/验证成本证据解释。
- [ ] 未实现自有投机或未测试额外 drafter 的能力如实标注。

下一步进入 [KV 交接与扩展手册](vLLMKV交接与并行扩展手册.md)。
