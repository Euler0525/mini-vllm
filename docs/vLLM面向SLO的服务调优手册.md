# vLLM 面向 SLO 的服务调优手册

本手册交付有依据的单卡服务配置、吞吐–尾延迟 Pareto 图和未参与调参负载的验证结果。对应主计划调优主题及 X09，预计 20–25 小时。前置是 [容量曲线](vLLM监控与容量评测手册.md)、KV 实验与 [编译/graph 对照](vLLM编译与CUDAGraph验证手册.md) 已完成。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 固定 SLO、质量与负载划分 | 2–3 小时 | 调参集、保留集、失败门槛 |
| 单变量筛选 | 4–5 小时 | token budget、序列上限、KV 预算曲线 |
| 小范围交互实验 | 3–4 小时 | 少量候选配置和 Pareto 表 |
| 保留负载与退化测试 | 3–4 小时 | 泛化、长请求、取消和过载 |
| X09 任务边界及可选量化 | 5–6 小时 | embedding/rerank 最小卡片，质量与资源边界 |
| 最终配置和报告 | 3 小时 | 可复用配置、选择理由、反例 |

## 先写判断条件

定义练习 SLO，例如 TTFT ≤1000 ms、TPOT ≤80 ms；若改变业务目标，作为新的实验版本。预先约定失败率、输出质量、显存、启动成本和过载条件，配置选择以这些条件内的 goodput 为主。不能测完再调宽 SLO 让配置通过。

| 数据部分 | 用途 | 控制 |
| --- | --- | --- |
| 调参集 | 128/128、1024/64、128/512 与固定混合到达 | 所有候选使用相同 IDs、seed、到达序列 |
| 保留集 | 不同 prompt 内容、长度比例或到达种子 | 选择配置前生成并封存 hash |
| 正确性集 | 已有 20 条固定输入及停止/取消边界 | 每次执行路径变更后回归 |
| 质量集 | 具有明确答案或任务断言的真实请求 | 与吞吐随机 token 负载分开 |

输入/输出总预算始终不超过服务上下文上限。质量集不能只检查输出非空；可用固定问答、结构化字段、代码小题或人工评分，但需固定判定规则与样本。量化或 adapter 改变权重时记录任务指标和退化样本。

## 固定基线并准备扫参工具

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/tuning/baseline
cp configs/inference/vllm-single-gpu.yaml artifacts/inference/tuning/baseline/server.yaml
git rev-parse HEAD > artifacts/inference/tuning/baseline/commit.txt
sha256sum artifacts/inference/inputs/prompts.jsonl \
  > artifacts/inference/tuning/baseline/input.sha256
../vllm/.venv/bin/vllm serve --help \
  > artifacts/inference/tuning/baseline/serve-help.txt
```

待开发 `benchmarks/inference/run_matrix.py` 消费描述实验臂的配置文件，顺序启动自己管理的服务、检查 ready、运行负载、采集结果并退出。支持计划预览、失败继续/停止策略与断点续跑；每臂独立目录、配置副本与退出码，不覆盖旧结果。

先人工跑通一个候选和其 baseline，再自动化。服务启动失败不应进入性能排名；更改参数触发的 backend/graph 降级必须进入结果。CPU 客户端供给不足时修正测量，再讨论服务器配置。

## 分阶段筛选参数

| 轮次 | 只改变的因素 | 起始候选 | 需要作出的决定 |
| --- | --- | --- | --- |
| token budget | 每轮可调度 token 数 | 256/512/1024/2048 | 长 prefill TTFT 与短 decode ITL 的权衡 |
| 序列上限 | `max_num_seqs` | 4/8/16 | 吞吐、排队和 KV 压力的平衡 |
| KV 容量 | 可用显存比例或显式 KV 预算 | 比例 0.60/0.70/0.80；显式预算另设实验 | 抢占/重算是否限制 goodput |
| 执行方式 | 已验证的 eager/compile/graph | 选择两种代表配置 | launch 收益能否抵消 capture 成本 |
| APC | 仅用于共享前缀负载 | 开/关 | 是否与目标输入一致，驱逐影响如何 |

先在固定并发与到达率下单变量初筛，每项保留 1–2 个候选，再选 4–6 组组合验证。不要对所有长度、并发、budget、显存和 graph 做完整笛卡尔积。

chunked prefill 保持开启时，小 budget 可以让长 prompt 分轮运行；关闭 chunking 的对照先核对该版本 budget/模型长度校验。显存比例与 `kv-cache-memory-bytes` 不是可互换的两个刻度，显式设置 KV 后记录其优先关系和实际块数。

客户端并发是负载条件，`max_num_seqs` 是引擎配置。提高后者不会自动产生更多请求；同时改变两者不能解释是哪一项带来收益。

## 判定候选配置

对每个候选至少三次独立重复，交替 baseline 与候选。输出每次吞吐、goodput、P95 TTFT/TPOT、失败率、抢占增量、KV 峰值、整卡峰值和 GPU 条件。

Pareto 图横轴为尾延迟，纵轴为 goodput 或输出吞吐，标注 SLO 线与配置。一个配置只有在不劣于其他关键约束且至少一项更好时才可能进入前沿。不要把统计噪声内的小变化写成确定收益。

| 现象 | 下一步 |
| --- | --- |
| 吞吐升高但 goodput 下降 | 检查短请求 TTFT、长队列和尾部等待 |
| KV 利用率高且抢占多 | 查看总活跃 token 与重算，调整并发/长度/容量 |
| 小 budget 改善短请求但总耗时增大 | 报告公平性与吞吐权衡，按 SLO 选择 |
| graph 配置丢失 KV 容量 | 隔离 capture 预算，保留相同 KV 字节对照 |
| 热前缀场景效果很好，保留集退化 | 说明输入分布依赖，不选成通用默认值 |

## 保留负载、质量和故障验证

选定配置后锁定参数，再运行保留集。比较 baseline 与最终配置在不同长度比例、不同到达 seed、突发流量、正常 EOS 和取消下的结果。若重新调参，原保留集成为调参集，需要另一组独立验证。

加入一个受控退化 case，例如输入更长、到达率超容量或 KV 预算更小。观察失败/拒绝/重算、队列是否可排空、恢复后能否服务。性能报告保留失败分母，不以丢弃慢请求提升吞吐。

最终配置待创建为 `configs/inference/vllm-interactive.yaml` 或其他用途名称，文件注释写明硬件、模型与适用负载。配置旁给出启动命令、SLO、已测范围和已知退化，避免使用“最佳配置”作为无条件标签。

## X09 任务模型与可选量化

先阅读固定快照的 embedding 与 rerank 入口，选择一个资源允许的 pooling 模型，固定其完整 revision、任务类型和输出语义。

```bash
cd ~/Desktop/vllm
rg --files examples/pooling/embed examples/pooling/score examples/generate | head -n 40
sed -n '1,150p' examples/pooling/embed/openai_embedding_client.py
```

在 `examples/vllm/pooling/` 做“查询/文档 → embedding → 相似度”或“query/doc → score → 排序”的最小实例。验证向量维度、批量顺序、空输入与长度限制；rerank 验证返回索引与原文档对应。embedding/rerank 吞吐按任务定义，不混入生成 tokens/s。

多模态先写输入模态、processor、视觉/音频 token、上下文和显存表。只有相应模型可在本机运行时才做小样本，不把文本 Qwen 当视觉/音频模型。没有适用资源则标为仅阅读，保留独立运行步骤。

量化为可选实验，先确认当前 Ada GPU、模型和实际 attention/linear backend 的支持条件。权重量化与 KV 量化分别测量：模型或 KV dtype、比例因子来源、权重/Cache/整卡显存、真实任务质量、TTFT/ITL 与吞吐。量化模型是新实验身份，不能继续标为原始 BF16 权重 revision。

若无法找到有清晰来源与质量验证的兼容量化资产，保留 BF16 主线。8GB 下节省容量并不自动代表更低时延；负收益同样写入报告。

## 交付与验收

报告为 `docs/reports/slo-tuning.md`，附最终配置、Pareto 图、保留集 hash 和原始数据索引。

- [ ] 调参前固定 SLO、质量门槛与负载划分。
- [ ] 单变量筛选后只做小规模组合，完整保存生效参数。
- [ ] 最终配置有至少三次重复和 baseline 对照。
- [ ] 保留集、正常 EOS、取消及受控过载均完成。
- [ ] 至少一个退化场景和质量/失败率结果公开。
- [ ] X09 的任务输出语义与运行状态齐全；可选项未被误写为已支持。

下一步按 [投机解码手册](vLLM投机解码验证手册.md) 测试另一类执行机制。
