# vLLM 监控与容量评测手册

本手册交付服务容量扫描、可观测指标和吞吐–尾延迟报告，回答增加流量后等待发生在哪里、何时超过交互目标。对应主计划的容量主题及 examples X04，预计 20–25 小时。前置是 [在线服务测量](vLLM在线服务与流式测量手册.md) 可复现；mini_llm 动态调度完成后也接入同一评测契约。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 指标发现与埋点核对 | 3 小时 | 指标字典与采集样本 |
| 监控配置与时间对齐 | 3–4 小时 | Prometheus 配置、仪表板或离线图 |
| 固定并发扫描 | 4–5 小时 | 并发–吞吐–P95 曲线 |
| 有限到达率与混合长度 | 4–5 小时 | 实际发送率、队列与长度分组延迟 |
| 开发汇总和图表工具 | 3–4 小时 | schema 校验、汇总表、可复现图 |
| 报告与项目演示 | 3–4 小时 | 容量结论、原始数据索引与演示 |

## 固定实验条件

沿用 [单卡服务配置](../configs/inference/vllm-single-gpu.yaml)，先保持 eager、APC 关闭、上下文 2048、token budget 1024、最大序列数 8。输入/输出先用 128/128，客户端并发从 1/2/4/8 开始。笔记本插电并固定性能模式，一次只启动一个模型实例。

TTFT ≤1000 ms、TPOT ≤80 ms 可作为练习 SLO，不代表行业标准。SLO 在扫参前固定，报告注明适用负载。吞吐定义为整个测量窗口内成功输出 token 数除以秒数；goodput 为同时满足所有既定 SLO 的成功请求数除以秒数。

## 发现实际暴露的指标

先按在线手册启动服务，完成预热，再保存指标；指标可能仅在请求后出现。

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/capacity/discovery
curl --fail-with-body http://127.0.0.1:8000/metrics \
  > artifacts/inference/capacity/discovery/metrics.txt
rg 'vllm:.*(running|waiting|cache|preempt|queue|first_token|inter_token)' \
  artifacts/inference/capacity/discovery/metrics.txt
rg -n 'name="vllm:|\.observe\(|\.inc\(' ../vllm/vllm/v1/metrics/loggers.py
```

| 指标族 | 用来回答的问题 | 解释边界 |
| --- | --- | --- |
| running/waiting 请求数 | 在执行还是排队？ | 瞬时 gauge，不能直接得到等待分布 |
| `vllm:kv_cache_usage_perc` | KV 池当前占用多少？ | 固定版本中值为比例，显示百分数需乘 100；不同于整卡显存 |
| prefix queries/hits | 查询和命中了多少 token？ | 先核对计数单位，使用窗口增量 |
| preemptions | 是否发生抢占与重算？ | 可恢复抢占与进程 OOM 分开 |
| request queue/prefill/decode 时间 | 请求时间分布在哪个阶段？ | 从埋点确认边界，不能随意相加成客户端 E2E |
| TTFT/ITL histogram | 服务端观察的延迟分布 | 与客户端观测点及聚合权重不同 |

为每项登记完整名称、类型、单位、label、更新位置、是否重启清零。不要从其他版本教程复制旧指标名。高频采样仅用于诊断，正式运行记录采样开销。

## 建立监控与时间对齐

待添加 `configs/inference/prometheus-local.yaml`，最小内容如下。它是需要落盘的配置模板，不是当前已有配置。

```yaml
global:
  scrape_interval: 1s
scrape_configs:
  - job_name: vllm_local
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

上述地址适合 Prometheus 运行在宿主机；容器模式需要根据网络调整地址。先能采到数据再做 Grafana 展示，仪表板 JSON 存入 `configs/inference/`。没有完整监控栈时，开发采样器定时抓取 `/metrics`，保存 UTC、单调时钟与原文，再离线绘图。

待开发 `benchmarks/inference/collect_metrics.py` 支持 `--url`、`--interval`、`--output`、运行窗口和正常退出；抓取失败保留时间与错误。使用相同窗口采集 GPU 条件：

```bash
cd ~/Desktop/mini-vllm
nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm \
  --format=csv --loop=1 \
  > artifacts/inference/capacity/discovery/gpu.csv
```

采集前记录空闲状态，结束时在该终端 Ctrl-C。若某字段不支持，删去字段并记入环境记录。`GPU-Util` 只能作为粗粒度活动指标，不能证明带宽饱和。

在 Prometheus 中先尝试以下查询，实际 label 过滤从样本中确认。使用 2 分钟窗口要求测试与采样覆盖足够时间，短测试直接分析原始请求数据。

```promql
histogram_quantile(0.95,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[2m])))
```

百分位不能跨运行直接求平均冒充合并分布；应从原始请求重算，或分别展示各次 P95 与范围。低样本直方图会受桶边界影响，客户端原始结果是容量报告的主要依据。

## 执行固定并发扫描

下列命令使用现有 vLLM bench。服务启动、预热和模型加载在测量窗口外；每个 case 单独 warmup，正式数据不启用 profiler。

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/capacity/concurrency
for run in 1 2 3; do
  for concurrency in 1 2 4 8; do
    ../vllm/.venv/bin/vllm bench serve \
      --backend vllm --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
      --model qwen2.5-0.5b --tokenizer artifacts/models/qwen2.5-0.5b \
      --dataset-name random --input-len 128 --output-len 128 --random-range-ratio 0 \
      --num-prompts 200 --num-warmups 10 --request-rate inf --max-concurrency "$concurrency" \
      --temperature 0 --ignore-eos --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --goodput ttft:1000 tpot:80 --save-result --save-detailed \
      --result-dir artifacts/inference/capacity/concurrency \
      --result-filename "c$concurrency-run$run.json"
  done
done
```

这是初始探索顺序。确认候选点后交替高低并发或反向复测，核对温度、时钟和漂移。200 条适合初筛，P99 必须连同样本数与跨次稳定性展示，最终候选增加样本到能覆盖目标分位与稳定区间。

记录每次完成数、失败数、实际输出长度、总窗口、吞吐和 P50/P95/P99。客户端并发超过 `max_num_seqs` 时服务器排队是预期现象，两项参数分别标注。

## 执行有限到达率与混合长度实验

先根据扫描结果估计成功请求吞吐，选其约 25%、50%、75%、100%、125% 的有限到达率做实验，不预设这台 GPU 的 QPS。复用命令时修改 `--request-rate`，客户端并发上限提高到足以提供负载且不压垮客户端的值，输出到独立目录。

待开发回放器保存计划到达、实际 dispatch、客户端等待、服务完成与未完成请求。若并发限制导致实际发送率低于目标，报告客户端限流，不能宣称服务承受了目标流量。发生持续队列增长时，运行到固定窗口并统计排空时间，不能仅截取已经完成的快请求。

| 负载 | 组成 | 需要观察 |
| --- | --- | --- |
| 平衡 | 128 输入 / 128 输出 | 吞吐与交互延迟的初始权衡 |
| prefill 较重 | 1024 输入 / 64 输出 | 排队和长 prompt 首 token |
| decode 较重 | 128 输入 / 512 输出 | 持续 decode、活跃 KV 与 ITL |
| 混合 | 80% 短输入、20% 1024 token 输入，固定到达顺序 | 短请求尾延迟是否被长 prefill 放大 |
| 正常停止 | 真实 prompt，允许 EOS | 自然输出长度下的失败率与体验 |

先分别测前三种，再实现混合回放，避免对长度和到达率做全组合。固定 token 预算满足总上下文不超过 2048；真实 prompt 用 tokenizer 记录实际长度，不能按字符计数。

## 汇总与容量判断

| 待开发工具 | 必须提供的能力 |
| --- | --- |
| `summarize_serving.py` | 读取详细 JSON；校验模型/输入/配置；保留失败；输出跨次表与数据质量提示 |
| `plot_serving.py` | 生成吞吐–P95、到达率–队列、长度分组延迟和 goodput 图，写出绘图输入 |
| `tests/test_serving_summary.py` | 覆盖空结果、失败、单 token 输出、条件不一致与百分位计算 |

图表使用可导出的 SVG/PNG，标注单位、模型、硬件、配置和样本数。选择“最大 SLO 内 goodput”的运行区间，再解释吞吐继续增加时 P95 是否恶化。容量上限应带条件，例如固定长度、SLO、到达模型和失败率；不能脱离工作负载给一个通用并发数。

可用 Little 定律作稳态核验：平均在途请求数约等于完成率乘平均响应时间。过载增长、窗口未排空或统计范围不一致时，不用它强行推导性能。

## 交付与验收

报告为 `docs/reports/serving-capacity.md`。完成第一次项目演示：一次流式请求、一组动态到达、监控变化和一条容量曲线，用源码解释等待来源。

- [ ] 指标字典、采集配置、时间对齐与 GPU 条件齐全。
- [ ] 固定并发和有限到达率分别完成，实际发送率可核对。
- [ ] 混合长度按请求类型分组，未完成和失败没有被删除。
- [ ] 至少三次重复，图表保留离散度与样本数。
- [ ] 给出容量区间、SLO、饱和证据和一个超载反例。
- [ ] 从配置与原始数据可以重新生成报告图表。

下一步按 [分页 KV 手册](mini_llm分页KV与前缀缓存手册.md) 检验容量与缓存行为。
