# vLLM 在线服务与流式测量手册

本手册把离线生成推进到可测量的在线服务，交付可复用客户端、统一事件记录和取消验证。对应 [主计划](vLLM学习与求职计划.md) 的引擎执行与服务基线，以及 examples X02。前置是 [基线手册](vLLM基线验证与性能评测手册.md) 的环境、模型、20 条输入和正确性对照已完成。预计 20–25 小时，X02 计入其中。

## 实施安排

| 工作 | 时间 | 具体交付 |
| --- | --- | --- |
| 启动服务与核对协议 | 3 小时 | 启动日志、模型名、completion/chat 对照 |
| 追踪一次引擎执行 | 3 小时 | API 到 EngineCore 的时序图与字段表 |
| 开发流式客户端 | 5–6 小时 | SSE 解析、时间戳、结果 schema |
| 验证异常与取消 | 3–4 小时 | 失败记录和取消后的资源观察 |
| 建立三路径基线 | 4–5 小时 | mini_llm、离线 vLLM、HTTP 的口径表 |
| 独立复现与报告 | 2–4 小时 | 服务基线报告与后续压测入口 |

## 启动可追溯的服务

下面调用现有配置。先确认 8000 端口没有自己的旧服务；保留其他进程。使用独立终端运行，后续以 Ctrl-C 正常结束这一实例。

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/serving/baseline
../vllm/.venv/bin/vllm serve --help \
  > artifacts/inference/serving/baseline/serve-help.txt
cp configs/inference/vllm-single-gpu.yaml artifacts/inference/serving/baseline/server.yaml
../vllm/.venv/bin/vllm serve --config configs/inference/vllm-single-gpu.yaml \
  2>&1 | tee artifacts/inference/serving/baseline/server.log
```

另一终端检查就绪状态、API 暴露的模型名和初始指标。HTTP 200 与能成功完成一次生成分别验证。

```bash
cd ~/Desktop/mini-vllm
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models \
  > artifacts/inference/serving/baseline/models.json
curl --fail-with-body http://127.0.0.1:8000/metrics \
  > artifacts/inference/serving/baseline/metrics-before.txt
curl --fail-with-body http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b","prompt":"Explain KV cache briefly.","temperature":0,"max_tokens":16}' \
  > artifacts/inference/serving/baseline/completion.json
curl --fail-with-body -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"Explain KV cache briefly."}],"temperature":0,"max_tokens":16,"stream":true}' \
  > artifacts/inference/serving/baseline/chat-stream.txt
```

这组请求用于核对协议。chat 会应用模板，completion 的裸文本与其不是同一 token 输入；正式性能对照使用统一 token IDs 的 completion，或将已渲染文本交给相同 tokenizer 后核验 IDs。`curl` 文件没有精确的逐事件时间戳，不作为 TTFT 数据源。

## 追踪服务与引擎边界

```bash
cd ~/Desktop/vllm
rg -n 'generate|abort|request_id|stream' vllm/v1/engine/async_llm.py
rg -n 'create_chat_completion|generate|stream' vllm/entrypoints/openai/chat_completion/serving.py
rg -n 'def step|schedule|execute_model|sample_tokens|update_from_output' vllm/v1/engine/core.py
rg -n 'prefill|decode|timed_device_call' ../mini-vllm/mini_llm/engine/prefill_decode.py
```

选择一个 32 token 请求，记录 API server、engine core 和 GPU worker 的 PID、request ID 与一轮调度的 token 数。只在诊断运行中加日志；正式基准移除逐 token 打印。

字段表至少包含 request ID、prompt IDs、sampling、到达时间、计算进度、结束原因。说明请求在哪一层排队、谁检测断连、谁释放运行中资源。不要把异步 API 的返回时刻当成 GPU 已完成，也不要在不同进程的 CPU 计时之间构造未经校准的精确时差。

## 开发可复用流式客户端

以下是待开发契约，当前尚无这些工具。先完成 SSE 解析与事件单测，再接真实服务。

| 待开发位置 | 必须实现的行为 |
| --- | --- |
| `benchmarks/inference/stream_client.py` | 复用异步 HTTP session；支持 completion/chat、文本/token 输入、stream 开关、超时和主动取消 |
| `benchmarks/inference/serving_schema.py` | 定义请求元数据、单调时钟事件、状态、usage、结束原因和错误 |
| `tests/test_stream_client.py` | 用本地假响应验证分片 SSE、空 delta、错误、超时与连接关闭 |
| `examples/vllm/streaming/client.py` | 只展示一个最小请求，复用正式客户端的事件解析 |

客户端参数至少支持 `--base-url`、`--model`、`--inputs`、`--output`、`--concurrency`、`--timeout`、`--max-new-tokens` 和 `--cancel-after-chunks`。一次 session 服务整批请求，限制并发不能通过每个请求新建连接池实现。记录 HTTP 状态、错误正文和部分输出；失败请求不能从结果中消失。

以 `time.perf_counter_ns()` 记录同一客户端进程内的事件。用墙钟时间关联服务器日志，额外保存运行起止 UTC；不要拿墙钟计算延迟。

| 字段 | 定义 | 注意事项 |
| --- | --- | --- |
| `scheduled_ns` | 回放计划要求到达的时刻 | 后续用于识别客户端排队 |
| `dispatch_ns` | 实际开始发送 HTTP 的时刻 | 包括连接等待的边界需在实现中写明 |
| `first_content_ns` | 首个有效生成内容到达 | 角色、空 delta、usage 不计作内容 |
| `chunk_ns` | 后续每个有效片段到达 | 片段可能包含多个 token |
| `finish_ns` | 收到正常结束或失败/取消的时刻 | 分别记录成功、超时、取消与错误 |

客户端 TTFT 为 `first_content_ns - dispatch_ns`；到达视角 TTFT 为 `first_content_ns - scheduled_ns`。E2E 使用对应起点到 `finish_ns`。片段间隔叫 chunk interval；仅凭 SSE 文本分片无法恢复精确 token ITL。重分词也可能受分片边界影响，token 数优先取服务端 usage 或可靠的 token 事件。缺少 usage 时明确标为估算。

先阅读上游 completion/chat 客户端，脱离原文件实现最小版本；为非流式响应单独设计输出结构，不能从一次最终响应推算 TTFT。`stream_options.include_usage` 等扩展先以固定版本协议核对再使用。

## 异常、停止与取消验证

| 场景 | 操作 | 应检查的证据 |
| --- | --- | --- |
| 模型名错误 | 请求一个未发布名称 | 非成功 HTTP 状态、错误正文、客户端非零退出或失败计数 |
| 参数错误 | 提交负的输出上限 | API 拒绝，请求不进入正常生成结果 |
| 上下文越界 | 固定 IDs，使输入与预算之和超过 2048 | 该版本的拒绝行为、状态码与实际 token 预算 |
| 正常 EOS | 使用允许自然结束的短回答 | usage、finish reason、输出长度；不要求一定提前 EOS |
| 长度停止 | 忽略 EOS，固定输出 32 | 输出数与 length 语义一致 |
| 中途取消 | 收到若干片段后关闭连接 | abort 路径、running 回落、下一请求可成功 |
| 读取超时 | 假服务延迟响应，再用真实服务验证可恢复性 | timeout 与 cancel 分开，连接释放 |

取消前后分别保存 `/metrics`。APC 关闭时检查活跃请求与资源释放；APC 开启时已完成前缀可能留在池中，不要求显存使用降为零。客户端断连到引擎取消存在传播时间，记录等待窗口并检查最终状态，不以一个瞬时采样下结论。

## 建立在线基准

先运行单并发小负载核对 token 长度，再运行三个独立重复。下列命令调用上游已有 bench；每次重跑使用新的结果目录。

```bash
cd ~/Desktop/mini-vllm
../vllm/.venv/bin/vllm bench serve --help \
  > artifacts/inference/serving/baseline/bench-help.txt
for run in 1 2 3; do
  ../vllm/.venv/bin/vllm bench serve \
    --backend vllm --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model qwen2.5-0.5b --tokenizer artifacts/models/qwen2.5-0.5b \
    --dataset-name random --input-len 128 --output-len 32 --random-range-ratio 0 \
    --num-prompts 100 --num-warmups 5 --request-rate inf --max-concurrency 1 \
    --temperature 0 --ignore-eos --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 \
    --save-result --save-detailed \
    --result-dir artifacts/inference/serving/baseline --result-filename "http-c1-run$run.json"
done
```

随机负载用于可重复的长度控制；它与已有 `bench-input.jsonl` 未必逐 token 相同。若比较三种路径，使用待开发客户端把该文件中的相同 IDs 送到 HTTP completion，再复用现有 `run_inference.py` 顺序执行 HF/mini/vLLM。运行离线模型前先停止在线实例，避免显存竞争。

对 mini_llm 增加可关闭的 prefill/decode 计时接口，沿用 `timed_device_call` 的同步边界，记录 prefill、N−1 次后续 decode 和总墙钟时间。同步计时会干扰吞吐，因此单列诊断运行。现有 `offline_call_ms` 继续保留原含义，不改名为 TTFT。

| 对照 | 可回答的问题 | 必须固定或说明 |
| --- | --- | --- |
| mini_llm 单请求与离线 vLLM | 两种运行时完成同一 token 工作量的时间 | 权重、IDs、dtype、EOS、输出长度、依赖差异 |
| 离线 vLLM 与 HTTP 单并发 | 两种调用方式的用户可见耗时 | 同一 IDs、eager/APC、预热、进程布局 |
| HTTP 单并发与并发 4 | 服务批处理能否提高完成能力 | 真实输出量、失败率、TTFT/TPOT、测量窗口 |

HTTP 与离线差值包含前处理、协议、进程组织和调度差异，不能全部归为网络成本。TPOT 按请求的 `(E2E−TTFT)/(输出 token 数−1)` 定义并处理输出不足 2 token 的情况；它与全部 token 间隔样本的分布不同。

## 故障定位与验收

连接被拒绝先查服务日志和就绪状态；只有空流先检查是否解析了角色/usage；取消后 running 不回落则沿 abort 链追踪。HTTP 输出与离线不同时先比较 IDs、模板、sampling 和停止语义，再排查数值路径。

完成后新增 `docs/reports/serving-baseline.md`，写入协议表、时间线、三路径计时边界、失败与取消证据和完整复现命令。

- [ ] 同一 token 输入能通过 completion 复现，chat 模板差异另有记录。
- [ ] 客户端正确解析分片 SSE，保留失败/取消与部分输出。
- [ ] TTFT、chunk interval、TPOT、ITL 的来源和边界明确。
- [ ] 真实取消到资源可用的路径经过验证。
- [ ] 三次在线基准和三路径比较都有环境、输入与配置元数据。
- [ ] 可用请求 ID 在源码中解释一次 `schedule → execute → sample → update`。

下一步进入 [动态调度手册](mini_llm动态调度与连续批处理手册.md)，复用本阶段的请求事件契约。
