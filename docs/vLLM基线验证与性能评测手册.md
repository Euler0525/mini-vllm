# vLLM 基线验证与性能评测手册

按本文从上到下执行，可以完成 [主计划](vLLM学习与求职计划.md) 的基线里程碑和 [examples 计划](vLLM-examples实践计划.md) 的 X01。预计 20–25 小时，安装和下载时间另计。环境按 Linux、RTX 4060 Laptop 8GB、单卡、BF16 设计；沿用已有 Qwen2.5-0.5B 权重与 mini_llm。

本阶段结束时应具备四份材料：可追溯的环境记录；20 条固定输入在 HF、mini_llm、vLLM 下的对照；一次预热后的单请求离线基准；从输入到输出的源码调用图。每日先完成命令与观察，再回答问题。遇到失败时按文末排查表定位，不把未运行的结果填成通过。

文中使用的工具已放在仓库的 `benchmarks/inference/`，按功能组织并持续复用。本文前四天的解释结合了本机保存的 GPU 运行结果和 [推理输出一致性排查流程](推理输出一致性排查流程.md)；实际结论仅适用于文中记录的模型 revision、源码版本、配置和输入集合。后续重新运行时应以新生成的日志为准，不直接沿用旧数值。

## 本阶段安排

| 天 | 时间 | 执行内容 | 当天验收 |
| --- | --- | --- | --- |
| 第 1 天 | 3–4 小时 | 固定路径、两个环境、模型 snapshot，运行 preflight 与已有测试 | 两个环境可访问 CUDA；模型 revision 与源码 commit 有记录 |
| 第 2 天 | 3 小时 | 跑 generate/chat；生成 20 条统一 token 输入；读入口 | 能解释原始文本、模板文本与 token IDs 的关系 |
| 第 3 天 | 3–4 小时 | 运行三种后端并定位首个输出差异 | 三份结果、对照报告与至少一段解释 |
| 第 4 天 | 3 小时 | 追踪请求、EngineCore 与调度输出 | 字段表、调用图和一个请求的状态说明 |
| 第 5 天 | 3–4 小时 | 跑离线小基准，准备后续在线服务阶段服务配置 | 三次独立运行结果、计时口径、温度/显存记录 |
| 第 6 天 | 3–4 小时 | 验证停止条件，读采样与输出路径 | EOS、输出上限、上下文上限能分别解释 |
| 第 7 天 | 2–3 小时 | 从记录复现一次实验，整理报告与演示 | 10 分钟讲解和后续在线服务阶段问题清单 |

## 第 5 天：保存第一份离线基准

### 测量预热后的单请求调用

本次只测 vLLM eager、APC 关闭、输入 128 token、输出固定 32 token、一次一个请求。每个新进程加载后预热 5 次，再测 30 次；运行三次独立进程。加载与 warmup 不计入测量窗口，JSON 写入也在计时之后。

```bash
cd ~/Desktop/mini-vllm
for run in 1 2 3; do
  nvidia-smi > "artifacts/inference/logs/bench-before-$run.txt"
  ~/Desktop/vllm/.venv/bin/python "benchmarks/inference/run_inference.py" \
    --backend vllm --inputs bench-input.jsonl \
    --output "vllm-bench-run$run.json" \
    --max-new-tokens 32 --ignore-eos --warmups 5 --repeats 30 \
    2>&1 | tee "artifacts/inference/logs/vllm-bench-run$run.txt"
  nvidia-smi > "artifacts/inference/logs/bench-after-$run.txt"
done
```

可在另一个终端采集温度、功耗、显存和粗粒度利用率。进入学习工程根目录后运行下面的前台命令；三个基准结束后在监控终端按 Ctrl-C，只结束自己的采集：

```bash
cd ~/Desktop/mini-vllm
nvidia-smi \
  --query-gpu=timestamp,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm \
  --format=csv --loop=1 \
  > "artifacts/inference/logs/gpu-monitor.csv"
```

若本机某一查询字段不支持，从命令中去掉该字段并记入笔记。查看基准日志，保存实际 backend、runner、KV 容量和启动信息。没有 profiler 的本轮只用于建立初始基线，不判断 kernel 瓶颈。

汇总三个运行结果：

```bash
cd ~/Desktop/mini-vllm
.venv/bin/python - <<'PY' > "artifacts/inference/results/benchmark-summary.json"
import json
import statistics
from pathlib import Path

root = Path("artifacts/inference")
summary = []
for run in (1, 2, 3):
    d = json.loads((root / f"results/vllm-bench-run{run}.json").read_text())
    rows = d["requests"]
    assert len(rows) == 30
    assert all(len(r["prompt_token_ids"]) == 128 and len(r["token_ids"]) == 32
               for r in rows)
    lat = [r["offline_call_ms"] for r in rows]
    elapsed = d["measurement_window_s"]
    assert elapsed > 0
    summary.append({
        "run": run, "requests": len(rows),
        "median_offline_call_ms": statistics.median(lat),
        "min_offline_call_ms": min(lat), "max_offline_call_ms": max(lat),
        "sequential_output_tokens_per_s": sum(len(r["token_ids"]) for r in rows) / elapsed,
        "model_load_s": d["model_load_s"],
    })
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
cat "artifacts/inference/results/benchmark-summary.json"
```

这里的 `offline_call_ms` 是一次完整同步调用的墙钟时间，包含引擎与 Python 调用开销；`sequential_output_tokens_per_s` 是此顺序负载测量窗口内的输出吞吐。它们都不是 HTTP TTFT、ITL，也不是服务在高并发下的最大吞吐。30 个请求不适合给出可信的 P99，因此本阶段只报告中位数和范围。`model_load_s` 还可能包含引擎初始化，不等于纯权重 IO 时间。

把结果写成一段完整陈述，注明硬件、模型 revision、输入/输出长度、并发、eager/APC 配置、预热与重复次数、三次离散程度。若温度/功耗不同且结果漂移，先写出证据，再决定是否需要补测。

### 准备后续在线服务阶段服务配置

核对仓库中的 [单卡服务配置](../configs/inference/vllm-single-gpu.yaml) 与 CLI 参数，HTTP 压测按 [在线服务与流式测量手册](vLLM在线服务与流式测量手册.md) 执行。该配置按功能命名，后续阶段继续复用。

```bash
cd ~/Desktop/mini-vllm
cat configs/inference/vllm-single-gpu.yaml
../vllm/.venv/bin/vllm serve --help \
  > artifacts/inference/logs/serve-help.txt
```

在编辑器中核对 YAML 与主计划参数的一致性，记录后续在线服务阶段的启动命令为 `../vllm/.venv/bin/vllm serve --config "configs/inference/vllm-single-gpu.yaml"`。这一步只是准备配置，没有验证 HTTP 服务已经运行。

## 第 6 天：停止条件、采样与输出

### 区分三种停止原因

先复用现有、使用确定性测试模型的单元测试，观察可控的 EOS、ignore-EOS 和上下文边界行为：

```bash
cd ~/Desktop/mini-vllm
.venv/bin/python -m pytest tests/test_generation.py -v -p no:cacheprovider \
  2>&1 | tee "artifacts/inference/logs/stopping-tests.txt"

.venv/bin/python -m mini_llm.local_generate \
  --prompt "Reply only with OK." --max-new-tokens 32 \
  --temperature 0 --max-model-len 2048 \
  2>&1 | tee "artifacts/inference/logs/mini-natural-stop.txt"

.venv/bin/python -m mini_llm.local_generate \
  --prompt "Reply only with OK." --max-new-tokens 8 \
  --temperature 0 --max-model-len 2048 --ignore-eos \
  2>&1 | tee "artifacts/inference/logs/mini-fixed-stop.txt"
```

第一个真实模型请求不保证一定提前产生 EOS，记录实际停止原因；确定性单元测试负责覆盖提前 EOS。第二个请求忽略 EOS 后应按 8 个输出 token 结束。停止后的 special token 是否被解码隐藏，与底层输出 token 数是两件事。

再用统一脚本取消 `--ignore-eos`，对前三条输入各最多生成 32 个 token：

```bash
cd ~/Desktop/mini-vllm
.venv/bin/python "benchmarks/inference/run_inference.py" \
  --backend hf \
  --output "hf-natural.json" --limit 3 --max-new-tokens 32

.venv/bin/python "benchmarks/inference/run_inference.py" \
  --backend mini \
  --output "mini-natural.json" --limit 3 --max-new-tokens 32

~/Desktop/vllm/.venv/bin/python "benchmarks/inference/run_inference.py" \
  --backend vllm \
  --output "vllm-natural.json" --limit 3 --max-new-tokens 32
```

对照每个结果的长度、末尾 token ID 和停止字段。HF/mini 的实验循环使用 tokenizer 的单个 EOS；vLLM 还按模型配置处理终止语义，需要核对其 EOS/stop token 配置。不同后端的字段字符串本身不要求相同，例如 `length` 与 `max_tokens` 可能描述同一类上限。

| 条件 | 含义 | 本阶段如何验证 |
| --- | --- | --- |
| EOS | 模型产生了被配置为终止的 token | 确定性单元测试 + 真实生成记录 |
| `max_new_tokens` / `max_tokens` | 最多生成多少新 token | 忽略 EOS 的固定 8/16/32 token 实验 |
| `max_model_len` | 输入和生成上下文的总长度限制 | 现有边界测试；统一脚本在加载前校验输入+请求上限 |

统一脚本会主动拒绝输入长度加请求输出预算超过 2048 的 case。这是实验脚本的前置检查，不是 vLLM 引擎边界处理已被测试的证据。引擎在不同接口下拒绝还是截断/约束预算，需要在后续接口测试中单独验证。

### 追踪从 logits 到文本

```bash
cd ~/Desktop/vllm
rg -n 'class Sampler|def forward|argmax|topk|temperature' vllm/v1/sample/sampler.py
rg -n 'max_tokens|eos_token_id|stop_token_ids|ignore_eos' vllm/v1/core/sched/utils.py
rg -n 'process_outputs|finish_reason|detoken' vllm/v1/engine/output_processor.py
rg -n 'stop|eos|decode' vllm/v1/engine/detokenizer.py
```

结合 [mini_llm 的生成循环](../mini_llm/engine/generation.py)、[vLLM Sampler](../../vllm/vllm/v1/sample/sampler.py) 和 [输出处理](../../vllm/vllm/v1/engine/output_processor.py)，写出四个边界：forward 的输出是什么；采样由谁做；哪些停止条件在引擎中处理、哪些与输出文本相关；客户端最终看到的字符串怎样产生。

本阶段 greedy 对照不会验证 temperature/top-p 的统计行为。若试随机采样，另存配置和结果，不能要求不同环境设置相同 seed 后文本必然逐字相同。

## 第 7 天：整理并独立复现

按下面提纲写 1–3 页报告，完成验证后整理到 `docs/reports/inference-baseline.md`（需要时创建 `docs/reports/`），内容填写实际观察，不复制计划作为实验结论。

| 部分 | 必填内容 |
| --- | --- |
| 环境 | GPU/显存/功耗条件、驱动、两个环境的包版本、源码 SHA、模型 revision、导入位置 |
| 输入 | 数据集校验值、模板、EOS、20 条 token 输入的长度范围 |
| 正确性 | HF/mini/vLLM 对照数量、第一处分歧、top-5 观察；无差异时说明范围 |
| 性能 | 128 输入/32 输出、顺序单请求、三次结果、计时边界、温度/显存观察 |
| 源码 | 主要调用图，5 个请求字段，`LLMEngine.step` 与 `EngineCore.step` 的区别 |
| 停止 | EOS、输出上限、上下文上限的证据与尚未覆盖的情况 |
| 问题 | 至少 3 个明确问题，各附下一步验证方法 |

使用 `--output` 的新文件名，独立重跑一份 vLLM 正确性结果或离线基准，核对输入 hash、配置和结论。运行产生的 JSON/日志是新产物，之前文件保留；模型缓存、虚拟环境和大二进制保留在本地，通用工具、配置与整理后的报告进入版本管理。

最后用 10 分钟解释：为什么先统一 token IDs；一个请求经过哪些组件；prefill 为什么可以得到第一个输出 token；生成 N 个 token 为什么通常只需要 N−1 次后续 decode；为什么离线调用时间不等于 TTFT；如果 greedy 结果不同，下一步怎样定位。

- [ ] mini_llm preflight 与已有测试通过，真实 CUDA case 没有被误算成通过。
- [ ] vLLM 安装来源、Python 导入位置和完整源码 SHA 已登记。
- [ ] generate/chat 均运行，能展示模板前后的 token 差异。
- [ ] 20 条输入、三份结果与对照报告齐全，差异有定位记录。
- [ ] 三次离线基准与正确计时口径齐全，GPU 条件已记录。
- [ ] 请求调用图、字段表与停止条件说明齐全。
- [ ] 重跑使用新文件名且能复现主要观察，在线服务配置已准备。

## 常见问题

| 现象 | 优先检查与处理 |
| --- | --- |
| 宿主机 `nvidia-smi` 正常，某个 Python 看不到 CUDA | 检查 `sys.executable`、PyTorch build、`torch.version.cuda`、设备可见性与容器/执行环境；分别验证两个环境，不先重装驱动 |
| preflight 找不到模型 | 确认完整 revision 与当前用户的 Hugging Face 缓存；按第 1 天下载，再重跑 preflight |
| `uv sync --frozen` 失败 | 保留依赖与网络错误；检查仓库锁文件和 Python 3.12，不通过随意修改 torch/transformers 版本掩盖问题 |
| 固定 SHA 的 wheel 404 / 不兼容 | 保存 URL 与安装日志；按源码构建条件处理，或另立一个版本完全对应的基线，不静默混用 latest |
| `undefined symbol` 或 torch/CUDA 扩展导入失败 | 核对两个环境是否混装、预编译扩展 SHA/variant、torch 版本；保留完整导入错误 |
| vLLM CUDA OOM | 确认其他模型进程已退出；观察空闲显存；先保持 seq = 1、2048 上下文、eager，必要时统一降到 1024 上下文并重新校验数据长度；按实际显存规划调整资源参数 |
| 显示 `FileExistsError` | 脚本保护旧结果；给 `--output` 使用新文件名，不删旧结果来伪装首跑 |
| 比较报告报输入或配置不一致 | 核对 input hash、模型路径、token IDs、ignore-EOS、logprobs 和输出长度；重新准备同一组配置 |
| 输出文本相同，但 token IDs 不同 | 以 ID 和 special token 记录为准，查看解码是否隐藏 EOS 或不同 tokenization |
| token 不一致但首 token top-5 一样 | 检查真正分歧步；使用共同前缀单步 probe，首 token 信息不足以诊断后续差异 |
| 本阶段想直接扫并发或写 kernel | 把问题记入后续实验；先完成当前输入、正确性和计时证据，使下一阶段有可用基线 |

## 工具入口与资源约定

| 入口 | 作用 | 默认资源位置 |
| --- | --- | --- |
| [prepare_workload.py](../benchmarks/inference/prepare_workload.py) | 准备固定 token 输入与合成性能负载 | `artifacts/inference/inputs/` |
| [run_inference.py](../benchmarks/inference/run_inference.py) | 运行 HF、mini_llm、vLLM，记录配置、输出与离线耗时 | `artifacts/inference/results/` |
| [compare_outputs.py](../benchmarks/inference/compare_outputs.py) | 检查实验条件并定位首个 token 差异 | 从 `artifacts/inference/results/` 读取结果 |
| [vllm-single-gpu.yaml](../configs/inference/vllm-single-gpu.yaml) | 单卡 eager 服务基线 | 从工程根目录启动服务 |

所有工具用 `--help` 查看参数；自定义相对模型路径按工程根目录解析。`run_inference.py` 的相对 `--inputs` 按 `artifacts/inference/inputs/` 解析，相对 `--output` 按 `artifacts/inference/results/` 解析；对照工具的相对结果文件名也按结果目录解析。绝对路径可用于独立的数据集与运行目录。

推理工具自动定位工程目录并记录源码 commit 与 vLLM 导入位置，不依赖自定义环境变量。源码 commit 与安装日志需要一起用于确认 Python 代码及扩展来源。默认 warmup 只覆盖第一条输入形状，正确性集合中其他形状的耗时不能直接解释为充分预热的性能结果；固定形状基准单独测量。

`artifacts/` 用于本地模型链接、生成数据、原始日志、结果和 trace，由 Git 忽略。通用代码与配置提交到仓库；需要展示的结果经整理后进入 `docs/reports/`，附复现命令、配置和适用边界。阶段时间只体现在计划与本文的推进顺序中，不进入程序、目录、配置和报告的命名。

本阶段验收通过后，继续执行 [在线服务与流式测量手册](vLLM在线服务与流式测量手册.md)。全部后续手册、项目交付范围与依赖见 [开源推理项目实施总览](开源推理项目实施总览.md)。
