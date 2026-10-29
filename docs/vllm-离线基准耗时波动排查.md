# vLLM 离线基准耗时波动排查

本文记录 2026 年 10 月 21 日对单请求离线基准波动的排查。负载是一条 128 token 输入，顺序生成 32 token，忽略 EOS；后端为 vLLM eager，关闭 prefix caching。运行环境使用 Qwen2.5-0.5B-Instruct、RTX 4060 Laptop GPU 和 i9-14900HX。[运行脚本](../../benchmarks/inference/run_inference.py) 将一次同步 `LLM.generate` 调用的墙钟耗时记为 `offline_call_ms`；模型加载、预热、结果解码和 JSON 写入不计入测量窗口。顺序输出吞吐为输出 token 总数除以 `measurement_window_s`。

## 现象

每格数值依次为请求耗时中位数（ms）和顺序输出吞吐（token/s）。前两行是排查期间读取到的历史结果；同名 JSON 后来被重新生成，不能再用当前文件复核这些旧数值。最后一行对应撰写本文时的 `artifacts/inference/results/vllm-bench-recheck-{1,2,3}.json`。

| 记录阶段 | 预热 / 重复 | 第 1 次 | 第 2 次 | 第 3 次 |
| --- | ---: | ---: | ---: | ---: |
| 最初观察 | 5 / 30 | 201.78 / 162.74 | 164.80 / 193.35 | 164.88 / 193.40 |
| 增加次数后 | 10 / 50 | 173.11 / 185.21 | 164.41 / 193.81 | 163.71 / 194.47 |
| 当前文件 | 10 / 50 | 163.84 / 194.44 | 163.68 / 194.76 | 163.74 / 194.57 |

最初的 30 次测量中，第 1 次运行前 15 个请求的耗时中位数为 223.76 ms，后 15 个降至 164.37 ms。增加到 10 次预热、50 次重复后，第 1 次仍有分段波动，按每 10 个请求统计的中位数依次为 179.85、171.98、165.77、175.57 和 169.40 ms；第 2、3 次约为 164 ms。三轮输入、生成长度和推理配置相同，每个请求均生成 32 token，因此吞吐差异主要来自调用耗时。预热次数不计入 `requests`；该字段的条数由输入条数与 `--repeats` 共同决定。

## 排查证据

- 两次额外的默认调度复测写入 `/tmp/mini-vllm-diagnostic-{1,2}.json`，均为预热 10 次、重复 50 次。耗时中位数分别为 190.44 和 176.96 ms；第 1 次按 10 个请求分组的中位数为 190.59、191.55、191.96、175.04、187.94 ms，第 2 次为 180.78、181.39、181.53、172.09、168.08 ms。第 2 次未连续采样 GPU，仍有慢速段，因此采样不是唯一原因。这些临时文件不属于仓库成果，可能被系统清理。
- 在较慢的复测中，GPU 运行时采样多为 P0、2595–2610 MHz，功耗约 73–80 W、温度约 57–61°C；另一次逐请求 NVML 采样得到约 2595–2610 MHz、59–64°C，功率限制读数为 125 W。没有观察到与慢速段对应的明显降频，但采样不能排除短于采样间隔的变化。
- 临时诊断将 vLLM 的 `disable_log_stats` 设为 `False` 以读取请求时间戳。排队时间通常小于 0.03 ms，`LLM.generate` 的墙钟耗时比引擎内 `scheduled_ts` 到 `last_token_ts` 的间隔约多 1 ms。额外耗时主要发生在 EngineCore 的请求执行阶段，而不是排队、结果解码或模型加载。开启统计会改变少量运行开销，这组数据用于定位阶段，不直接替代正式基准。
- `lscpu -e=CPU,CORE,MAXMHZ` 显示逻辑 CPU 0–15 的最高频率为 5.6–5.8 GHz，16–31 为 4.1 GHz；电源配置为 `balanced`。限制在 0–15 上的临时运行中位数为 225.00 ms，限制在 16–31 上则为 394.55 ms，说明 CPU 分配对 eager 路径影响显著。前者比默认调度更慢且在运行中继续变慢，因此不能把“只绑 P 核”当作修复。另一次默认调度诊断中，EngineCore 在请求结束时均位于 0–15，无法据此断言最初第 1 次运行曾被迁移到 E 核。

## 电源模式实测差异

在这台 i9-14900HX 机器上，逐个执行 `powerprofilesctl set <模式>` 后立即读取内核参数。下表中 CPU 0 代表 P 核，CPU 16 代表 E 核；频率单位是 MHz，PL1/PL2 是 CPU 封装功率限制。测试前的模式为 `performance`，读取完已恢复。

| 本机实际参数 | `performance` | `balanced` | `power-saver` |
| --- | --- | --- | --- |
| CPU 0 / 16 的 `energy_performance_preference` | `performance` / `performance` | `balance_performance` / `balance_performance` | `power` / `power` |
| CPU 0 / 16 的 `scaling_governor` | `powersave` / `powersave` | 同左 | 同左 |
| CPU 0 的 `scaling_min_freq` / `scaling_max_freq` | 800 / 5600 | 同左 | 同左 |
| CPU 16 的 `scaling_min_freq` / `scaling_max_freq` | 800 / 4100 | 同左 | 同左 |
| `intel_pstate` 的最低 / 最高性能百分比 | 15% / 100% | 同左 | 同左 |
| Turbo 开关 `no_turbo` | `0`，允许 Turbo | 同左 | 同左 |
| CPU 封装 PL1 / PL2 | 140 / 190 W | 同左 | 同左 |
| `lscpu` 的 `MAXMHZ` | P 核 5600–5800，E 核 4100 | 同左 | 同左 |

**这台机器上能直接确认的模式区别是 CPU 能效偏好值。**`performance` 要求 CPU 更积极地追求性能，`balanced` 取中间倾向，`power-saver` 更偏向省电；[Linux `intel_pstate` 文档](https://www.kernel.org/doc/html/latest/admin-guide/pm/intel_pstate.html)说明，HWP 会利用这个提示自行选择运行状态。`scaling_governor` 三种模式下都叫 `powersave`，它不是 `powerprofilesctl` 当前模式的名称。`lscpu` 的 `MAXMHZ` 和 sysfs 频率上下限是允许范围，本机切换模式不会改写它们；`lscpu` 的实时 `MHZ` 随负载变化，单次读数不能当作模式的固定频率。本机 `powerprofilesctl list` 的 `CpuDriver` 均为 `intel_pstate`，`balanced` 和 `power-saver` 的 `PlatformDriver` 为 `placeholder`，也没有可读取的 `/sys/firmware/acpi/platform_profile`。上述读数不能推断 GPU 时钟或 GPU 功率限制是否变化。

可用下面的命令逐个复核表中最关键的差异，并在结束时恢复原模式。

```bash
original_mode=$(powerprofilesctl get)
for mode in performance balanced power-saver; do
  powerprofilesctl set "$mode"
  printf '%s: ' "$mode"
  cat /sys/devices/system/cpu/cpufreq/policy{0,16}/energy_performance_preference | paste -sd /
done
lscpu -e=CPU,MAXMHZ,MINMHZ | awk 'NR == 1 || $1 == 0 || $1 == 16'
powerprofilesctl set "$original_mode"
```

2026 年 10 月 21 日按下表顺序各运行一次相同负载：一条 128 token 输入，预热 10 次，测量 50 次，每次生成 32 token。每格耗时是 50 个请求的中位数，吞吐是 1600 个输出 token 除以测量窗口。JSON 保存在 `/tmp/mini-vllm-power-<模式>.json`，首尾两次分别使用 `performance-before` 和 `performance-after`，临时文件可能被清理。

| 测试顺序与模式 | CPU 能效偏好 | 请求耗时中位数（ms） | 输出吞吐（token/s） | 每 10 次请求耗时中位数（ms） |
| --- | --- | ---: | ---: | --- |
| 1. `performance` | `performance` | 172.17 | 186.19 | 171.87、169.95、173.04、172.26、172.94 |
| 2. `balanced` | `balance_performance` | 172.17 | 185.73 | 172.64、170.06、173.77、170.56、173.03 |
| 3. `power-saver` | `power` | 183.79 | 155.74 | 220.65、221.48、184.27、177.28、183.91 |
| 4. 恢复 `performance` | `performance` | 183.55 | 175.78 | 183.41、184.01、183.48、183.21、183.74 |

前两组几乎相同；节能模式的前 20 次测量明显变慢，拉低了整轮吞吐。恢复性能模式后耗时仍比第一组高约 11 ms，表明测试期间存在模式之外的时间漂移。这四次顺序测试说明本机模式会改变 CPU 的能效偏好，也观察到节能模式中的慢速段，但不足以单独量化三种模式对推理速度的因果影响，更不能据此认定最初三轮差异由模式切换造成。测试前的模式是 `performance`，结束后已恢复并用 `powerprofilesctl get` 确认。

## 结论与复查

已确认波动发生在 vLLM 的请求执行阶段，CPU 亲和性能够显著改变耗时；单纯增加预热或重复次数不能保证跨进程结果一致。三种电源模式会改变本机 CPU 的能效偏好，但模式对照中也出现了时间漂移。当前证据不足以把最初第 1 次变慢唯一归因于某次核心迁移、电源模式、GPU 降频、温控或后台任务，因为那次运行没有同步记录进程核心位置与硬件状态。报告结果时应保留三轮数值及离散程度，不应无说明地丢弃较慢的一轮，也不要把旧的 `benchmark-summary.json` 与重新生成的同名 JSON 混用。

复查时从仓库根目录运行，输出文件名需要此前未使用过；脚本拒绝覆盖结果。先用 `powerprofilesctl get` 记录当前模式，需要对照时用 `powerprofilesctl set performance`、`powerprofilesctl set balanced` 或 `powerprofilesctl set power-saver` 切换，测试结束恢复原模式。另一终端可用 `nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500` 同步记录 GPU 状态，必要时再记录 EngineCore 的 CPU 核心位置。复核亲和性影响时，可在下面的 Python 命令前分别加 `taskset -c 0-15` 和 `taskset -c 16-31`，并使用不同输出文件名。连续采样本身可能影响短请求计时，正式基准与诊断采样应分开比较。

```bash
for run in 1 2 3; do
  ~/Desktop/vllm/.venv/bin/python benchmarks/inference/run_inference.py \
    --backend vllm --inputs bench-input.jsonl \
    --output "vllm-diagnosis-$run.json" \
    --max-new-tokens 32 --ignore-eos --warmups 10 --repeats 50
done
```

下列命令使用实际请求条数，不再固定断言为 30，并显示每 10 个请求的耗时中位数。若输出文件名改变，同步修改文件名模式。

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from statistics import median

root = Path("artifacts/inference/results")
for run in (1, 2, 3):
    data = json.loads((root / f"vllm-diagnosis-{run}.json").read_text())
    rows = data["requests"]
    times = [row["offline_call_ms"] for row in rows]
    throughput = sum(len(row["token_ids"]) for row in rows) / data["measurement_window_s"]
    blocks = [round(median(times[i:i + 10]), 2) for i in range(0, len(times), 10)]
    print(run, len(rows), round(median(times), 2), round(throughput, 2), blocks)
PY
```
