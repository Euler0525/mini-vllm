# vLLM KV 交接与并行扩展手册

本手册在单卡上完成顺序 KV 保存/加载和加载失败恢复，再按资源选择真实多卡对照。对应主计划扩展主题与 X11，预计 20–25 小时。前置是 [分页 KV](mini_llm分页KV与前缀缓存手册.md) 的布局理解、模型完整 revision 和正确性基线。

必需交付是单卡顺序 connector 与恢复案例；TP/DP、并发 P/D 和网络 connector 为可选。顺序文件交接展示 KV 生命周期，不能用来声称已完成多 GPU P/D 服务或扩展性能验证。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| connector 契约与数据流 | 3–4 小时 | metadata/KV 双路径图、身份表 |
| 改写小模型顺序示例 | 5–6 小时 | prefill 保存、decode 加载、重算对照 |
| 同步/异步失败恢复 | 4–5 小时 | 注入故障、完成与输出核验 |
| IO 与重算成本 | 3–4 小时 | 字节量、时间边界、适用范围 |
| 并行阅读或可选实测 | 3–4 小时 | TP/DP/P/D 设计或真实多卡记录 |
| 复现与报告 | 2 小时 | 生命周期说明与限制 |

## 先阅读脚本，避免照搬默认值

```bash
cd ~/Desktop/mini-vllm
mkdir -p artifacts/inference/kv-transfer/discovery
cat ../vllm/examples/disaggregated/example_connector/README.md
cat ../vllm/examples/disaggregated/kv_load_failure_recovery_offline/README.md
sed -n '1,180p' ../vllm/examples/disaggregated/example_connector/prefill_example.py
sed -n '1,180p' ../vllm/examples/disaggregated/example_connector/decode_example.py
cat ../vllm/examples/disaggregated/kv_load_failure_recovery_offline/run.sh
rg --files ../vllm/vllm/distributed/kv_transfer | rg 'example|base|factory'
```

这些示例硬编码模型、相对存储目录和文本文件；prefill 生成一个 token 后将文本拼回 prompt，decode 重新读文本。改写时保留来源与版权说明，显式限制上下文、并发、dtype、显存和输出长度。先读 `run.sh` 的清理行为，不在共享目录直接执行含删除动作的脚本。

## 定义交接身份与文件清单

| 身份字段 | 为什么需要 |
| --- | --- |
| 模型与 adapter 完整 revision | 权重不同，即使 shape 相同 KV 也不能直接复用 |
| 原始与续接 prompt token IDs | 文本 decode/re-encode 可能改变 token 边界 |
| KV dtype、层数、KV heads、head_dim | 保证数据格式与容量一致 |
| 位置编码与上下文设置 | 相同 token 在不同位置语义下 K 可能不同 |
| block size、布局、TP 分片信息 | 物理块格式与 rank 归属可能不同 |
| 已保存 token 数与首个输出 token | 区分已计算 KV 和只生成尚未写入的 token |
| connector/schema 版本、文件 hash | 防止读取残留或不完整产物 |

上游演示 connector 不一定实现全部身份检查。自己的变体增加 manifest 和运行前校验；兼容性不明时拒绝使用旧目录，不将缺少校验包装为生产可用功能。

待开发 `examples/vllm/kv_transfer/` 中的 producer、consumer 和 recovery 入口，共享输入/manifest 工具。参数支持 `--model`、`--inputs`、`--storage-dir`、`--output`、资源限制和注入模式，结果拒绝覆盖。

原例文本往返与规范 token 输入要分开验证。优先把 token IDs 写入 JSONL，再使用固定版本支持的 token prompt；若 connector 的 key 依赖文本或其他 metadata，先追实现并保持一致，不擅自更换输入形式。报告可加载的完整块数与尚需重算的尾部。

## 顺序执行正常路径

接口实现后按下面顺序运行，不要求两个模型同时驻留显存。

| 步骤 | 操作 | 通过标准 |
| --- | --- | --- |
| producer | 从固定 prompt prefill 并保存 KV/manifest | 文件完整、实际 token 进度可解释 |
| 退出 | 正常结束 producer，确认显存释放 | 同一 GPU 可启动下一实例 |
| consumer | 读取同一 manifest，加载 KV 并续写 | 有加载证据，不只看到成功输出 |
| 重算臂 | 同样的续接 token 上下文，不加载外部 KV | 输出与加载臂在可解释范围内对照 |
| 重复 | 新目录再次 producer/consumer | 不依赖上次残留文件或 APC |

对齐“续接上下文”尤其重要：producer 产生的首个 token 是否已写入 KV，与 consumer 需要处理的 token 数分开。冷启动时间与保存/加载工作时间分开记录；正常进程退出不代表异步文件写入一定已经完成，确认 connector 的完成协议。

至少三次独立运行，每次记录 KV 文件大小、读写字节、保存/加载窗口、重算 prefill 时间、续写 TTFT、E2E 与输出。OS page cache 的冷/热状态单列；不使用全局 drop-cache 操作干扰其他任务。

## 注入同步与异步加载失败

固定快照 recovery 示例的 `--simulate-failure` 和 `--async-load` 是已有参数。先查看帮助与自定义 connector，再将其行为移入自己的小模型变体；不直接用默认 Llama 模型启动。

```bash
cd ~/Desktop/mini-vllm
~/Desktop/vllm/.venv/bin/python \
  ../vllm/examples/disaggregated/kv_load_failure_recovery_offline/decode_example.py --help \
  > artifacts/inference/kv-transfer/discovery/recovery-help.txt
sed -n '1,220p' \
  ../vllm/examples/disaggregated/kv_load_failure_recovery_offline/load_recovery_example_connector.py
rg -n 'kv_load_failure_policy|invalid|recompute' ../vllm/vllm/v1/core/sched/scheduler.py
```

| 模式 | 操作 | 应验证 |
| --- | --- | --- |
| 正常 | 读有效 KV | 加载完成、输出正确 |
| 同步失败 | 使用示例故障注入 | 无效块识别、重排/重算、请求成功完成 |
| 异步失败 | 故障注入加 async load | 等待与恢复状态转换，无死锁或双重提交 |
| 身份不匹配 | 修改 manifest 中兼容性字段 | wrapper 运行前拒绝；不伪称底层自动检测 |
| 中途取消 | 加载或重算等待时取消 | 在途操作与释放顺序正确，下一请求可完成 |

保存恢复前后日志、无效块/request ID、重算 token 数、完成时间与输出 token。greedy 差异采用相同条件前缀定位，不能只比较打印文本。恢复性能可能更慢，这是恢复成本，重点是状态一致与完成能力。

## 分析 IO 与重算成本

用分页手册的 `12 KiB/token` 作理论容量检查，再读取实际文件字节；序列化、padding 和 metadata 会使二者不同。文件 IO 时间不等于网络传输时间，OS 缓存命中也不是 GPU-to-GPU 带宽。

比较“保存 + 加载 + 补算尾部”与“重算 prefill”的成本，分别给出是否包含启动。传输收益需由减少的计算抵消 IO、同步、排队和存储成本；不能仅凭 KV 可加载就推断 TTFT 更好。

数据流图标明 scheduler 控制、worker KV 数据、host/device 拷贝、磁盘与完成通知。KV events 是事件流，不是实际 KV 数据传输；两个概念在图中分开。

## 可选的真实多卡验证

```bash
cd ~/Desktop/vllm
rg -n 'tensor_model_parallel|data_parallel|pipeline' vllm/distributed/parallel_state.py
rg -n 'all_reduce|all_gather|reduce_scatter' vllm/distributed/communication_op.py
rg --files examples/features/data_parallel examples/features/torchrun
```

无多卡时交付分片/通信图与实验设计，注明仅阅读。获得两张真实 GPU 后，记录 GPU 型号、拓扑、互连、软件栈和总 GPU 数，再比较单卡、TP=2 与两个 DP 副本。DP 的吞吐统计包含两副本总成功请求，TP 模型分片与 KV heads 的可分性需预先校验。

| 实验 | 固定条件 | 解释要求 |
| --- | --- | --- |
| TP 1/2 | 相同模型、全局负载与 SLO | 通信、单请求延迟、总吞吐及每 GPU 成本 |
| 双副本 DP | 同样两张卡、相同总负载 | 路由、负载均衡、排队与容量 |
| 并发 P/D | 独立 prefill/decode GPU 与实际 connector | KV 字节/带宽、排队、TTFT/ITL、失败恢复 |

单卡多进程不作为扩展效率证据；顺序磁盘例不作为网络 P/D 结果。LMCache、Mooncake 等外部 connector 先写依赖、介质与故障责任表，按真实资源选择一种，不在本机堆齐全部服务。

## 交付与验收

报告为 `docs/reports/kv-transfer-recovery.md`，多卡实测如有则单独整理。

- [ ] 顺序 producer/consumer 使用小模型与独立目录，可从空结果目录复现。
- [ ] 模型、token、位置与 KV 布局身份齐全，首 token 与尾部重算可解释。
- [ ] 正常、同步失败、异步失败均有输出和状态恢复证据。
- [ ] 保存/加载、进程启动、重算和续写时间分开报告。
- [ ] 理论字节、文件大小、磁盘 IO 与网络性能没有混淆。
- [ ] 多卡能力明确标实测或仅设计。

下一步按 [发布与独立复现手册](开源推理项目发布与复现手册.md) 形成可维护版本。
