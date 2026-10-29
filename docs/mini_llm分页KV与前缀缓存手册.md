# mini_llm 分页 KV 与前缀缓存手册

本手册交付物理 KV 池、每请求 block table 与 PyTorch 参考 attention，并用 vLLM 验证自动前缀缓存的收益边界。对应主计划的 KV 主题及 X05，预计 20–25 小时。前置是 [动态调度](mini_llm动态调度与连续批处理手册.md) 的请求生命周期与真实 batch oracle 已通过。

mini_llm 分页参考实现是必需交付；mini_llm 共享前缀是后续扩展。vLLM APC 冷热对照在本手册内完成，不依赖先实现自有 APC。参考版允许 gather/scatter，目标是机制正确性；未经实测不声称实现了高性能 PagedAttention kernel。

## 实施安排

| 工作 | 时间 | 交付 |
| --- | --- | --- |
| 容量与布局设计 | 3 小时 | 映射公式、容量表、资源不变量 |
| 物理存储与事务分配 | 4–5 小时 | KV 池、预留/提交、释放与复用 |
| attention 和 runner 接入 | 5–6 小时 | 参考 attention、跨块 prefill/decode |
| 边界与多请求回归 | 3–4 小时 | oracle、隔离与耗尽行为测试 |
| vLLM APC 实验 | 3–4 小时 | 开关/冷热/前缀差异对照 |
| 报告与复现 | 2–3 小时 | 分页实现说明与缓存收益边界 |

## 设计真实物理布局

先阅读现有 block manager 和连续 Cache，画出请求、逻辑块、物理块与每层 K/V 的对应关系。

```bash
cd ~/Desktop/mini-vllm
rg -n 'allocate|append|deallocate|ref_count' mini_llm/engine/block_manager.py
rg -n 'cache_position|key_cache|value_cache' mini_llm/layers/attention.py
rg -n 'length|shape|allocate' mini_llm/engine/kv_cache.py
```

建议第一版每层 K/V 各用 `[num_blocks, block_size, num_kv_heads, head_dim]`，与现有连续布局间显式转换。请求的第 t 个已缓存 token 映射为 `logical=t//block_size`、`offset=t%block_size`、`physical=block_table[logical]`。展平 slot 可记为 `physical*block_size+offset`。表项、长度与物理槽位必须共同校验。

| 待开发位置 | 合同 |
| --- | --- |
| `mini_llm/engine/paged_kv_cache.py` | 持有物理池；按层写入、读取；检查槽位范围和实际容量 |
| `mini_llm/engine/block_manager.py` | 在现有接口上增加本轮容量预留，分配失败不提交部分状态 |
| `mini_llm/layers/paged_attention.py` | gather 有效 K/V，构造 causal mask，输出与连续版一致 |
| `mini_llm/engine/model_runner.py` | 消费每请求 block table、已计算长度和 slot mapping |
| `tests/test_paged_kv_cache.py`、`tests/test_paged_attention.py` | 映射、跨块、隔离、回收与 oracle |

先用不共享块的模型，每个物理块只属于一个请求。释放不必清零，但新请求只能读取其有效长度内已写槽位；测试用哨兵值检测越界读和串线。旧 `KVCache` 与单请求生成入口保持可用，避免失去 oracle。

### 容量预算

全注意力、无 TP 分片、相同 KV dtype 下，缓存大小近似为 `2×层数×KV heads×head_dim×元素字节数×已分配 token 槽位数`。本模型为 24 层、2 KV heads、head_dim=64，BF16 每 token 约 12 KiB。

| 量 | block_size=16 时的示例 | 需注意 |
| --- | --- | --- |
| 一个完整块 | 16×12 KiB = 192 KiB | 包含全部层的 K 和 V |
| 单请求 2048 token | 24 MiB | 不含其他运行时分配 |
| 单请求 17 token | 2 个块、32 个槽位 | 有 15 个尾部未用槽位 |
| 128 MiB KV 池 | 可容纳的块数向下取整 | 池容量与模型可启动显存分开 |

多请求分配量使用 `block_size×Σceil(cached_length/block_size)`。运行时若为下一步预留槽位，应将预留计入分配量并在报告区分有效 token。该估算不等于 DRAM 实际读写量，不能直接当成 kernel 带宽证据。

## 先验证存储，再验证 attention

用小模型配置和 CPU FP32，写入可辨识的层号/请求号/token 位置，逐个 gather 检查。随后用固定 Q/K/V 比较连续 attention 与分页参考版，再接完整 Qwen 模型。

分块 prefill 的 query 绝对位置从此前已计算长度开始。对每个 query 只开放本请求中不晚于该位置的 key；padding、其他请求和未来位置不可见。decode 的单 query 读本请求所有有效 KV。GQA 仍按 KV head 分组映射，物理分页不改变模型 head 语义。

| 测试 | 操作 | 通过依据 |
| --- | --- | --- |
| 块边界 | prompt 长度 15/16/17，继续生成跨块 | K/V、logits 与连续 oracle 对齐 |
| 多块 | prompt 31/32/33，分块 prefill 再 decode | 位置和 causal mask 无偏移 |
| 交错分配 | A/B/C 不同长度，穿插追加 | 各请求映射与内容隔离 |
| 释放复用 | 释放 B，由 D 复用相同物理块 | D 不读取 B 的旧值，A/C 不变 |
| 容量耗尽 | 只剩不足一轮所需的空闲块 | token/表/引用数不被部分提交 |
| 取消 | waiting、prefill 后、decode 中分别取消 | 结束后块可重用，无重复释放 |
| 最后输出 | EOS 或长度上限结束 | 未写 KV 的末尾生成 token 不被当成已缓存 |

完整模型先检查每个相同条件前缀上的 logits，再比较 greedy token 输出。记录 FP32 与 BF16 各自误差；当前三后端 20 条输入可继续作为 GPU 回归集合。资源操作测试在 CPU 跑，真正分页模型路径单独标 GPU。

## 增加调度与内存对照

复用动态回放器，比较连续槽位池与分页池：固定物理内存预算、相同输入与到达序列、相同采样。报告有效 KV、已分配 KV、尾块浪费、完成请求数、取消后空闲块和 makespan。

参考 gather attention 可能增加临时连续张量和拷贝。分别报告池占用与整卡峰值，不把逻辑节省直接等同整卡峰值下降。若分页版本更慢，列出 gather/scatter、临时分配和 Python metadata 的成本，留给后续 profiler 检验。

## 用 vLLM 验证 APC

```bash
cd ~/Desktop/vllm
rg -n 'get_computed_blocks|allocate_slots|free' vllm/v1/core/kv_cache_manager.py
rg -n 'hash|cache|evict|ref_cnt' vllm/v1/core/block_pool.py
rg --files examples/features | rg 'automatic_prefix|kv_events'
```

上游 APC 示例有大模型和长文本默认值，先阅读并在 `examples/vllm/prefix_caching/` 改成自己的模型、长度和 engine 配置。待开发 prefix workload 生成器按 tokenizer 构造 256/1024 token 共同前缀，保存完整 IDs、后缀、前缀长度和 hash。

每个独立实验臂重启服务，先用无关前缀预热 kernel，再执行一次测量用冷请求和第二次共享前缀的热请求。至少三组独立重复；APC 开/关各自运行，其他配置一致。冷/热在同一服务生命周期内，跨实验臂不继承 cache。若用 reset API，先确认固定版本支持、无在途请求，核对重置结果。

| 对照 | 输入处理 | 观察 |
| --- | --- | --- |
| APC 关，冷/热 | 相同前缀、不同后缀 | 排除普通预热与温度带来的变化 |
| APC 开，冷/热 | 相同 token 前缀 | TTFT、prefill、命中 token 增量 |
| 开头变更 | 第一段 token 改变 | 命中减少及链式 hash 行为 |
| 块边界变更 | 在已确认的实际块边界前后改一个 token | 完整块的命中范围 |
| decode 较重 | 相同前缀，输出从 32 增到 512 | TTFT 收益与 E2E 收益的区别 |
| 驱逐 | 插入足够多不相关请求，保持可启动容量 | 空闲块、缓存命中下降与再次计算 |

边界实验以 vLLM 日志中的实际块大小为准，不能假设它等于 mini_llm 的 16。APC 命中完整块不代表所有前缀都免计算；最后 logits 或未缓存尾部可能仍需执行。命中率按该窗口 hits/queries 增量计算，不使用服务运行至今的累计比例。

需要制造 Cache 压力时，在确认单请求可容纳后使用该版本的 `--kv-cache-memory-bytes`，例如 134217728 字节。它与显存比例推算分别记录；先保留正常容量对照，不靠不断增加并发触发 OOM。

## 可选的 mini_llm 前缀共享

只有分页正确性完成后，才增加完整块 prefix hash、引用计数和空闲缓存保留策略。hash 需含前块依赖与兼容性身份；模型 revision、KV dtype、位置语义或 adapter 不同不能盲目共享。

共享块只读，未满尾块默认不共享；需要写入共享尾块时做 copy-on-write 或明确拒绝。定义“活跃引用为零但仍缓存”的状态与驱逐条件。测试两个请求共享后取消其一、全部结束后复用、驱逐后重新计算和追加写入隔离。第一版不同时开发跨实例共享。

## 交付与验收

完成后写 `docs/reports/paged-kv.md` 和 `docs/reports/prefix-caching.md`，分别描述自己的实现与 vLLM 实验，不混淆两者能力。

- [ ] 分页池真实存放模型 K/V，block table 已用于 attention 读取。
- [ ] 跨块、交错分配、释放复用、取消和耗尽测试通过。
- [ ] 真实 Qwen BF16 与连续 oracle 的比较有记录。
- [ ] 显存报告区分有效 token、分配槽位、临时张量与整卡峰值。
- [ ] APC 开关/冷热/不同前缀、至少一个无收益案例已验证。
- [ ] 自有共享前缀若未完成，README 明确标为计划中。

下一步进入 [执行链与性能分析](vLLM执行链追踪与性能分析手册.md)，解释参考分页实现和 vLLM 的真实开销。
