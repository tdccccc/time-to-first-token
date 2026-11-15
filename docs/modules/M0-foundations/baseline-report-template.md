# 基线报告（模板）

先按[快速上手](../../quickstart.md)或第 8 章第 2 节起服务，按[实验记录约定](../../experiments.md)保存同一份可追溯材料。空格写待测，模拟数据不填入真实模型性能栏；一次 HTTP 成功不代表模型质量或 GPU/NPU 性能已验证。

## 报告头与验证状态

| 项目 | 值 |
|---|---|
| `test_id` | 本次运行/重复编号；每档使用 JSON 中的 `levels[].test_id` |
| `conditions_id` | 环境与测量条件编号；自动 `client-...` 哈希仅覆盖客户端参数，需另附环境清单 |
| `raw_path` | 原始 JSON 的实际路径，对应 `--output`；每个请求可追到 `request_id` |
| 日期 / 报告人 | YYYY-MM-DD / |
| 当前已验证 | 分别填写：文档核对 / CLI 与 mock / 实机接口 / 质量回归 / 真实性能；未做的写待测 |
| 本次数据状态 | mock 已验证 / 真实环境已测 / 失败 / 无效 / 待测；不得把 mock 升级为实机测量 |
| 未运行检查 | GPU/NPU、精度、编译、监控或其他未完成项及补测条件 |

## 环境

| 项目 | 值 |
|---|---|
| 硬件 / 隔离条件 | 型号、显存、卡数、互联；是否有其他负载。mock 写 CPU fixture，无 LLM/GPU |
| 驱动 | `nvidia-smi` 的驱动版本或目标平台对应输出 |
| CUDA toolkit / runtime | toolkit 用 `nvcc --version`；运行时/容器版本另记。`nvidia-smi` 顶部 CUDA 是驱动支持上限，不是已装 toolkit |
| 引擎 / 镜像 / 插件 | 包版本、镜像 digest、插件版本；未核实不猜版本号 |
| 模型 / tokenizer | 权重仓库与 revision、量化格式、tokenizer 和聊天模板 |
| 服务启动命令 | 完整命令，包含上下文、并发、显存、缓存与编译的实际配置 |

## 测量条件

| 项目 | 值 |
|---|---|
| 客户端脚本 | [`bench_openai.py`](scripts/bench_openai.py) 与 JSON 的 `script_sha256` |
| 固定输入 | JSON 内完整 `prompt`、`prompt_sha256` 与 `prompt_source`；默认英文短句不等于 512 token |
| 实际输入长度 | `request_records[].prompt_tokens` / 每档 `prompt_tokens` 分布；usage 不提供时为未知 |
| 输出上限 / 实际长度 | `max_tokens` / `request_records[].completion_tokens` 与每档 `completion_tokens` 分布 |
| token 来源 | `completion_tokens_source`、`tokens_estimated`、`estimated_requests`；估算值不当精确 token/s |
| C / N | 请求并发上限与每档总请求数；默认 N=128，正式示例 N=1024；20 请求仅作 C=1/4 的 mock 冒烟，N < 任一 C 必须拒绝 |
| 在途与窗口 | `observed_peak_inflight`、`observed_mean_inflight`、`wall_s`；有限批次包含启动/排空，不声明恒定并发或稳态 |
| 预热 / 重复 / 超时 | 另做的预热方式和请求数、r1/r2/r3、socket 超时；脚本不会自动剔除启动与排空期 |
| 统计口径 | `(n−1)×q` 线性插值；成功请求分位数、样本量、失败/超时另列；单 token 或单内容 chunk 的 TPOT 为 `null` |

## 结果

每档一行；成功总输出 token 除以整档墙钟时间，不只计成功请求占用的时间。ITL 为内容 chunk 间隔近似，未定义 TPOT 填 n/a。

| 档位 `test_id` | C / 观测峰值 | 成功 / N；失败 / 超时 | TTFT p50/p95/p99（s） | ITL chunk p50/p95（s） | TPOT p50（s） | 成功总延迟 p95（s） | 输出 token/s | 验证状态 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

- 原始来源：`levels[].request_records`；成功长度分布 `completion_tokens`，总输出 `total_completion_tokens`，吞吐 `throughput_tok_per_s`。
- 成功延迟：`total_s` 分布；到成功结束或失败判定的全请求耗时：`all_request_total_s`，两者分开报告。
- 未定义 TPOT 数量：`tpot_undefined_requests`；失败原因：逐请求 `error_type` / `error`。
- 服务端另采：waiting/running、KV、抢占、显存；未观测的项目写未观测，客户端在途不是服务端队列或 batch。

## 拐点与归因

- 相邻档位的吞吐增益、尾延迟、失败率变化；拐点写区间，饱和/失败档位保留并单列状态，不冒充正常容量。
- SLO 与候选配置：明确 TTFT/TPOT/成功率预算、实际输入输出长度和负载方式。
- 瓶颈候选 / 证据 / 下一步对照：
- 是否独立取得稳态窗口：没有则不以配置 C 直接进行 Little 定律反查，也不以 p50 代替均值。

## 结论与补测

- 在 `conditions_id=...` 下，已验证 ______；本次数据能支持 ______，不能支持 ______。
- 待测项、需要的设备/版本、使用的命令、验收标准：
- 后续优化沿用这份记录与条件编号；改变数值路径后重新做正确性/质量回归，再比较端到端性能。
