# 工程脚本与配置参考 { .chapter-title }

<p class="chapter-summary">集中索引本仓库保留的参考源码、最小监控配置模板与跨芯片评估核心维度，供对照正文原理随时查阅。</p>

全书正文侧重讲清原理直觉、公式推导与工程权衡，不让冗长的脚手架代码打断阅读节奏。本页集中整理仓库内保留的完整可执行参考脚本、可观测性最小 YAML / PromQL 模板、GPU 算子优化源码入口以及跨芯片对比的核心评估框架，方便你在深入实践或搭建内部工具链时直接翻阅。

---

## 一、流式压测与本地模拟参考源码

对应 [第 3 章（推理度量与压测方法）](../chapters/chapter-03.md) 与 [测量口径与对比分析方法](methodology.md)，仓库提供了一对**零外部依赖（纯 Python 标准库）**的流式测量与回环模拟参考实现：

- **压测客户端**：[`bench_openai.py`](../modules/part1-foundations/scripts/bench_openai.py)
- **回环模拟流服务端**：[`mock_sse.py`](../modules/part1-foundations/scripts/mock_sse.py)

### `bench_openai.py` 核心设计契约

脚本面向任何 OpenAI 兼容的 `/v1/chat/completions` 流式接口，把 [测量口径手册](methodology.md) 中的严格边界直接固化进代码：

| 模块 | 实现要点 | 防止的测量误差 |
| --- | --- | --- |
| **SSE 帧解析** | 完整解析 `data:` 多行事件、跳过注释行、校验 `[DONE]` 终止符与 UTF-8 合法性 | 拦截半截流、非法 JSON 或静默截断，失败请求不混入成功吞吐 |
| **精确计时口径** | 全程采用 `time.perf_counter()`；TTFT 仅在首个非空 `delta.content` 到达时触发；ITL 保存真实内容 chunk 到达间隔 | 防止空 `role` 块提前结束 TTFT；拒绝把多 token chunk 伪造为逐 token 时间 |
| **Token 计数来源** | 请求强制携带 `stream_options.include_usage=true`；若服务端未返回 `usage` 才回退 `len(text)//4` 并在 JSON 与终端显式标记 `tokens_estimated=true` | 避免把字符经验估算与精确分词统计混为同等数据 |
| **有限负载在途追踪** | 使用 `ThreadPoolExecutor` 施加请求上限 $C$（强制校验 $N \ge C$），记录 `observed_peak_inflight` 与时间加权 `observed_mean_inflight` | 明确区分配置并发上限 $C$ 与真实稳态在途数，防止误套 Little 定律 |
| **完整原始留档** | 输出 JSON 保留 `prompt_sha256`、`script_sha256`、逐请求时间戳/错误类型及线性插值 $(n-1)q$ 分位数汇总 | 确保事后可复算任意分位数、排查异常长尾请求 |

### `mock_sse.py` 与离线脚手架演示

`mock_sse.py` 仅监听本机回环地址，能模拟真实服务端的空 `role` 块、可控首包延迟（`--first-content-delay`）、chunk 间隔（`--interval`）与可选 `usage` 帧（`--no-usage` 可用于验证客户端估算降级分支）。在接入真实 GPU 推理引擎前，可用两条命令快速验证解析与计时链路：

```bash
# 终端 1：启动本地 SSE 模拟流服务
python3 docs/modules/part1-foundations/scripts/mock_sse.py \
  --port 8400 --tokens 32 --first-content-delay 0.05 --interval 0.01

# 终端 2：运行标准流式并发扫描
python3 docs/modules/part1-foundations/scripts/bench_openai.py \
  --base-url http://127.0.0.1:8400 \
  --model mock --max-tokens 16 --concurrency 1 4 --requests 20 \
  --output /tmp/bench-mock.json
```

---

## 二、可观测性最小参考配置（vLLM + Prometheus + Grafana）

对应 [第 7 章（可观测性与容量评估）](../chapters/chapter-07.md)，下面给出拉起 `vLLM /metrics → Prometheus → Grafana` 最小监控闭环的配置参考。

### 1. Prometheus 抓取配置（`prometheus.yml`）

压测或精细观察短时排队毛刺时，建议将 `scrape_interval` 设为 `5s`（默认生产常为 `15s`，但更长间隔容易漏掉几秒级的突发排队）：

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

### 2. 容器编排（`docker-compose.yaml`）

在 Linux 宿主机上运行容器访问宿主机 `8000` 端口时，须显式声明 `extra_hosts: ["host.docker.internal:host-gateway"]`（正式生产部署请将 `latest` 替换为固定镜像版本号或 digest）：

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]
    extra_hosts: ["host.docker.internal:host-gateway"]

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

启动后优先访问 `http://localhost:9090/targets` 确认 `vllm` 目标状态为 `UP`，再在 Grafana（`http://localhost:3000`）中添加数据源 `http://prometheus:9090`。

### 3. 核心 PromQL 速查片段

在 Grafana 中直接查询 vLLM 指标时，务必保留完整的 **`vllm:` 前缀**；当同一 Prometheus 抓取多个模型实例时，请在向量选择器中显式过滤 `model_name="..."` 或在 `sum by (...)` 中按 `model_name` 分组，防止跨模型混合分位数：

```promql
# 1. 首字延迟 TTFT p95（秒）
histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[5m])))

# 2. 每输出 token 平均时间 TPOT p95（秒/token，按请求记样本）
histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:request_time_per_output_token_seconds_bucket[5m])))

# 3. 流式 token 间延迟 ITL p95（秒，按流式事件记样本）
histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:inter_token_latency_seconds_bucket[5m])))

# 4. 端到端请求延迟 p95（秒）
histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))

# 5. 请求在引擎内的排队时间 p95（秒）
histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:request_queue_time_seconds_bucket[5m])))

# 6. 系统输出生成吞吐（token/s）
sum by (model_name) (rate(vllm:generation_tokens_total[5m]))

# 7. 当前执行中 / 等待调度请求数（Gauge 快照）
vllm:num_requests_running
vllm:num_requests_waiting

# 8. KV 缓存块池使用率（0~1，面板乘 100 展示百分比）
vllm:kv_cache_usage_perc

# 9. 按结束原因拆分的完成请求数（区分 stop / length / abort）
sum by (finished_reason, model_name) (increase(vllm:request_success_total[5m]))
```

---

## 三、GPU 算子优化与引擎接入源码索引

对应第三篇（[第 12 章](../chapters/chapter-12.md)、[第 15 章](../chapters/chapter-15.md) 与 [第 17 章](../chapters/chapter-17.md)），仓库保留了三套结构完整的参考源码，均内置输入校验与计时骨架：

| 参考源码入口 | 对应章节 | 核心实现内容 | 编译 / 运行参考命令 |
| --- | --- | --- | --- |
| [`gemm_bench.cu`](../modules/part3-operators/examples/gemm_bench.cu) | [第 12 章](../chapters/chapter-12.md) | FP32 GEMM 四版演进完整 CUDA 入口（`gemm_naive` → `gemm_tiled` → `float4` 向量化搬运 → `2×2` 线程粗化），内置 CPU 参考校验、非整除尾块保护与 CUDA Events 计时 | `nvcc -O3 -std=c++17 -arch=sm_80 gemm_bench.cu -o /tmp/gemm_bench`（按实际 GPU 计算能力调整 `-arch`，如 H100 用 `sm_90`，A40 用 `sm_86`） |
| [`quant_matmul.py`](../modules/part3-operators/examples/quant_matmul.py) | [第 15 章](../chapters/chapter-15.md) | Triton 双路径量化矩阵乘完整实现：INT8 **W8A8**（`matmul_int8_kernel`，per-token × per-channel FP32 缩放 + L2 Swizzle）与 **W8A16 融合反量化**（`matmul_fused_dequant_kernel`） | `python3 docs/modules/part3-operators/examples/quant_matmul.py` |
| [`silu_plugin/`](../modules/part3-operators/examples/silu_plugin/pyproject.toml) | [第 17 章](../chapters/chapter-17.md) | 接入 vLLM 平台扩展体系的最小自定义算子插件包，通过 `vllm.general_plugins` 入口点注册 `SiluAndMul` 的 Triton 实现（含 [`kernel.py`](../modules/part3-operators/examples/silu_plugin/my_plugin/kernel.py) 与 [`silu_and_mul_plugin.py`](../modules/part3-operators/examples/silu_plugin/my_plugin/silu_and_mul_plugin.py)） | 在目标 vLLM 环境中执行 `pip install --no-build-isolation --no-deps -e docs/modules/part3-operators/examples/silu_plugin` |

---

## 四、跨芯片对比评估核心维度框架

在开展跨芯片（如 NVIDIA GPU 对照昇腾 NPU 或其他国产芯片）的正式技术选型或回归评审时，建议按 [第 23 章](../chapters/chapter-23.md) 的范式固化以下七个核心评估维度，使任何结论都能追溯到精确条件并在环境升级后有序复核：

| 评估维度 | 必须固定的关键要素 | 工程防错要点 |
| --- | --- | --- |
| **1. 双侧环境锁** | 硬件型号/卡数/互联拓扑、宿主驱动与固件、实际 CUDA/CANN 版本、PyTorch 与框架适配层、引擎与硬件插件版本、容器镜像 digest | 宿主驱动不在容器内；`nvidia-smi` CUDA 字段仅是上限 |
| **2. 模型与评测集** | 权重仓库 revision、量化参数、Tokenizer 与聊天模板、评测集样本清单与判分脚本、固定采样参数与预设非劣退化阈值 $\delta$ | 禁止跨组换模板或仅靠肉眼抽查代替配对评测 |
| **3. 完整复现命令** | 服务启动参数（TP/PP、显存比例、图模式、前缀缓存开关）与压测参数（并发档位、请求数、预热、超时）全文 | 一个环境变量差异就可能改变图编译或通信后端 |
| **4. 配对精度回归** | accuracy / EM / F1（及 NLL / PPL）的同题配对差异区间与异常截断计数 | 区分“统计不显著”与“满足预设非劣门槛” |
| **5. 多档性能曲线** | 至少覆盖单路（$C=1$）、拐点附近与饱和边界三档的 TTFT / ITL / 吞吐、成功率与实际输出长度 | 重复运行报中位数与范围，稳态容量与过载观测分离 |
| **6. 算子覆盖与归因** | Top 耗时算子在芯片侧的实现路径（原生 / 插件 / 统一层 / 回退）、五步归因表（现象 → 定位 → 机制 → 单变量对照 → 边界） | 占比小的回退不盲目重写，聚焦关键路径热点 |
| **7. 结论有效期** | 报告日期、核实日期与重测触发条件（驱动、SDK、引擎或插件任一升级即触发复核） | 版本演进后旧结论仍对原条件有效，不直接外推新栈 |
