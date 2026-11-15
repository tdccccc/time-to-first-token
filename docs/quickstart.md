# 快速上手：跑通并测量第一个服务

先完成一个可检查的闭环，再回章节解释原理。这里有两条路径：**无 GPU 的 CPU mock**验证客户端；**真实模型服务**建立后续优化的基线。两者的结果不能混用。

!!! note "运行目录与验证范围"
    以下命令都从仓库根目录执行。CPU 路径只用 Python 3.10+ 标准库，不需要安装 MkDocs、PyTorch 或推理引擎。GPU 路径是参考配方，仍需在你的支持设备、版本组合和模型上实机验证。

## 路径 A：无 GPU，先验证测量工具 { #cpu-mock }

### 1. 检查 Python 与脚本入口

```bash
python3 --version
python3 docs/modules/M0-foundations/scripts/bench_openai.py --help
python3 docs/modules/M0-foundations/scripts/mock_sse.py --help
```

完整源码：[压测客户端](modules/M0-foundations/scripts/bench_openai.py) · [mock SSE 服务](modules/M0-foundations/scripts/mock_sse.py)。

### 2. 在终端 A 启动 mock

```bash
python3 docs/modules/M0-foundations/scripts/mock_sse.py --port 8400
```

mock 监听本机回环地址，按固定节奏发送内容，并返回服务端 token 计数。它没有模型计算、KV 管理或 GPU 调度，不能用于比较芯片性能。

### 3. 在终端 B 保存条件并跑冒烟

先保存实际客户端环境。下面的条件文件描述的是 CPU mock，不是 GPU 模型：

```bash
python3 - <<'PY' > conditions-cpu-mock.json
import json, platform, sys
print(json.dumps({
    "conditions_id": "cpu-mock-env1",
    "platform": "cpu-mock",
    "model": "mock",
    "python": sys.version,
    "system": platform.platform(),
    "client": "docs/modules/M0-foundations/scripts/bench_openai.py",
    "server": "docs/modules/M0-foundations/scripts/mock_sse.py",
    "server_command": "python3 docs/modules/M0-foundations/scripts/mock_sse.py --port 8400",
    "note": "Only the measurement workflow is under test; no model or device benchmark."
}, ensure_ascii=False, indent=2))
PY

python3 docs/modules/M0-foundations/scripts/bench_openai.py \
  --base-url http://127.0.0.1:8400 --model mock \
  --max-tokens 16 --concurrency 1 4 --requests 20 \
  --test-id cpu-mock-r1 --conditions-id cpu-mock-env1 \
  --output results-cpu-mock-r1.json
```

`20` 是冒烟请求数；并发只有 `1/4`，不是宣称用 20 个请求测出 32 或 64 并发。默认 mock 为每个内容 chunk 设置约 10 ms 间隔；本例每请求最多 16 个输出 token，总时长为数秒量级，具体以本机调度为准。

### 4. 检查输出，而不只看“命令成功”

打开结果 JSON，逐档确认：

- 成功/失败数量、实际在途峰值与请求数符合预期。
- 第一个 role 空块不计 TTFT；首个非空内容块才开始内容时间线。
- token 来源是服务端 usage，而不是字符估计；每条请求有长度和时间记录。
- `ITL` 是内容 chunk 的间隔近似，不能把合并 chunk 还原成真实逐 token 时间。
- 多 token 请求的 TPOT 与内容时间线一致；不足两个输出 token 时不报告虚构的平均间隔。
- 汇总吞吐使用整个测量窗口，失败耗时不从分母中悄悄删除。

你现在得到的是**客户端方法验证记录**。不要把这份 JSON 的延迟、吞吐填进真实模型或跨芯片报告。

结束后在终端 A 按 `Ctrl+C` 停止 mock。进一步理解见[第 4 章](chapters/chapter-04.md)与[第 2 章](chapters/chapter-02.md)。

## 路径 B：有支持设备，部署真实模型 { #real-service }

### 1. 先固定环境与硬件预算

使用目标引擎对应版本的安装说明，不把三个引擎装进同一个未经核对的环境。保存 Python、PyTorch、引擎、运行时与驱动版本，以及镜像 digest 或环境锁定文件。

`nvidia-smi` 中的 CUDA 字段表示驱动支持的 CUDA 上限，**不是**已安装 Toolkit 或 PyTorch 构建版本。分别核对 `nvcc --version`（若安装 Toolkit）、PyTorch 运行时及引擎环境。

7–8B 的 16 位权重通常约 14–17 GB，还需要 KV、工作区与图捕获空间。24GB 可作为短上下文起点，不保证所有模型、版本和并发组合都能运行。

### 2. 起一个短上下文、低并发的服务

下面使用无需先接受 gated 模型许可的 Qwen2.5-7B-Instruct 作为参考；硬件、模型与量化支持以目标版本文档为准。本轮不把这条配方标为实机验证。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype half --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 --host 127.0.0.1 --port 8000
```

先保存完整命令、模型 revision、实际版本与 KV 池日志。命令在服务机上执行；远程客户端使用 SSH 隧道等受控入口，不要直接暴露未鉴权的服务。

### 3. 用流式内容冒烟

在同一服务机的另一终端运行：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Explain why the sky is blue in one short paragraph."}],"max_tokens":16,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

确认内容块、结束状态与 usage 均正常，而不是仅看到进程启动或 HTTP 200。若发生 OOM、401/403、端口拒绝或字段不支持，先修环境和接口，不继续跑容量扫描。

### 4. 跑第一份有限工作负载基线

把路径 A 的条件表升级为真实环境记录，改写 `platform/model/conditions_id`，补上设备、权重 revision、服务命令和缓存设置。再运行：

```bash
python3 docs/modules/M0-foundations/scripts/bench_openai.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 128 --concurrency 1 8 16 --requests 128 \
  --test-id gpu-baseline-r1 --conditions-id qwen25-env1 \
  --output results-gpu-baseline-r1.json
```

这是第一次基线，不是稳态容量结论：有限工作负载有启动与排空期；`max_tokens` 只是输出上限；默认短 prompt 也不等于 512 个输入 token。输入/输出实际长度从 usage 或相同 tokenizer 核实。

固定 prompt 可复现输入，但重复前缀可能命中缓存。将本次明确标为热缓存/冷缓存/关闭缓存中的一种；不同缓存条件的数不能直接比较。正式容量和优化实验继续到[第 9 章监控](chapters/chapter-09.md)、[第 11 章容量评估](chapters/chapter-11.md)；再按瓶颈选读优化。

## 下一步选一条路线

| 目标 | 下一步 |
|---|---|
| 解释指标与瓶颈 | [基础与度量](parts/part-1.md)、[第 6 章](chapters/chapter-06.md) |
| 调优服务 | [框架路线](parts/part-2.md)：监控、容量、按问题优化 |
| 写并接入算子 | [算子路线](parts/part-3.md)：正确性、计时、分发与端到端 |
| 适配昇腾 | [国产路线](parts/part-4.md)：先装环境，再查覆盖与迁移 |

每一次实验都升级同一份记录，使用[统一实验约定](experiments.md)；需要解释术语时查[术语表](glossary.md)。
