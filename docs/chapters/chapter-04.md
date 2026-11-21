# 第 4 章 Profiling 方法论：看时间花在哪 { .chapter-title }

<p class="chapter-summary">按 torch.profiler → nsys → ncu 三层粒度读懂 CPU/GPU 双泳道，定位热点与时间线空白。</p>

Roofline 告诉你“理论上限在哪”，外部压测告诉你“延迟究竟多慢”，但在两者之间始终隔着一层黑盒：一次 52 ms 的 Prefill 或 10.5 ms 的 Decode 里，究竟哪几个算子吃掉了大头？GPU 是真在拼命算矩阵乘，还是有一半时间干晾在那里等 CPU 发指令？

**Profiling（性能剖析）** 就是把这层黑盒掀开，把一次推理展开成微秒级时间线（Trace）。瓶颈从此不再是一句笼统的“受带宽限制”，而是一个精确的算子名、一段裸露的 GPU 空白，或一行不该出现的同步调用。

## 4.1 三级工具分工与自顶向下原则

GPU 性能分析工具按观测粒度天然分为三层，各司其职、互不替代：

| 层级 | 工具 | 观测粒度与核心内容 | 典型出场时机 |
|---|---|---|---|
| 第一层（算子级） | **`torch.profiler`** | 每个 PyTorch 算子的 CPU/CUDA 耗时、显存占用、调用栈 | **排查起点**：先拉出算子热点榜单，判断时间是否集中在少数重算子上 |
| 第二层（系统级） | **`nsys`**（Nsight Systems） | 全景时间线：CPU 线程、CUDA API、kernel、H2D/D2H 拷贝、NCCL 通信与跨进程调度 | 算子榜解释不了端到端耗时、GPU 空白过多、怀疑框架调度或跨进程同步阻塞时 |
| 第三层（硬件级） | **`ncu`**（Nsight Compute） | 单个 kernel 的底层硬件计数器：SM 计算利用率、HBM 带宽利用率、Occupancy、Warp 停顿原因 | 已经锁死一个占端到端大头的具体 kernel，准备动手改写 CUDA/Triton 实现时 |

### 用 Amdahl 账解释为何顺序不能反

很多新手一听说要优化性能，直接上来就拿 `ncu` 对着某个看起来很慢的 kernel 猛查寄存器和 Bank 冲突——这是典型的本末倒置。背后的硬道理是 **Amdahl 定律**：若某个环节占总耗时的比例为 $p$，将其单独加速 $k$ 倍后，端到端整体加速比 $S_{\text{total}}$ 存在严格天花板：

$$
S_{\text{total}} = \frac{1}{(1 - p) + \frac{p}{k}} < \frac{1}{1 - p}
$$

代两个数立刻清醒：

- 假设一个算子只占端到端总时间的 $p = 5\%$，哪怕你花两周把它的速度翻一倍（$k = 2$），端到端耗时也只从 100% 降到 **97.5%**（$S_{\text{total}} \approx 1.026\times$）；哪怕把它优化到 0 耗时，极限提速也不过 **5.3%**。
- 反之，如果在 `torch.profiler` 榜单里发现 top-3 算子加总才占不到 **15%**，剩下 85% 散落在几百个细碎小算子、Python 调度与 GPU 空转里，这时拿 `ncu` 抠单个 kernel 毫无意义，必须立刻退回第二层 `nsys` 看系统全景。

因此，标准排查顺序永远是**自顶向下：`torch.profiler` 摸清大头 → `nsys` 抓全景空白与同步 → `ncu` 钻单点热点 kernel**。（唯一例外是症状已明显指向框架调度、跨进程等待或多卡集合通信时，可直接由 `nsys` 切入。）

### 最小可用的 `torch.profiler`、`nsys` 与 `ncu` 入口

第一层算子级采集只需要几十行标准 PyTorch 代码：先空跑几轮热身，再打开 CPU 与 CUDA 活动记录，导出 Chrome Trace 文件并用 [Perfetto UI](https://ui.perfetto.dev) 打开可视化查看：

```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule


def profile_forward(model: torch.nn.Module, input_ids: torch.Tensor, trace_path: str = "trace.json") -> None:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    prof_schedule = schedule(wait=1, warmup=3, active=2, repeat=1)

    with torch.no_grad(), profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=False,   # 初次排查先关闭，减少采集开销
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(6):     # 1 wait + 3 warmup + 2 active
            model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```

注意三个开关默认先关：`record_shapes`、`profile_memory` 和 `with_stack` 会显著放大 host 端记录开销，容易扭曲原本的时间线；先抓出大头，定位到可疑算子后再针对性开启。

当需要下探到第二、三层时，对应的典型命令行入口非常简洁：

```bash
# 第二层（nsys）：抓取 CUDA、NVTX 与系统级时间线，生成 .nsys-rep 报告
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -o decode_trace \
  python3 run_step.py

# 第三层（ncu）：只锁死某个热点 kernel 的少量调用，读取 Speed-of-Light 算力/带宽利用率
ncu --section SpeedOfLight --kernel-name-base demangled \
  -k "regex:gemm|flash_fwd" --launch-skip 10 --launch-count 3 \
  python3 run_step.py
```

看 `ncu` 报告的第一眼，永远先盯 **GPU Speed Of Light Throughput** 中的两个百分比——**Compute (SM) Throughput** 与 **Memory Throughput**：如果 Memory 冲上 80%+ 而 Compute 只有 10%，就是实锤的带宽受限；反之 Compute 逼近 80%+ 则是计算受限；如果**两者都远低于 50%**，说明既没喂饱算力也没跑满带宽，往往卡在低占用率（Occupancy）、Warp 停顿（如等待长延迟访存或同步屏障）或 Grid 并行度不足上——第三篇第 12 章会带着你顺着这些硬件计数器逐级改写 kernel。

---

## 4.2 读懂 CPU/GPU 双泳道 Trace

打开 Perfetto 中的 `trace.json` 或 `nsys` 报告，界面上最重要的永远是上下平行的**两条泳道**：

- **上方 CPU 泳道（Host 端）**：记录 Python 代码执行、PyTorch 算子分发（dispatch）、CUDA API 发射（`cudaLaunchKernel`）以及同步阻塞（`cudaStreamSynchronize`）；
- **下方 GPU 泳道（Device 端）**：记录硬件真正执行每个 kernel 以及显存拷贝（`cudaMemcpy`）的起止时间。

由于 CUDA kernel 发射默认是**异步的**（CPU 把任务丢进流队列就立刻往下走，GPU 稍后按序取出执行），同一个算子在 CPU 泳道和 GPU 泳道上的位置往往错开，宽度也大相径庭。**把 CPU 泳道的宽度误当成 GPU 计算耗时，是看 Trace 的第一大忌。**

<figure class="book-figure">
  <img src="../../../assets/figures/profiler-trace.svg" alt="CPU 与 GPU 双泳道 trace 示意：kernel、空白与同步等待" loading="lazy">
  <figcaption><span class="book-figure__label">图 4-1</span>读图顺序：先扫 GPU 泳道找成片空白，再沿虚线向上对齐 CPU 泳道看此刻 Host 在做什么——开头空等源自 CPU 准备，中段空等源自显式同步读回结果，后段则是紧凑的长 kernel。</figcaption>
</figure>

### 对齐双泳道识别四种经典波形

把两条泳道上下对齐观察，推理系统绝大多数性能病灶都会呈现为以下四种肉眼可辨的波形：

1. **CPU 长、GPU 短（发射跟不上 / 开销受限）**：CPU 泳道密密麻麻排满 Python 逻辑与小算子分发，下方 GPU 泳道上却只零星飘着几根微秒级细针，中间隔着大片空白。GPU 算得太快，CPU 根本喂不过来——这是典型的第 1 章“启动开销受限”波形。
2. **GPU 紧挨大块、块间无缝隙（算力/带宽吃满）**：GPU 泳道被连续的粗大色块填满，CPU 泳道早就把后续几步的 kernel 提前塞进队列里排队了。这说明主机调度不是瓶颈，此时直接看 `Self CUDA` 排名前几的重算子即可。
3. **GPU 空白，正上方 CPU 在拷贝或同步**：如图 4-1 中段所示，GPU 突然停摆，向上对齐发现 CPU 正在执行 `cudaStreamSynchronize`、`.item()`、`.cpu()` 或同步 `cudaMemcpy`。一次看似无害的 Python 打印日志或采样结果回传，就会强行把整条异步流水线拦腰斩断。
4. **流水线气泡 vs 结构性等待**：GPU 出现短暂空白并不一定意味着写出了 Bug。判断一段空白是无伤大雅的“小 batch 流水线气泡”还是致命的“结构性等待”，有一个极简检验法——**把 batch 调大一档再采一次**：
   - 增大 batch 后空白迅速被拉长的 kernel 盖住消失，说明只是单流吃不饱造成的气泡；
   - 增大 batch 后空白依然每步雷打不动地卡在原地，那就是实打实的结构性等待（同步屏障、锁竞争或跨卡通信阻塞）。

!!! note "服务级 Trace 的请求归因"
    在 vLLM 或 SGLang 这类连续批处理引擎中，同一段 GPU kernel 往往同时服务几十条混合请求。想在 `nsys` 中区分某一段到底属于哪次迭代、是 Prefill 还是 Decode，应借助引擎内置的 **NVTX（NVIDIA Tools Extension）** 标记给调度、前向、采样阶段打上可读区间名，或在单点诊断时先用 `batch=1` 隔离排查，切忌在混批 Trace 里凭感觉把某块 kernel 算给单条请求。

---

## 4.3 两个常见读数误区与采集纪律

看懂波形之后，在从 Profiling 报表里提取结论前，先绕开两处高频数字陷阱。

### 误区一：分母陷阱（GPU 时间 vs 端到端时间）

经常听到有人说：“Profiler 显示这个 GEMM 占了 **80%**，优化它一定能让吞吐起飞！”——这句话缺了最要命的东西：**80% 的分母究竟是谁？**

对比下面两组端到端同为 100 ms 的推理任务：

| 场景 | 端到端总耗时 | GPU 实际活跃时间 | 某算子耗时 | 占 GPU 时间比例 | 占端到端时间比例 | 优化方向 |
|---|---:|---:|---:|---:|---:|---|
| 场景 A（空洞多） | 100 ms | 50 ms | 40 ms | **80%** | **40%** | 先消灭 50 ms 的空白与同步，再动算子 |
| 场景 B（吃得满） | 100 ms | 95 ms | 76 ms | **80%** | **76%** | 直接优化该算子收益极大 |

同样是“占 GPU 时间 80%”，落到真实用户体验上的端到端占比差了近一倍。每次读百分比之前，必须先核对分母是 **GPU 活跃时间**、**CPU 时间**，还是**端到端墙钟时间**。

### 误区二：混淆 `Self CUDA` 与 `CUDA total`

在 `torch.profiler` 打印的算子表里，每个算子都有两列 CUDA 时间：

- **`Self CUDA`**：只算该算子**自己直接发射的底层 kernel** 耗时；
- **`CUDA total`**：包含该算子及其内部调用的所有子算子的嵌套总时间。

把 `CUDA total` 排名前五的行直接相加，父算子（例如 `aten::linear`）会把子算子（`aten::addmm`）的时间重复加一遍，轻轻松松加出超过 200% 的荒唐结果。**找最热底层算子一律按 `Self CUDA` 排序；只有想看某个高层模块（如整层 Attention）的总账时才看 `CUDA total`。**

### 四条必须遵守的采集纪律

1. **先热身，再开录**：冷启动前几轮夹杂着 JIT 编译、cuDNN 算法搜索、显存池扩容与 GPU 睿频爬升。务必跳过预热步（如上文 `schedule(wait=1, warmup=3, active=2)`），只录制稳定态。
2. **带 Profiler 跑出的耗时绝不当基准成绩**：任何 Profiler 都会注入钩子记录起止时间戳；`ncu` 更会反复重放同一个 kernel 数十次以采集不同组硬件计数器，彻底打乱 L2 缓存状态与时钟频率。**定位用带 Profiler 的 Trace，报性能成绩必须彻底关闭 Profiler 用相同负载重跑。**
3. **分清“采样频率”与“采集范围”两个旋钮**：想把探针开销压低，先砍采集范围（`torch.profiler` 关掉 `with_stack` / `profile_memory`，`nsys -t cuda,nvtx` 只开必要事件，`ncu` 用 `--launch-count` 限定少量目标 kernel 而不全局 `--set full`），再调低 CPU 采样频率。
4. **提前核查 `ncu` 性能计数器权限**：`torch.profiler` 和 `nsys` 主要使用底层 **CUPTI** 的活动追踪接口，普通容器通常即开即用；但 `ncu` 需要读取 GPU 硬件性能计数器，很多云主机和容器默认出于多租户安全将其关闭（典型报错 `ERR_NVGPUCTRPERM`）。遇到报错先核对宿主机驱动参数与容器权限，不必怀疑代码本身。

---

## 4.4 从时间线症状到全书优化地图

至此，第一篇的四块拼图全部合拢：**第 1 章 Roofline 定物理下界 → 第 2 章模型账本算清 M/F/R → 第 3 章压测定位外部拐点 → 第 4 章 Profiling 锁定内部病灶**。

每当你拿出一份 Profiling Trace，都可以直接对照下表，把肉眼看到的症状精准映射到全书第二、三、四篇的工程武器库：

| Profiling 时间线典型症状 | 背后物理根因 | 第二篇：引擎与部署优化 | 第三篇：GPU 算子优化 | 第四篇：国产芯片排查 |
|---|---|---|---|---|
| **低算术强度算子占大头**（Decode 线性层/Attention、大量逐元素读写） | 显存带宽受限（$I < I_{\text{ridge}}$） | 连续批处理（第 5 章）、前缀复用（第 6 章）、权重量化与 KV 量化（第 8 章）、投机解码（第 9 章） | Triton 融合（第 13 章）、FlashAttention（第 14 章）、W8A16 量化 Matmul（第 15 章） | 核对 HBM 带宽利用率与 UB 双缓冲流水（第 20/22 章） |
| **高算术强度大块连续顶满 GPU**（长 Prefill GEMM / Attention） | 计算受限（$I > I_{\text{ridge}}$） | 原生 FP8 计算（第 8 章）、PD 解耦隔离长 Prefill（第 11 章） | GEMM 四版分块阶梯（第 12 章）、Triton 2D GEMM（第 13 章）、FA2/FA3 与 CUTLASS（第 16 章） | 检查矩阵核 AIC 利用率与块划分适配（第 20/22 章） |
| **小 kernel 针刺密集、CPU 泳道挤满、GPU 碎片空白** | 启动开销受限（Host 发射跟不上） | CUDA Graph 静态图重放（第 5/10 章）、双进程异步引擎调度（第 5 章） | 算子融合减少启动次数（第 13 章）、自定义算子接入与分发验证（第 17 章） | 排查 ACL Graph / 图下沉是否生效与图外断链（第 19/21 章） |
| **GPU 周期性成片空白，上方为同步、拷贝或 CPU 执行** | 结构性等待（显式同步、H2D/D2H 或算子回退） | 消除同步拷贝与抢占换出（第 5/7 章）、PD 间异步 KV 传输（第 11 章） | 异步拷贝 `cp.async` 与片上流水线（第 16 章） | **重点排查 CPU 回退、图外执行与静默精度降级**（第 21 章） |

!!! tip "收口原则：每次优化只闭环一件事"
    一份合格的性能诊断笔记只需要三行：第一，**最热算子或最长空白占端到端时间的比例**；第二，**它落在三区间中的哪一区、对应哪条根因**；第三，**下一步只改一个变量的具体动作，以及改完后重新压测与重拉 Trace 的验证结果**。

带着这套“先算账、再量数、后切时间线”的工程方法论，接下来我们在 [第 5 章](chapter-05.md) 正式进入第二篇，掀开现代推理引擎的引擎盖，顺着一条请求的完整生命周期，拆解 **vLLM** 如何用 PagedAttention、连续批处理与双进程架构重塑吞吐与延迟边界。
