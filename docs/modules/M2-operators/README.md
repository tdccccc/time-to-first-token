# M2 · 算子优化

> 前置：第一部分（Roofline、profiling、压测脚本）与第二部分第 6–8 章（vLLM 服务已跑起来）。本部分的终点：自己写的 kernel 跑进推理引擎，端到端变快——“会写 kernel”只是起点。
>
> 代码示例分两种：能直接跑的（按官方教程与源码核对过写法）与示意性的（标“示意”，只讲结构、不保证编译通过）。示意代码不要当可运行代码。
>
> 本部分的深度标准：优化做成阶梯，每一步先用 `ncu` 定位瓶颈，再用 Roofline 判断这一步卡在哪；正文数字是算例，实际收益以你在自己机器上的测量为准。6 章分工：第 19–21 章写 kernel，第 22–23 章读工业实现，第 24 章接回引擎。

## 为什么单开一个模块

M1 让你会用引擎、会调参数，但引擎内部的 attention、matmul、量化 kernel 对你还是黑盒。M2 的目标是打开这个盒子：先学会写 kernel，再把 kernel 接回引擎，最后用 M0 的压测脚本量出端到端的变化。现有的练习资源（GPU Puzzles、Triton Puzzles、LeetGPU）都在 kernel 层面收尾，**“接入引擎”这一步目前没有现成课程**，是本模块要补的关键一节。

## 第 19 章 · CUDA 优化阶梯：一个 GEMM，四次提速

一个 4096×4096 的 FP32 矩阵乘法（GEMM，General Matrix Multiplication，通用矩阵乘法：C = A×B 的矩阵乘法）在 A100 上有一对相差很远的数。按 FP32 峰值 19.5 TFLOPS（每秒 19.5 万亿次浮点运算）算，137.4 GFLOP（十亿次浮点运算）只要 7 毫秒；按最朴素的写法，每个输出元素都把 A 的一行和 B 的一列从显存重新读一遍，549.8 GB 的搬运量在 2 TB/s 带宽下要 274 毫秒。39 倍的差距不在乘加次数上——无论怎么优化，137.4 GFLOP 都不会少——而在数据怎么搬、指令怎么发、延迟怎么藏。本章把同一个 GEMM 写四版：朴素版、shared memory 分块版、向量化访存版、占用率调优版。每版只比上一版多改一个变量；先用 ncu 读出瓶颈，再改，再测，最后把点标回 Roofline。

**本章目标。** 写出四个版本的 GEMM，用 GPU 事件测出各自的耗时与实测算力，用 ncu 的 SM%、DRAM% 与占用率说明每一版卡在哪，把四个点标在同一张 Roofline 图上，并填出一张每一步都有依据的阶梯表；测不到的格子写“待补测”，不写推测值。

数字分两类。按 A100 80GB 公开规格（FP32 19.5 TFLOPS、显存带宽约 2 TB/s）算出来的是预期值，正文标“算例”；你在自己机器上测出来的是实测，填进表里。换一张卡，预期值要重算——40GB A100 的带宽约 1.6 TB/s，H100 的算力与带宽又是另一组数——方法不变。本章 kernel 用 C++ 写法给出，用来说明结构与每一步改哪一行，没有在本书环境编译验证；启动配置与编译选项（例如 `-arch=sm_80`）依赖你的卡，标“示意”的片段只保留关键行。

### 第 1 节 · GPU 执行模型与第一个 kernel

先算清楚这份活有多大。输出 4096×4096 = 16,777,216 个元素，每个元素做 4096 次乘加，总共 4096³ = 68,719,476,736 次乘加，约 687 亿次；一次乘加算 2 次浮点操作，合计 137.4 GFLOP。一个 CPU 核心按每秒 20 亿次浮点运算算，要 69 秒。GPU 的做法是把这 687 亿次分给上万个执行单元同时算；分给谁、怎么分、算出来的数放哪，决定了 kernel 的快慢。

GPU 上最小的执行单位是线程（thread）。线程按三层组织：

| 层 | 是什么 | 共享什么 |
|---|---|---|
| thread | 最小执行单位 | 私有寄存器 |
| block | 一组线程，常见 128 或 256 个，最多 1024 个 | 一块 shared memory，block 内可同步 |
| grid | 所有 block 的集合 | 只通过 global memory 通信 |

分三层的原因是同步有成本。同一 block 的线程可以互相等待、共享一块片上存储；跨 block 的线程只能走 global memory。写 kernel 时每做一个“让线程互相配合”的决定，先问这配合发生在 block 内还是跨 block：前者便宜，后者昂贵；判断错了，后面的优化会落在错误的资源上。

线程是编程模型里的单位，硬件实际调度的是 warp（线程束：32 个线程，在同一批指令上锁步执行）。一个 warp 的 32 个线程执行同一条指令、带不同的数据，一套取指与解码逻辑供 32 份数据用。代价在同一 warp 内不整齐的时候：线程走不同分支（分化，divergence），两条路只能串行，走一条时另一条空等。19.4 的 bank conflict 是另一种 warp 内不整齐，代价同样按串行长度算。

一个 block 整体分到一个 SM（Streaming Multiprocessor，流式多处理器：GPU 上独立的计算与调度单元）上；SM 里的 warp 调度器每次从就绪的 warp 里挑一个发指令。block 是调度与共享的边界：同一 block 在同一个 SM 上，才能共享 shared memory 与同步原语；跨 block 只能走 global memory。block 常见 128 或 256 个线程，即 4 或 8 个 warp（128 ÷ 32 = 4）；硬件上限 1024 个线程，即 32 个 warp。这个换算在 19.5 算占用率时要用：驻留的 warp 是调度器手里的备选，备选太少，一个 warp 等访存时就没有别的可发。

每个线程靠三个内置变量定位：`blockIdx`（第几个 block）、`threadIdx`（block 里的第几个线程）、`blockDim`（每个 block 多少线程）。一维下标是：

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

验算一次：16 个元素，4 个 block × 4 个线程，block 2 的 thread 1 算元素 2×4+1 = 9。再验算带余数的：向量长 15、启动 16 个线程，多出来的 1 个线程没有元素，空闲比例 1 ÷ 16 = 6.25%；这个线程必须被边界判断挡住，否则会越界写。

二维下标每个维度各算一次：`row = blockIdx.y * blockDim.y + threadIdx.y`，`col = blockIdx.x * blockDim.x + threadIdx.x`。拿 16×16 的 block 验算：block (2, 1) 里的线程 (3, 5)，行号 1×16 + 5 = 21，列号 2×16 + 3 = 35。GEMM 的规模也这样定：输出 16,777,216 个元素、一个线程算一个，就要 16,777,216 个线程；block 取 16×16 = 256 个线程，grid 每维 4096 ÷ 16 = 256 个 block，256 × 256 = 65,536 个 block，65,536 × 256 = 16,777,216，正好覆盖输出。grid 与 block 的乘积要自己验算一遍，差一个 block 就漏算一片输出。

数据放在哪一层，和怎么分同样重要：

| 层 | 大小 | 延迟 | 谁可见 |
|---|---|---|---|
| global memory | 几 GB 到几十 GB | 几百个周期 | 所有线程 |
| shared memory | 每个 block 几十 KB | 几十个周期 | 同一 block 内 |
| register | 每个线程几十个 | 最快 | 仅本线程 |

存储做不到又大又快，分层就是把常用的放得近、少用的放得远；代价是程序员要自己决定数据放哪一层。GEMM 里 A 的每一行有 4096 个元素，每个元素都会被同一行的 4096 个输出元素各读一次，每次都发一条 global load；重复读把搬运量放大，19.3 的分块就是把被复用的那一块搬进 shared memory。

第一个 kernel 是向量加法。CUDA（Compute Unified Device Architecture，统一计算设备架构：NVIDIA 的 GPU 并行计算平台与编程模型）的 kernel 用 `__global__` 标记，从 CPU 启动，在 GPU 上由线程并行执行：

```cpp
__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

函数体里没有循环，并行体现在线程组织里：CPU 上“算 16 个数”要写一个 16 次的循环，GPU 上启动 16 个线程，每个线程算一个数，循环消失。读一个 kernel，先找并行度藏在哪，答案通常在 grid 与 block 的配置里。`if (i < n)` 是 15 个元素那笔账的守卫：多出来的线程下标越界，必须直接返回；越界写会踩到别的数组甚至别的 kernel 的数据，错得无声，分支只是让那个线程少干活。

边界上的 `if` 会让落在边界上的 warp 走两条路：一部分线程进分支、一部分跳过，两条路串行。只有边缘的 warp 受影响，代价是几条指令。真正贵的分化是同一 warp 长期走不同分支，代价按受影响的 warp 数与路长算。

怎么知道算对了？用 CPU 参考实现逐项对照：向量加法是一次 IEEE 加法，GPU 与 CPU 对同样的两个数做同一次加法，结果应当逐元素相等，差一个也不算通过。以后每个 kernel 都配一个这样的参考实现：GPU kernel 的错常是“算得很快，结果不对”，速度数字会骗人，逐项对照不会。参考实现用最直白的写法，不求快，只求对；19.3 改分块、19.4 改向量化之后，第一件事都是重新跑它。

GEMM 的对照要松一档：CPU 参考实现与 GPU 的累加顺序、FMA（fused multiply-add，乘加融合：一次舍入完成 a×b+c）可能不同，K = 4096 的累加会出现末位差异。先在小尺寸（比如 512×512）上逐元素比，再在 4096 上抽几行几列比，阈值取相对误差 1e-4 量级，先抓漏算、错块这类结构错误；要求逐 bit 相等，只会把正确实现判错。

做 GPU Puzzles 的 Puzzle 1（Map）、Puzzle 2（Zip）、Puzzle 3（Guards）：一个线程负责一个元素，线程数多于元素数时加边界判断，验收都是 `problem.check()`。这三题练的是本节的两件事：下标公式与边界判断。环境上，Numba 内置的 CUDA 目标已弃用，后续开发移到 NVIDIA 的 `numba-cuda` 包；以 GPU Puzzles 仓库自己锁定的 numba 版本与安装说明为准，不要自行升级——升级会把一个验证过的组合换成一个没人验证过的组合。

向量加法太小，量不出阶梯：它只搬三个 16 KB 量级的向量，时间在微秒级，启动开销占大头。本章的主对象是 GEMM；要比较四个版本，先得有一把能重复使用的尺子。

### 第 2 节 · 搭测量台：计时、Roofline、第一次 ncu 读数

优化之前先把尺子做好。没有尺子，“快了 3 倍”没法验证，也判断不了下一步该改哪里。尺子有三件：计时、上限、定位。计时用 GPU 事件，取 kernel 在 GPU 上真正跑的时间；上限用 Roofline 算，先写出这一版最多能有多快；定位用 ncu，读出瓶颈落在算力、带宽还是延迟上。这一章要量四个版本：朴素版、分块版、向量化版、占用率调优版。每次只改一个变量，让对应的那个数动：分块看 DRAM%，向量化看 load 指令数，占用率看活跃 warp 数。图 19-1 是这四级台阶的示意，柱高只表示方向，不是实测值。

#### 待测对象：朴素版

四个版本都从这一版出发：一个线程算一个输出元素，循环 K 次，每次从 global memory 读 A、B 各一个 float。

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// 启动：block = (16, 16)，grid = (N / 16, M / 16)
```

读这段先看循环体：每个线程算一个输出，循环跑 K = 4096 次，每次两条 load、一次乘加。一个线程发 2 × 4096 = 8192 条 load；16,777,216 个线程合计 137,438,953,472 条，约 1374 亿条。每条 load 搬 4 字节，就是 549.8 GB——和 19.2 后面先算的那笔搬运量是同一笔账的两种写法。`acc` 在寄存器里累加，只有最后写回一次；朴素版做对的一件事是输出只写一次，重复的全是读。

#### 计时：为什么用 GPU 事件，不用 CPU 时钟

CPU 时钟量到的是 host 端的时间：调用开销、kernel 启动、同步等待，全算在内。CUDA Events（CUDA 事件：GPU stream 上的时间戳）量的是 kernel 在 GPU 上真正跑的时间。一次 kernel 启动的延迟在微秒量级：对 274 ms 的朴素版是零头，对 1 ms 级的优化版就是千分之几；更麻烦的是同步——CPU 计时通常要在 kernel 后等一次设备同步，调度抖动与等待都会算进去。被测时间越短，host 端的杂项占比越高，越该用 GPU 端的时间戳。

计时写法固定成三段：热身、连续启动、同步后取平均。

```cpp
// 示意：同一 stream 上连续启动 N 次，用事件量总时间
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

for (int i = 0; i < 5; ++i)                       // 热身：模块加载、缓存、时钟都稳定下来
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
cudaGetLastError();                               // 启动配置错了，这里就能看到

cudaEventRecord(start);
for (int i = 0; i < 20; ++i)
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
cudaEventRecord(stop);
cudaEventSynchronize(stop);                       // 必须等 stop 完成
float ms_total = 0.f;
cudaEventElapsedTime(&ms_total, start, stop);
float ms = ms_total / 20.f;
```

`cudaEventRecord` 只是在 stream 里插一个时间戳，不阻塞；kernel 可能还在跑。`cudaEventElapsedTime` 要求 stop 已经完成，否则返回 `cudaErrorNotReady`，`ms_total` 不被更新——不检查返回值，就会拿初值 0 或上一次的数当结果，快得像假的一样。`cudaEventSynchronize(stop)` 等 stop 完成再读，这一步不能省。事件与 kernel 必须在同一个 stream 上，时间戳的先后才有保证；20 次连续启动取平均，事件本身的开销被摊薄，也把每次启动之间的间隙平均进去。

第一次跑先看 `cudaGetLastError()`：启动配置写错（block 超过 1024 个线程、shared 超过上限）不会抛异常，只会让后面的耗时变成 0。热身 5 次是让 GPU 时钟从 idle 拉起来、让指令缓存与 L2 进入稳定状态；前几次的数不报。GEMM 的分母固定：137.4 GFLOP ÷ 实测秒数 = 实测算力，四个版本用同一个分母，算力才能横向比较。

#### 先算：Roofline 预期是一道算术题

M = N = K = 4096，FP32。计算量：2 × 4096³ = 137,438,953,472 FLOP ≈ 137.4 GFLOP。这个数不随优化改变，是全章的分母。

搬运量（朴素版）：每个输出元素读 A 的一行 4096 个 float 和 B 的一列 4096 个 float，8192 × 4 B = 32 KB；16,777,216 个元素 × 32 KB = 549,755,813,888 B ≈ 549.8 GB。这份账按 global load 计，没有扣掉 L2 缓存命中：A 的同一行、B 的同一列会被重复读，其中一部分可能留在 L2（GPU 芯片上的二级缓存，容量比 shared memory 大、比显存小）里。L2 接住得越多，真正走 HBM 的字节越少，后面的预期值就越偏保守。

算术强度 = 137.4 GFLOP ÷ 549.8 GB = 0.25 FLOP/字节。每搬 1 字节只做 0.25 次浮点运算，落在带宽那一侧。

A100 80GB 的公开规格：FP32 19.5 TFLOPS，HBM（High Bandwidth Memory，高带宽显存：GPU 的主存，A100 80GB 用 HBM2e）带宽约 2 TB/s。脊点 = 19,500 ÷ 2,000 = 9.75 FLOP/字节；0.25 远低于脊点，所以上界由带宽定：0.25 × 2,000 GB/s = 500 GFLOP/s（0.5 TFLOPS）。按这个速度，137.4 GFLOP 要 274 ms。这是算例，不是实测；实测比它慢是常态。

实测与预期对不上时，先查三件事：计时有没有漏同步；编译器有没有把循环优化掉（结果没被使用）；L2 是否吃掉了重复读。第三条会让实测超过这个预期，说明真实 HBM 搬运量比 549.8 GB 小，有效算术强度比 0.25 高，不是测量错了。举个具体的对照：548 ms 对应 251 GFLOP/s，是 500 GFLOP/s 的一半，说明带宽只用了一半；100 ms 对应 1374 GFLOP/s，是预期的 2.7 倍，带宽给不出这个速度，要么计时错了，要么 L2 吃掉了大部分重复读。

#### 定位：ncu 的两个数

ncu 回答的是“这个 kernel 为什么慢”，不回答“哪个 kernel 慢”；后者是第 3 章 `torch.profiler` 的活。先看 SpeedOfLight 一节的 Compute (SM) 与 Memory 两个百分比：前者是 SM 吞吐占峰值的比例（下称 SM%）；后者取 L1、L2、DRAM 里最紧的一项。显存这一侧要单独确认，用 `dram__throughput` 取 DRAM%（显存带宽占峰值的比例）。三个指标一条命令就能取到：

```bash
ncu --section SpeedOfLight --section Occupancy --kernel-name regex:gemm ./gemm_bench

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name regex:gemm ./gemm_bench
```

三个区间对照着看：

- SM% 高（>60）、DRAM% 低（<40）→ 计算受限：SM 一直在算，数据供得上。
- SM% 低、DRAM% 高（>70）→ 带宽受限：数据通道满了，SM 在等数据。
- 两个都低（<40）→ 延迟受限：SM 既没算也没搬，在等依赖、同步或驻留 warp 不够。

只看一个数会误判：DRAM% 高可能是带宽真跑满，也可能是 kernel 偶尔访存一次、每次都把带宽打满；SM% 低可能是等数据，也可能是 warp 不够。两个数拼在一起才有诊断价值。60、40、70 是经验分界，读数看的是哪边明显高、哪边明显低。

朴素版的预期是 DRAM% 高、SM% 低。分块之后 DRAM% 应明显下降；没降说明分块没生效，先查 tile 索引、同步与 grid 覆盖，不要接着调别的旋钮。每次改动都要有它负责的那个数：分块负责 DRAM%，向量化负责 load 指令数（MemoryWorkloadAnalysis 里的 global load 指令与请求数），占用率负责 `sm__warps_active`。数没动，优化就没命中瓶颈。

ncu 的操作细节：

- `--launch-skip 5 --launch-count 3` 跳过热身：模块加载、可能的 JIT（即时编译）编译、冷缓存、GPU 时钟爬升只污染前几次。
- ncu 会读硬件计数器并重放 kernel，报告里的耗时不能当性能数字；性能数字用 Events 测，ncu 只用来定位（第 3 章讲过同一条口径）。
- 如果你先用第 3 章的 `torch.profiler` 给四个版本排名，按 `Self CUDA` 排，不要按 `CUDA total`：后者把子算子的时间重复计进去。
- ncu 需要访问 GPU 性能计数器的权限。Linux 上普通用户默认没有；容器与云主机常默认禁用（驱动参数限制或容器权限不足），典型报错是 `ERR_NVGPUCTRPERM`。解决要管理员在驱动层放开计数器访问，具体做法随驱动版本与发行版不同，按 Nsight Compute 文档的权限说明操作；容器里还要确认容器被授予了对应能力。跑不起来先查权限与容器配置，再查命令，最后才怀疑工具。

每版的记录固定为五项：实测耗时、实测算力、占本行 Roofline 上界的比例、ncu 的 SM%/DRAM%、一句话瓶颈判断。五项齐了，才进入下一节。

### 第 3 节 · 第一步：Shared memory 分块

19.2 的读数是 DRAM% 高、SM% 低。原因不在算，在重复读：A 的同一个元素被 N = 4096 个输出元素各读一次，B 的同一个元素被 M = 4096 个输出元素各读一次。一个元素只参与 4096 次乘加，却被从显存读了 4096 次；理想情况是读一次、在片上复用 4096 次。分块要追的就是这 4096 倍里的大部分。

把 A、B 各切 32×32 的小块（tile），搬进 shared memory，在片上算部分和。一个 tile 里的元素被同一 block 的 32 个输出元素复用，global memory 的读取次数除以 32：搬运量 = 2 × 4096² × (4096 ÷ 32) × 4 B = 17,179,869,184 B ≈ 17.18 GB，是 549.8 GB 的 1/32。计算量不变，仍是 137.4 GFLOP。

从 block 一级算上去，结果一样：每个 block 每次迭代搬两块 32×32×4 B = 8 KB，128 次迭代搬 1 MB；grid 有 (4096 ÷ 32)² = 128² = 16,384 个 block，16,384 × 1 MB = 17.18 GB。按矩阵分：A 的每个元素被读 N ÷ 32 = 128 次，16,777,216 × 128 × 4 B = 8.59 GB，B 同样 8.59 GB，合计 17.18 GB。

解法是分块（tiling）：把 A 的一块与 B 的一块先搬进 shared memory，在片上算部分和，再搬下一块：

```cpp
#define BLOCK 32
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;
    float acc = 0.f;
    for (int t = 0; t < (K + BLOCK - 1) / BLOCK; ++t) {
        int a_col = t * BLOCK + tx;
        int b_row = t * BLOCK + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < BLOCK; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// 启动：block = (32, 32)，共 1024 个线程
dim3 block(32, 32);
dim3 grid((N + 31) / 32, (M + 31) / 32);   // 不能整除时也要覆盖全部输出
gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

先把循环结构读清楚。外层沿 K 走，一步走 BLOCK = 32 列；K = 4096，除 32 得 128，外层跑 128 次。每次迭代三步：搬、算、等。搬是两块 32×32 的 float 进 shared memory；算是内层 32 次乘加，读的全是 shared memory；等是两次 `__syncthreads()`，保证搬完再算、算完再搬。一个线程在整个 kernel 里做 128 × 32 = 4096 次乘加，正好等于 K——分块改的是数据从哪来，不改每个线程的算术总量。算术总量必须还是 4096 次乘加，多一次少一次都是错的。

搬运量降了，为什么快？shared memory 在片上：从 global memory 读一次要几百个周期，从 shared 读只要几十个。分块没有减少乘加，减少的是等数据的时间；算术强度从 0.25 涨到 8，涨的正是“每次搬运能供多少次乘加”这个比值。

先算，再测。算术强度 = 137.4 GFLOP ÷ 17.18 GB ≈ 8 FLOP/字节，也等于 BLOCK ÷ 4 = 32 ÷ 4：一个 float 占 4 字节，BLOCK 个元素的复用摊到每个字节上，就是 BLOCK/4。上界 = min(8 × 2,000, 19,500) = 16,000 GFLOP/s = 16 TFLOPS，预期至少 137.4 ÷ 16,000 秒 = 8.59 ms，比朴素版的 274 ms 快 32 倍。分块版的算术强度 8 仍低于脊点 9.75，所以这一版还在带宽斜线上；要让 AI 越过脊点，BLOCK 至少 39，取 2 的幂就是 64——那是 19.5 的取舍。

这里写的“至少 8.59 ms”是按 global load 字节数算的预期：L2 若接住一部分重复读，真实 HBM 搬运量会更小，实测可能低于 8.59 ms。读到比 8.59 ms 还快的数，先确认计时与 L2 命中，再下结论。

每个 block 用 8 KB shared（两块 32×32×4 B）。A100 每个 SM 有 164 KB shared，光按 shared 算能放 164 ÷ 8 = 20.5，向下取整 20 个 block。这只是 shared 一项的上限：这个 kernel 的 block 是 1024 个线程，A100 每 SM 最多 2048 个线程，线程数一项只允许 2 个 block；寄存器够不够还要看编译器给的数。20 个 block 在这里用不上，19.5 会按线程、寄存器与 shared 三项资源取最小，block 数还有一道单独的上限。

两个 `__syncthreads()` 缺一不可。第一个在搬完、开算之前：没有它，跑得快的线程会读还没写完的 tile。第二个在算完、搬下一块之前：没有它，跑得快的线程会覆写还在被读的 tile。缺一个的错是间歇性的：线程快慢取决于调度与访存延迟，某次跑对了不代表下次也对。每个 block 要执行 128 × 2 = 256 次同步。

shared memory 的访问顺序也有讲究：`As[ty][k]` 在一个 warp 内是同一个地址（这个 warp 的 ty 相同、tx 从 0 到 31），硬件一次广播满足；`Bs[k][tx]` 的地址随 tx 连续变化，32 个线程落在 32 个不同 bank 上。广播与并行都不冲突；把 B 的 tile 转置存放，就会撞 bank——19.4 用一个转置布局把冲突造出来，再用 padding 修掉。

跑计时 + ncu，DRAM% 应明显下降、SM% 上升。把两个点标到同一张 Roofline 图上：朴素版在左下（AI 0.25、算力低于 500 GFLOP/s），分块版向右（AI 8）、向上（算力向 16 TFLOPS 靠近）。如果只向右不向上，说明搬运省了、执行没跟上，19.4 与 19.5 处理；如果 DRAM% 没降，先查分块是否真的生效。

### 第 4 节 · 第二步：向量化访存与 bank conflict

分块之后，瓶颈换了一个形态。搬运总量降到 17.18 GB，HBM 这条路不那么堵了；但每个线程每次还是只读 4 字节，数据总量小了，取数据的动作没少。看 ncu 的 MemoryWorkloadAnalysis：如果 global load 指令数与 L1 吞吐仍然很高，而 DRAM% 已经下来，问题就在指令数，不在带宽。

SM 每个周期能发的指令数有限。一条 4 字节 load 和一条 16 字节 load 各占一个发射槽；同样搬 16 字节，标量要 4 条指令，`float4` 只要 1 条。指令数降到 1/4，省出来的是发射槽和等待次数：每条 load 发出后都要等数据回来，4 条是 4 次等待，1 条是 1 次。这个瓶颈在 DRAM% 上看不出来，要靠 MemoryWorkloadAnalysis 的指令与请求计数。

先分清两件事。分块版的标量 load 已经是合并的：一个 warp 的 32 个线程读 32 个连续 float，正好 128 字节，硬件一次取回。向量化改的不是合并，是每条指令搬多少字节。合并解决“一次取多少连续字节”，向量化解决“一条指令取多少字节”；前者已经满足，后者还能改 4 倍。

指令数算一遍。一个 32×32 的 tile 有 1024 个 float：标量读法 1024 条 load，`float4` 读法 256 条。一个 block 每次迭代搬 A、B 两块，128 次迭代：标量 2 × 128 × 1024 = 262,144 条，`float4` 2 × 128 × 256 = 65,536 条，省 196,608 条。16,384 个 block 合计约 32.2 亿条 load 指令。比例是 1/4，这个比例比总数有用。

改法是让每个线程一次读 4 个 float。1024 个线程的 block 里，只用前 256 个做搬运：

```cpp
// 示意：只保留搬运那几行。1024 个线程里前 256 个各搬一个 float4。
// kt 是外层 K 方向的迭代下标。
int t = ty * 32 + tx;                 // 0..1023
if (t < 256) {
    int r  = t / 8;                   // tile 内第几行：每行 32 个 float = 8 个 float4
    int c4 = (t % 8) * 4;             // 行内起始列，必为 4 的倍数
    const float4* pa = reinterpret_cast<const float4*>(
        &A[(blockIdx.y * 32 + r) * K + kt * 32 + c4]);
    float4 va = *pa;
    As[r][c4 + 0] = va.x; As[r][c4 + 1] = va.y;
    As[r][c4 + 2] = va.z; As[r][c4 + 3] = va.w;
    // B 的 tile 同样处理：行号是 kt*32 + r，列号是 blockIdx.x*32 + c4
}
__syncthreads();
```

`float4` 要求 16 字节对齐：`cudaMalloc` 返回的指针满足；行内偏移 `c4` 是 4 的倍数，K = 4096 也是 4 的倍数时，`(blockIdx.y * 32 + r) * K + kt * 32 + c4` 一定是 4 的倍数，乘 4 字节就是 16 的倍数。`As` 的写入用四个标量，避开 shared 数组的对齐假设。K 或 N 不是 4 的倍数时行首会错位，只能退回标量或按行单独处理——这是向量化的边界。

`__syncthreads()` 不能写进 `if (t < 256)`：同一 block 的所有线程必须到达同一个同步点，分支里的同步是未定义行为。

向量化不改搬运量（17.18 GB）、不改算术强度（8）、不改计算量（137.4 GFLOP），改的是 global load 指令数。测的时候看 MemoryWorkloadAnalysis 的 load 指令数与 L1 吞吐；耗时不动，说明瓶颈不在指令数，别继续加向量宽度。

接着看 shared memory 一侧。shared memory 分成 32 个 bank，每个 4 字节宽；一个 warp 的 32 个线程如果落在 32 个不同 bank，一次读完；落在同一个 bank 的不同地址，只能串行。以 float 为单位，地址 a 落在 bank a % 32。

分块版的计算循环读 `As[ty][k]` 与 `Bs[k][tx]`：

- `As[ty][k]`：warp 内 ty 相同、tx 从 0 到 31，地址与 tx 无关，32 个线程读同一个地址，硬件广播，一次完成。
- `Bs[k][tx]`：地址是 k × 32 + tx，bank = tx，32 个线程落在 32 个不同 bank，一次并行完成。

把 B 的 tile 转置存放，再算一遍：写入改成 `BsT[tx][ty] = B[(kt*32 + ty) * N + col]`，计算时读 `BsT[tx][k]`。数值不变——存的还是同一块 B，只是行列互换——但读地址变成 tx × 32 + k，bank = k，一个 warp 的 32 个线程全撞在同一个 bank 上，32 路串行。写的时候地址是 tx × 32 + ty，bank = ty，32 个线程也全在同一个 bank。读写两头都串行，这段 shared 访问的时间被拉长。

修法是 padding（补一列）：把 `BsT` 声明成 `__shared__ float BsT[32][33]`。地址变成 tx × 33 + k，bank = (tx × 33 + k) % 32 = (tx + k) % 32；固定 k 时 32 个线程的 bank 各不相同，冲突消失。写的时候 (tx × 33 + ty) % 32 = (tx + ty) % 32，也不再撞。多出来的那一列不存数据，只用来看地址。

用 ncu 的 `--section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables` 看 Shared Memory 一节的 bank conflict 计数：冲突版会明显上升，padding 版回落。计数与耗时一起看：只看耗时，上升可能是别的原因；只看计数，计数上升未必拖慢整体。转置版与 padding 版只差一列 padding，因果才钉得住。

global 一侧看合并与指令数，shared 一侧看 bank。两者都对的布局，`Bs[k][tx]` 与 padding 后的转置版都满足。下一章用 Triton 写同样的分块，这两个旋钮被编译器接走；要精确控制布局与 padding，还得回到 CUDA，第 23 章的 CUTLASS 是工业级的做法。

### 第 5 节 · 第三步：Occupancy 调优 + 阶梯总表

占用率是一个 SM 上同时驻留的 warp 数，占它最多能驻留的 warp 数的比例。warp 等访存时被挂起，调度器切到另一个就绪的 warp 继续发指令；驻留的 warp 越多，越容易找到能发的。占用率低不一定慢，但低到没有备选时，一个 warp 停，SM 就空着。

ncu 的 Occupancy 一节给两个数：理论占用率（由寄存器、shared memory、block 大小算出的上限）与实测占用率（`sm__warps_active.avg.pct_of_peak_sustained_active`）。理论低，问题在资源用量；理论高、实测低，问题在延迟隐藏不足：等访存时没有别的 warp 可切，或者同步太频繁。先看理论，再看实测，两个数一减，问题定位在哪一层就清楚了。

以 19.3 的 kernel 为例算占用率的上限。block 1024 个线程（32 个 warp），每 block 8 KB shared。A100 每个 SM：

- 线程：2048 ÷ 1024 = 2 个 block。
- shared：164 ÷ 8 = 20.5，向下取整 20 个 block。
- 寄存器：65,536 ÷ (1024 × 32) = 2 个 block（按 32 个寄存器/线程算）；编译器给 40 个，就只剩 1 个。
- block 数上限 32 个，不构成限制。

四项取最小是 2 个 block，也就是 64 个 warp，理论占用率 100%——前提是寄存器不超过 32 个/线程。shared 允许的 20 个 block 在这里完全用不上；把“164 ÷ 8 = 20 个 block”当成占用率上限，是这类 kernel 最常见的算错。最终占用率取线程、寄存器、shared、block 数四项限制里最小的那个，ncu 的 Occupancy 一节会直接给出是哪一项。

1024 线程的 block 粒度太粗：寄存器超一点，整块从 2 个掉到 1 个，占用率从 100% 掉到 50%。常见改法是线程粗化：block 改成 16×16 = 256 个线程，每个线程算 2×2 = 4 个输出，tile 仍是 32×32、shared 仍是 8 KB。线程上限变成 2048 ÷ 256 = 8 个 block；寄存器若给 40 个/线程，65,536 ÷ (256 × 40) = 6.4，取 6 个 block（1536 个线程，75%）。`__launch_bounds__(256, 8)` 让编译器把寄存器压到 32 个以内，换 8 个 block（100%），代价可能是 spill。

```cpp
// 示意：tile 仍是 32×32，block 改成 16×16，每个线程算 2×2 个输出
// gemm_tiled_2x2 指线程粗化后的 kernel
dim3 block(16, 16);          // 256 个线程
dim3 grid(N / 32, M / 32);   // grid 不变

int blocks = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm_tiled_2x2, 256, 0);
```

`__launch_bounds__` 是给编译器的承诺：这个 kernel 每个 block 最多 256 个线程、每个 SM 至少驻留 8 个 block。编译器据此把寄存器压到 32 个以内；压不下来就把变量 spill 到 local memory（物理位置在显存里，走 L1/L2），读一次要几十到几百个周期。占用率涨了、耗时也涨了，这次调整就是失败的，要改回去。调优的目标是耗时下降，占用率只是中间指标。

别为了占用率缩小 tile。算术强度 = BLOCK ÷ 4：BLOCK 从 32 缩到 16，AI 从 8 掉到 4，上界从 16 TFLOPS 掉到 8 TFLOPS，预期时间从 8.6 ms 涨到 17.2 ms；占用率涨了，上界掉了。BLOCK 加到 64，AI 到 16，越过脊点 9.75，上界变成算力平线 19.5 TFLOPS；但每 block shared 变成 64×64×4×2 = 32 KB，光 shared 一项只放 164 ÷ 32 = 5 个 block，每个线程要算的输出数也跟着涨，寄存器压力上升。哪一项先撞墙，用 occupancy API 与 ncu 看，不要按单项资源下结论。

阶梯总表是本章的交付物。`算术强度`与`上界`按正文算例填好；`实测`各列留给你，没测就写“待补测”。不要把别人的数抄进来，也不要按公式反推耗时冒充实测。

| 版本 | 算术强度 (FLOP/B) | Roofline 上界（算例） | 实测耗时 | 实测算力 | 占上界 | ncu SM% / DRAM% | 瓶颈判断 |
|---|---|---|---|---|---|---|---|
| 朴素版 | 0.25 | 500 GFLOP/s（274 ms） | 待补测 | 待补测 | 待补测 | 待补测 / 待补测 | 待补测 |
| 分块版 | 8 | 16 TFLOPS（8.6 ms） | 待补测 | 待补测 | 待补测 | 待补测 / 待补测 | 待补测 |
| 向量化版 | 8 | 16 TFLOPS（8.6 ms） | 待补测 | 待补测 | 待补测 | 待补测 / 待补测 | 待补测 |
| 占用率调优版 | 8 | 16 TFLOPS（8.6 ms） | 待补测 | 待补测 | 待补测 | 待补测 / 待补测 | 待补测 |

表怎么读。先看算术强度：只有分块改了它（0.25 → 8），向量化与占用率都没改——这两步不改变搬运量与计算量，算术强度自然还是 8。再看上界：每行的上界不同，占上界那一列要按本行的上界算，不能跨行比百分比；跨行比的是耗时与实测算力。ncu 两列是瓶颈判断的证据，判断与证据对不上，这一行就没写完。

哪一行的数没动，说明那一步没命中瓶颈。分块后 DRAM% 没降，先查 tile 索引、同步与 grid 覆盖；向量化后 load 指令数没降，先查 `float4` 是否真的生成（对齐、`reinterpret_cast`、K 的倍数）；调占用率后活跃 warp 没涨，先看是哪项资源卡住、有没有 spill。代码查过再继续加旋钮；顺序反了，后面的数会互相污染。

本章的检查清单，四条：

1. 每个版本先与 CPU 参考实现对过，再计时；向量加法要求逐元素相等，GEMM 用相对误差 1e-4 量级。
2. 每次只改一个变量；改完回到 ncu，看这个变量负责的计数动了没有。
3. 计时用 CUDA Events，`cudaEventSynchronize` 之后才读；ncu 的耗时不当性能数字。
4. 表里没测的格子写“待补测”；四个版本各用一句话说清瓶颈，且这句话要有 ncu 读数撑着。

四个版本量的是三件不同的事：分块省重复搬运（549.8 GB → 17.18 GB），向量化省访存指令（load 指令降到 1/4），占用率省延迟暴露（让 SM 在等数据时有别的 warp 可发）。三者都不改 137.4 GFLOP 的计算量。Roofline 说每一版的上界在哪，ncu 说实测离上界差在哪个资源；把这两个数放进同一行，优化才不是碰运气。下一章用 Triton 重写同一套分块，看哪些旋钮被编译器接走、哪些还得自己管。

**延伸**：[GPU Puzzles](https://github.com/srush/GPU-Puzzles)、[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)、[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)、[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（从朴素版一步步做到接近 cuBLAS 的完整账本）、[Stanford CS336](https://www.youtube.com/watch?v=6OBtO9niT00)（GPU 与显存层级一讲）。

## 第 20 章 · Triton 入门：用 Python 写 kernel

第 19 章的 GEMM 阶梯里，每改一次分块大小，线程下标、shared memory 声明、边界判断与同步都要跟着改一遍。想验证一个更小的想法——把 `add` 与 `relu` 融成一个 kernel 能不能省掉一半访存——也要先铺满这套模板代码，才能上卡量。时间花在组织线程上，不在算法上。

Triton 换掉的就是这一层：你写一个程序处理一整块数据，线程怎么分、shared memory 怎么搬、加载要不要向量化，交给编译器。代码短了，控制粒度也粗了——线程映射、shared memory 布局、warp 级同步不再由你写。这一章把“粗到什么程度”量出来：先写向量加法与三类谜题，再写 fused softmax，然后把每个结构翻译回 CUDA，最后收成一张判断表。

**本章目标。** 能写出 block 粒度的 Triton kernel（`program_id`、`arange`、mask、`tl.load/store`、`tl.dot`），能算出融合省下多少访存，能把一个优化需求判到 Triton 或 CUDA/CUTLASS 那一边。

20.1 讲语法与 block 编程模型，并算第一笔访存账；20.2 用 Triton Puzzles 的三类题（逐元素与广播、归约、矩阵乘）读出访存图；20.3 写 fused softmax，把数值稳定性、`do_bench` 的计时口径与 autotune 一起写清；20.4 把这一章写过的结构翻译回 CUDA，标出编译器替你做了哪些决定；20.5 给出取舍表与检查清单。20.1 与 20.2 能在解释器里做完，不需要 GPU；20.3 的计时必须上卡。

### 第 1 节 · Triton 语法与 block 编程模型

eager 模式的代价先算成一笔账。取 N = 10⁶ 个 fp32 元素，一个张量占 10⁶ × 4 字节 = 4 MB。写 `y = relu(x + 1)`：eager 模式先算加法，读 4 MB、写 4 MB；再算 relu，读 4 MB、写 4 MB，一共 16 MB。融合成一个 kernel 之后只读 4 MB、写 4 MB，一共 8 MB，访存减半。加法和 relu 各做 10⁶ 次，一次没少——省掉的是中间结果在显存上的一趟往返，外加一次 kernel 启动。8 MB 是下限，不含中间张量的分配与启动收尾的开销；真实差距只会比一倍大，不会更小。

Triton 的核心是按 block 编程：一个程序处理一整块数据，对应 CUDA 里的一个线程块（block）。`tl.program_id(0)` 告诉你现在是第几个程序，相当于 `blockIdx.x`；`tl.arange(0, BLOCK)` 生成一块连续下标，相当于把 `threadIdx` 变成向量。CUDA 里每个线程用 `blockIdx.x * blockDim.x + threadIdx.x` 算出一个标量下标；Triton 里 `pid * BLOCK + tl.arange(0, BLOCK)` 是一整块向量下标，这 BLOCK 个元素怎么分给 32 × num_warps 个线程，由编译器决定。

拿一组数把分工落到实处。向量长 n = 10000、BLOCK = 1024：程序数是 ⌈10000 / 1024⌉ = 10，0–8 号程序各处理 1024 个元素，9 号程序只剩 10000 − 9 × 1024 = 784 个有效元素，另外 240 个位置是凑数的。grid 里填的就是这个 10，也就是 `triton.cdiv(n_elements, BLOCK)` 的值。换一组：n = 15、BLOCK = 16，只启动 1 个程序，15 个位置有效、1 个位置凑数。

`BLOCK` 必须是 2 的幂，这是硬限制：[`tl.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) 的 start 与 end 都要求是 2 的幂，长度上限 2²⁰ = 1,048,576，block 的每个维度也按 2 的幂组织；官方 fused softmax 教程把它写成“each block must have a power-of-two number of elements”。BLOCK 取 1000 会在编译时报错。这条规则不是风格偏好：编译器按固定宽度切向量通道、排布 shared memory，2 的幂是切得齐的前提。

向量加法写成 Triton，用的是官方教程 `01-vector-add` 的结构（`BLOCK` 是编译期常量）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK=1024)
    return output
```

三处细节各有代价。`mask` 处理尾巴：元素数不是 BLOCK 整数倍时，多余的位置不读也不写——读了可能越界，写了会写坏邻居张量的内存。`BLOCK: tl.constexpr` 表示编译期常量：编译器要按 BLOCK 特化生成代码（循环展开、shared memory 大小、向量宽度），改它会重新编译；autotune 能工作的前提就是这一点，它换 BLOCK 重编一遍再测。`grid` 写成 lambda 而不是固定元组：`triton.cdiv` 是向上取整除法，autotune 换了 BLOCK，程序数要跟着重算；写死成 `(10,)` 之后，BLOCK 从 1024 换成 256 时程序数还是按 1024 算的，后面七成多的数据没人处理。

mask 是这一章最容易写错的地方，两边必须同进同退。只给 load 加 mask、不给 store 加：凑数的 lane 把读到的垃圾写进输出张量的尾巴。只给 store 加、不给 load 加：读的那一边仍可能越界。上面 n = 10000 的例子里，9 号程序的 240 个凑数 lane 就是被同一个 mask 在两边同时挡住的。常见的替代方案是把张量 padding 到 BLOCK 的整数倍，省掉 mask；代价是改形状、多占内存、下游算子都要认这个新形状。mask 的代价只是一次比较，原形状上就能解决。

编译发生在第一次调用，不在 import 时。`@triton.jit` 的函数在第一次调用时编译，BLOCK 的取值在那一刻写进生成的代码；之后同样的 BLOCK 命中编译缓存，改 BLOCK 则重编一次。所以计时前必须先热身，第一次跑的耗时里混着编译，量出来的不是 kernel 的速度（20.3 的 `do_bench` 把这件事包了进去）。这个 kernel 里 `tl.load` 没有给 `other`：被 mask 掉的 lane 读进来是未定义值，但它们只参与同样被 mask 掉的写，不影响结果；softmax 那种“读进来要参与 max 与求和”的场合就必须给 `other`。

正确性照第 19 章的办法：与 `x + y` 的 PyTorch 结果做 `torch.allclose`。逐元素加法的每个输出都是一次独立的 IEEE 浮点加法，kernel 与 PyTorch 算的是同一个加法，逐 bit 相等本来就该成立，这里的 `allclose` 只是沿用检查习惯。容限真正成为问题要等到两处：20.2 的 `tl.dot` 在 fp32 下默认走 TF32，20.3 的归约求和顺序不同。检查方法一样，背后的理由不一样。

CUDA 与 Triton 的分工可以写成一张表，20.5 的取舍判断就是从这里长出来的：

| 维度 | CUDA | Triton |
|---|---|---|
| 写的单位 | 一个线程算一个元素 | 一个程序算一整块 |
| 下标 | 标量 `blockIdx * blockDim + threadIdx` | 向量 `pid * BLOCK + tl.arange(0, BLOCK)` |
| 线程与访存 | 自己组织、自己同步 | 编译器组织，编译器管搬运与向量化 |
| 能调的旋钮 | 线程、shared memory、同步、流水线 | BLOCK、num_warps、num_stages、访存顺序 |
| 够不着的 | —— | warp 级同步、shared memory 精确布局、异步流水线的分工 |

旋钮只有两个是常调的：`BLOCK`（一个程序处理多少元素）与 `num_warps`（一个程序用多少个 warp，blockDim = 32 × num_warps）。BLOCK 越大，每程序要扛的寄存器越多，压不住就 spill 到 local memory（溢出量实际落在显存里），反而更慢——“BLOCK 越大越快”不成立，快慢要量。`@triton.jit` 的函数编译成机器码，Python 只在编译与启动时跑，kernel 执行时 Python 早就不在路径上，慢的从来不是“Python”这个标签。mask 防的也不只是越界崩溃：写坏邻居张量的尾巴不会立刻报错，只会在很久以后以另一个错误爆出来。

### 第 2 节 · 三类谜题：逐元素、归约、矩阵乘

Triton Puzzles 用 Triton 的解释器跑，不需要 GPU：本地设 `TRITON_INTERPRET=1` 走解释器，notebook 里用 triton-viz 的 tracer，启动方式以仓库 README 与 notebook 顶部说明为准。解释器在 CPU 上逐元素模拟，速度没有参考价值，它的用处是把每个程序读了哪块、写了哪块画出来。谜题编号会随仓库更新，本节按当前版本写的是 1 Constant Add、2 Constant Add Block、3 Outer Vector Add、7 Long Sum、11 Matrix Multiplication；编号对不上时按标题找，以仓库当前版本为准。

谜题 1 是 `z_i = 10 + x_i`，块大小 B0 等于向量长度 N0，一个程序盖满整条向量。这道题要看的只有一件事：一个程序读的是不是一段连续地址、写的是不是同一段。以 1024 个连续的 fp32 为例，一共 4096 字节，相邻 lane 读相邻地址，硬件一次搬运能覆盖整块；让相邻 lane 去读相隔很远的地址，同样是 1024 次读取，搬运次数要翻几倍。元素个数没变、算术没变，变的是地址的排列。

谜题 2 让 B0 小于 N0，用 program_id 沿输入铺开。这就是 20.1 的 n = 10000 那张图：10 个程序各读一段，9 号程序的 240 个凑数 lane 被 mask 挡在图外。看图时查两件事——相邻程序读的地址是否相邻，尾巴有没有被挡住。这两件事对了，kernel 的访存骨架就对了。

谜题 3 是 `z[j, i] = x_i + y_j`，两条一维向量生成二维输出。写法是 `x[None, :] + y[:, None]`，两个方向各广播一次，得到一个 B1 × B0 的块。这个形状是矩阵乘 tile 的雏形：一个程序负责输出矩阵的一块，读 A 的一行段与 B 的一列段。

归约题（Long Sum，7）对一批向量做行内求和：输入是 N0 行、每行 T 个元素，每个程序负责其中几行，沿 T 方向以 B1 为块循环，每块用 `tl.sum(x, axis=1)` 得到块内部分和，累加成整行的和。`tl.sum` 只在 block 内做：一个程序内部的线程可以同步、可以交换寄存器里的数；程序与程序之间没有会合点，Triton 没有 grid 级 barrier，一个程序没法等另一个程序。要把 10⁶ 个数分给多个程序各算一段，就得再开一层：第一轮每个程序算局部和，977 个程序（BLOCK 1024）得到 977 个部分和，共 977 × 4 = 3908 字节；第二轮一个程序读这 3908 字节收尾。第二轮的活不到第一轮的千分之一。

block 内的树可以拆开看。`tl.sum` 对 1024 个数求和是两两相加的树，每轮减半，log₂ 1024 = 10 轮收完。树怎么排是编译器定的，两个 kernel 的树不一样，FP32 的舍入就不一样——这一条在 20.3 的容限里兑现。

矩阵乘题（Matrix Multiplication，11）用 `tl.dot(a, b)`，两个输入是二维或三维 block（三维时第一维是 batch），内维对齐；这道题用三个 program id 轴分别管 batch、行、列，K 方向以 B_MID 为块循环。`tl.dot` 最终落到 tensor core 的 MMA 指令，MMA 的最小分片是 16 行 × 8 列这个量级（Ampere 上 fp16 是 `m16n8k16`，fp32/TF32 是 `m16n8k8`），所以 tile 的 M、N、K 一般从 16 起；比这个尺寸小的 `tl.dot` 不会被降到 MMA，要么退回逐元素 FMA 的路径，要么在编译期报错，具体行为以当前版本为准。读入 tile 有两种写法：手动算指针加 mask（`a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak`），每一维的地址都算在明处；`tl.make_block_ptr` 把指针、形状、步长、边界打包，越界由它兜底，写得短。先把一种写通。

1024 × 1024 的矩阵乘可以先手算一遍。取 BLOCK_M = BLOCK_N = BLOCK_K = 32：M、N 各切出 32 块，程序数是 32 × 32 = 1024 个；每个程序沿 K 走 32 步，每步从 A 读一块 32 × 32 × 4 字节 = 4096 字节、从 B 读同样一块，共 8 KB；每程序 32 × 8 KB = 256 KB，1024 个程序合计读约 268 MB。A 与 B 各约 4 MB，各被读了 32 遍——重复读的量由 tile 尺寸决定。算术强度 = 2 × 1024³ FLOP ÷ 268 MB ≈ 8 FLOP/字节，正好等于 BLOCK ÷ 4，与第 19 章分块 GEMM 的算式一致；BLOCK 从 32 提到 64，搬运量减半、算术强度翻到 16。

fp32 的 `tl.dot` 还有一条精度边界：`input_precision` 的默认值是 `"tf32"`，只在设备有 tensor core、输入是 f32 时生效。TF32 的尾数只有 10 位（加上符号与 8 位指数共 19 位），官方文档还提醒 f32 输入可能被直接截断而不带舍入。与 PyTorch 的 fp32 矩阵乘逐项对比时，差的往往不是分块错，是精度模式：要么把容限放到 TF32 的量级，要么 `input_precision="ieee"`（慢），要么把 PyTorch 侧也开成 TF32 再比。不写清这一条，“矩阵乘结果差一点”会被当成 bug 查很久。

这几道题的验收标准与第 19 章相同：与 PyTorch 参考实现逐项对比，数值对上才算过。对比点要选得靠近边界：尾巴 mask 的错在规模小时看不出来，规模一大就露馅；分块边界的错在中间几行几列全对，边界行列才露馅。

### 第 3 节 · Fused softmax：融合的账、数值稳定性、计时与 autotune

softmax 拆开算是六遍。max 读一遍；减 max 再 exp 读一遍、写一遍；求和读一遍；除法读一遍、写一遍——读 4 遍、写 2 遍。一行 1024 个 fp32 是 4096 字节：拆开是 4 × 4096 读 + 2 × 4096 写 = 24 KB，融合版只读一遍写一遍 = 8 KB，3 倍。行变长倍数不变：4096 列的一行是 16 KB，拆开 96 KB，融合 32 KB。

这个 6 遍把“减 max 与 exp”算成了一步。PyTorch 的朴素写法会把 `x - x_max` 物化成一个中间张量，那就是读 5 遍、写 3 遍，共 8 遍，官方教程按这个口径算出的加速比约 4 倍。两个数都对，差别只在遍数怎么数；引用倍数时把遍数一起写出来，别让两组数对不上。放到整个矩阵上：4096 × 4096 的 fp32 矩阵占 67,108,864 字节 ≈ 67 MB，8 遍 ≈ 537 MB，6 遍 ≈ 403 MB，融合版 2 遍 ≈ 134 MB；本章按 6 遍的口径，省下约 269 MB 的显存往返。exp 与除法的次数一次没少，省的全是搬运。

做法是一行一个程序，整行读进寄存器，算完写回：

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=BLOCK)
    return y
```

这段结构与官方教程 `02-fused-softmax` 的循环体一致；当前教程版本还按占用率算出程序数、在 kernel 外套一层 `tl.range` 让一个程序处理多行，并用 `num_stages` 让行循环流水起来。作用相同，写法以当前版本文档为准。

`BLOCK = triton.next_power_of_2(n_cols)` 是硬要求：整行要放进一个 block，BLOCK 取大于等于列数的最小 2 的幂。代价是寄存器：1024 列时一行占 4096 字节，扛在寄存器里没问题；513 列时 BLOCK 跳到 1024，513 个有效元素配 511 个 mask 掉的 lane；1025 列时 BLOCK 取 2048，一行 8 KB；1,048,576 列时 BLOCK 得取 2²⁰，一行约 4 MB，寄存器扛不住，而 2²⁰ 已经是 `tl.arange` 的长度上限——“一行一个程序”这条路走到头了。第 21 章的 online softmax 是它的分块版：行切块、每块维护运行最大值与运行和，最后拼起来。单块版与分块版是同一个公式的两种情况，不是两种算法。

`row - tl.max(row)` 不是优化，是正确性。softmax 对整体平移不变：exp(x_i − m) / Σ exp(x_j − m) 里分子分母的 exp(−m) 约掉，结果与不减 max 完全一样。不减 max，FP32 下 x 超过 88.72 就溢出：FP32 最大约 3.4028235 × 10³⁸，ln 它约 88.72；exp(88) ≈ 1.65 × 10³⁸ 还在范围内，exp(89) ≈ 4.49 × 10³⁸ 已经超过上限，成 inf。fp16 更早：最大 65504，ln 它约 11.09，exp(11) ≈ 59874 还在，exp(11.1) ≈ 66171 已经超了。本章的 softmax 输入是 fp32；第 22–23 章碰到 fp16 kernel 时这个数会再用上。

拿数验一遍平移不变性。x = [0, 1, 2]，减 max 得 [−2, −1, 0]，exp 得 [0.1353, 0.3679, 1]，和 1.5032，softmax 是 [0.0900, 0.2447, 0.6652]。整行加 1000，x = [1000, 1001, 1002]，减 max 还是 [−2, −1, 0]，结果一模一样；不减 max 直接算 exp([1000, 1001, 1002])，三个都是 inf，inf / inf 得 NaN，整行全毁。

`other=-float('inf')` 是同一逻辑的另一半，两个作用落在两行代码上。max 那一步：mask 掉的位置读进来是 −inf，任何有限数都比它大，−inf 赢不了最大值。求和那一步：exp(−inf) = 0，mask 掉的位置对分母的贡献是 0。漏掉 `other` 时，文档只说被 mask 的位置是“未定义值”，没有承诺是 0：如果恰好是 0，整行全负时 0 会赢过真实的最大值，exp(0) = 1 又会污染分母；如果不是 0，错得更没规律。两种都错，排查时却只看到一个“结果不对”。

正确性检查与 `torch.softmax(x, dim=-1)` 对比，留 `atol=1e-5`。容限是必须的：两个 kernel 的求和树不一样，FP32 舍入落在不同位置，逐 bit 相等不该指望；官方教程对 1823 × 781 的随机矩阵用默认 `torch.allclose` 就能过，1e-5 比它还宽松。误差到 10⁻² 量级就不是舍入了：先查 mask 与 `other`，再查 BLOCK 是否盖住了整行。

计时用 `triton.testing.do_bench`，口径要写清（参数名与默认值随版本变，以当前版本文档为准）：`warmup=25`、`rep=100` 的单位是毫秒，不是次数。它先跑一次估出单次耗时，再按 warmup ÷ estimate 与 rep ÷ estimate 算出热身次数与重复次数；每次重复用一对 CUDA event 计时；每次重复前把一个缓冲区清零，把 L2 冲掉，所以量到的是冷 L2 的数，小张量会因此偏慢。默认 `return_mode="mean"` 返回均值，要中位数就传 `return_mode="median"`（或 `quantiles=[0.5]`）。如果 kernel 套了 autotune，第一次调用会先扫一遍配置，这次耗时同样落在热身里，不计入结果。

```python
# 在 softmax() 里：n_rows、n_cols、BLOCK、y 都已经算好
ms = triton.testing.do_bench(
    lambda: softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=BLOCK)
)
ms_median = triton.testing.do_bench(
    lambda: softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=BLOCK),
    return_mode="median",
)
```

均值与中位数的差别可以拿一组数看：五次测量 [10.2, 10.1, 10.3, 45.7, 10.2] 毫秒，45.7 是某次系统中断拖出来的异常值。均值 (10.2 + 10.1 + 10.3 + 45.7 + 10.2) ÷ 5 = 17.3 毫秒，被一个异常值拉高 7 毫秒；中位数是排序后中间的那个，10.2 毫秒，异常值对它没有影响。一次中断就让均值偏了约 70%。热身解决的是另一件事：第一次调用要编译，编译是毫秒到秒级，一次 softmax 执行在微秒到毫秒级，不把第一次排除掉，量到的主要是编译。

再套一层 `@triton.autotune`，让 Triton 自己扫配置。这个 kernel 的 BLOCK 由 n_cols 决定（必须 ≥ 列数才能算对），真正能扫的只有 num_warps：

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK: tl.constexpr):
    # 与上面相同的 kernel 体：autotune 只改 num_warps，BLOCK 由调用方传
    row_idx = tl.program_id(0)
    ...
```

`key` 是重新调优的依据：n_cols 变了，最优配置可能也变，autotune 会重新扫一遍；同一形状下只在第一次调用时扫，后面直接用记下来的配置。候选集里放什么由你定。对 BLOCK 可以自由取的 kernel（逐元素、矩阵乘），两个旋钮的配平可以先手算：一个 warp 是 32 个线程，num_warps = 4 是 128 个线程，num_warps = 8 是 256 个线程；BLOCK 个元素分给这些线程，每线程扛 BLOCK ÷ (32 × num_warps) 个。BLOCK = 128 配 4 个 warp，每线程 1 个元素；BLOCK = 1024 配 8 个 warp，每线程 4 个；BLOCK = 1024 配 4 个 warp，每线程 8 个，寄存器压力翻倍，压不住就 spill。

autotune 只挑最快的，不检查正确性：候选集里放一个 BLOCK 小于列数的配置，它算出来是错的，而不是慢的。softmax 这类“块必须盖住整行”的 kernel，候选集要么把 BLOCK 固定住、只扫 num_warps（如上），要么用 `prune_configs_by` 把不合法配置先剪掉（参数以当前版本为准）。

写一个与 PyTorch 对齐的 fused softmax，再留一份 autotune 记录：哪个 n_cols 选了哪个配置、各自多少毫秒。第 21 章的 online softmax 是它的分块版；第 24 章接回引擎的练习用的是 SiluAndMul，这个 softmax 留作对照。

### 第 4 节 · 把 kernel 翻译回 CUDA 语义

写得出 Triton 不代表知道它变成了什么。把这一章写过的 kernel 逐行翻译回 CUDA，能标出两件事：编译器替你做了哪些决定，哪些决定它替你做不了。前者是 Triton 省下的工作，后者就是 20.5 的边界。

| Triton 里的写法 | 编译后大致对应什么 | 谁做这个决定 |
|---|---|---|
| `tl.program_id(0)` | `blockIdx.x` | 你定 grid 大小 |
| `pid * BLOCK + tl.arange(0, BLOCK)` | `blockIdx.x * blockDim.x + threadIdx.x` 加循环或展开 | 编译器：哪个 lane 算哪个元素 |
| `BLOCK`、`num_warps` | `blockDim = 32 × num_warps`，每线程扛 BLOCK ÷ blockDim 个元素 | 你给数，编译器给映射 |
| `tl.load(ptr, mask=mask, other=...)` | 带谓词的 `ld.global`；相邻地址合并成 128 字节事务 | 编译器：合并与向量化 |
| `tl.store(ptr, v, mask=mask)` | 带谓词的 `st.global` | 编译器 |
| `tl.max`、`tl.sum(axis=0)` | warp 内 shuffle 归约 + 跨 warp 的 shared memory 暂存与同步 | 编译器：树的形状与缓冲 |
| `tl.dot(a, b)` | `ldmatrix` 取数 + `mma.sync`（Hopper 上是 `wgmma`）+ shared memory 暂存 | 编译器：流水线与排布 |
| `tl.constexpr` | 模板参数，每个取值生成一份特化代码 | 你定特化点 |
| `@triton.autotune` | 编译多份、各测一遍、按 key 缓存 | 你定候选集 |

20.1 的向量加法翻译成 CUDA 是这样（示意，只保留与 Triton 版对应的部分）：

```cuda
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
// 启动：add_kernel<<<(n + 127) / 128, 128>>>(x, y, out, n);
```

两版逐项对上：Triton 的 `mask` 就是这里的 `if (i < n)`，`tl.load` 与 `tl.store` 的谓词来自同一个 mask；`grid` 是 `<<<...>>>` 的第一个参数，`num_warps` 换成了 `blockDim = 128`（默认 4 个 warp）。逐元素 kernel 两边差不多长——Triton 省下的不是行数，是“哪个线程算哪个元素”这个决定：CUDA 版由你写出 `i`，Triton 版由编译器把 BLOCK 个元素分给 32 × num_warps 个 lane。

分块矩阵乘就差得多了。Triton 的 K 循环是 `for k in range(0, tl.cdiv(K, BLOCK_K))`，每轮两次 `tl.load`、一次 `acc += tl.dot(a, b)`；同样的循环用 CUDA 写，要自己声明 shared memory、自己把 tile 搬进去、在每轮首尾插 `__syncthreads()`、自己处理 K 不是 BLOCK_K 整数倍的尾巴。Triton 把 shared memory 暂存与 MMA 指令的排布交给编译器，这是它在矩阵乘上省下一个量级代码量的地方。

翻译不出来的部分，正好是编译器替你决定、也替你锁住的部分：

- shared memory 布局：你能说“这个 block 读哪块”，说不了“这块在 shared memory 里按什么 stride 摆、要不要 padding”。第 19 章用 padding 解 bank conflict，那一步在 Triton 里没有对应的旋钮。
- warp 级同步与 warp 分工：`__syncwarp()`、`__shfl_sync()` 没有对应写法；归约内部用它们，但不暴露给你。FlashAttention 第 3 代那类“一部分 warp 搬、一部分 warp 算”的排布也没有对应写法（截至当前版本，以文档为准）。
- 异步拷贝流水线：`cp.async`、TMA、多级 double buffer 由编译器排；`num_stages` 只给级数，哪一级搬哪块、什么时候发出去，不由你定。
- 寄存器分配：只能通过 BLOCK 与 num_warps 间接影响，没有逐变量的控制。

编译产物可以拿来核对这张表。编译后的 kernel 对象带着 IR 与 PTX（`CompiledKernel.asm`，键名随版本变；也可以设 `TRITON_KERNEL_DUMP=1` 把中间产物写到磁盘，环境变量名以当前版本文档为准）。在 PTX 里搜 `ld.global`、`bar.sync`、`mma.sync`，表里的每一行都能落到一条指令上：`tl.load` 是带谓词的 `ld.global`，多 warp 时 `tl.sum` 会引入 block 内的同步（`bar.sync`），`tl.dot` 是 `mma.sync`（Hopper 上是 `wgmma`）。搜不到的行，要么表写错了，要么那一行被优化掉了。

拿 20.3 的 softmax 做这个练习：逐行写它的 CUDA 对应，写不出来的行标出来。标出来的那些，就是 20.5 判断表里“回 CUDA”那一列。

### 第 5 节 · Triton 什么时候够用，什么时候必须回 CUDA

判断标准只有一条，分两面说：优化能用 block 语言说清——“哪个程序读哪块、按什么顺序读”——就留在 Triton；必须说到“哪个 warp 干什么、哪个 bank、流水线的第几级”，就回 CUDA。道理不复杂：Triton 抽象掉的正是线程、warp、shared memory 的管理，而硬件细节的优化要调的恰恰是这些东西，抽象藏住了你需要控制的对象。

| 场景 | 选什么 | 原因 |
|---|---|---|
| 逐元素、归约、简单融合 | Triton | 优化的全部内容是访存模式，编译器接手了线程与向量化 |
| 需要精确控制 shared memory 布局或 warp 级同步 | CUDA | 这两样在 Triton 的抽象层里没有对应写法 |
| 要榨到接近硬件上限（FlashAttention 3 那个级别） | CUDA / CUTLASS | warp specialization 与异步流水线要逐级排布 |
| 快速验证一个融合想法 | Triton | 解释器无卡可跑，改一版只动 BLOCK 与 mask |

把这套标准套到本章的三个 kernel 上。向量加法的全部内容是“10 个程序各读哪段、尾巴怎么挡”，矩阵乘是 tile 尺寸与 K 方向的步进，softmax 是“6 遍变 2 遍”——三个都能用 block 语言说完，所以这一章没有一道题必须回 CUDA。这不是 Triton 万能，是这三道题还没碰到边界。

边界长什么样，可以对着表里的第 2、3 行看。shared memory 布局与 padding 是第 19 章解 bank conflict 的手段，Triton 里能说“这个 block 读哪块”，说不了“这块在 shared memory 里怎么摆”。warp 级同步与 warp specialization 是 FlashAttention 3 的工程核心，一部分 warp 专职搬运、一部分专职计算，流水线的第几级在干什么都要排好，这些在 Triton 里没有对应写法。异步拷贝的级数能通过 `num_stages` 给，但“谁搬、谁算、什么时候发”给不了。手写 CUDA 能调到最细；CUTLASS 把流水线与分块做成可复用的模板，站在模板上改比从零手写省事。

工作流是两条道接力，不是二选一。想法先在 Triton 里跑通：解释器无卡也能验正确性，访存图能看对，BLOCK 改一版只是改个数。profile 说瓶颈还在访存模式，就继续在 Triton 里调 BLOCK 与 num_warps；profile 说瓶颈已经下沉到硬件细节，再把验证过的 Triton 版当参照物搬到 CUDA——两个版本对同一组输入给出同样的数，移植引入的错就只剩性能问题。

选错方向的代价可以提前算。简单的融合想法用 CUDA 起手，访存账不会因为换了语言而变（24 KB 对 8 KB 还是 3 倍），多花的是手写线程、shared memory 与同步的时间；需要 warp 级控制的 kernel 用 Triton 写到底，写到要调布局与流水线时才发现旋钮不在抽象层里，前面的代码要推倒。判断的价值在于把这次返工提前到动笔之前。

本章攒出来的检查清单：

- mask 在 load 与 store 两边都写，且用同一个条件；
- BLOCK 是 2 的幂，且不超过 2²⁰；
- 读进来要参与计算的 masked lane 必须给 `other`（softmax 用 −inf）；
- 计时写明是均值还是中位数，热身与重复的次数口径一起写；
- fp32 的 `tl.dot` 写明 `input_precision` 是 TF32 还是 ieee；
- BLOCK 与 num_warps 配对，算出每线程扛几个元素。

往后几章的判断会继续加码。第 21 章的 online softmax 仍是 block 语言——rescale 是 block 级的更新规则，Triton 够用；第 22 章 int8 的 `tl.dot` 与 autotune 走同一条路；第 23 章读 FlashAttention 2/3 与 CUTLASS 时，话题下沉到 warp 分工与流水线排布，天平往 CUDA 移；第 24 章把 kernel 接回引擎，量的是端到端的变化。判断表每章加一行，不是给语言排座次，是给下一个 kernel 选起手的地方：能用 block 说清的先在 Triton 里写完，说不清的再从 CUDA 起手。

**延伸**：[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)（vector add、fused softmax、matrix multiplication 三篇对应 20.1–20.3）、[Triton 语言文档](https://triton-lang.org/main/python-api/triton.language.html)（`program_id`、`arange`、`load`、`dot` 的参数与限制）、[Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles)（解释器里的访存可视化，编号以仓库为准）、[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://dl.acm.org/doi/10.1145/3315508.3329973)（Tillet、Kung、Cox，MAPL 2019：tile 级编程模型的设计动机）、[Datawhale《大模型算法与系统教程》](https://github.com/datawhalechina/llm-algo-leetcode)（第三部分 Triton Kernels，20 个可运行 Notebook，编号以仓库为准）。

## 第 21 章 · Fused Attention 深讲：先推导，再写 kernel

> 硬件：本周需要 GPU 跑性能对比；无卡可用 LeetGPU 的免费 GPU 评测先做正确性验证；第 1–2 节的推导与计算无卡可做。
>
> 本周的顺序是反过来的：先推导，再写 kernel——第 3 节的每一行代码都能在第 1 节的公式里找到对应。

### 第 1 节 · Online softmax 的数学推导

**目标**：手推 online softmax 的 rescale 公式，并证明分块算出的 attention 与标准 attention 完全相等（不是近似）。

**怎么做**（60 分钟，无 GPU 可做）：

先把困难说清楚。标准 softmax 是一行向量 `x`，`softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)`，其中 `m = max(x)`。减最大值是为了数值稳定：FP32 下 x 超过 89 左右，exp(x) 就溢出成 inf，整个结果全毁；减了 max 之后 exp 的输入全 ≤ 0，结果全 ≤ 1，溢出不可能发生。

困难在于，`m` 和分母的求和都是整行级别的量：算任何一个 `exp(x_i − m)` 之前，你得先知道整行的最大值；算分母之前，你得先把整行加完。分块之后，处理第一块的时候第二块还没读进来，`m` 和分母都还不知道——softmax 看起来是没法分块的。

解法是维护两个运行量（running statistics），每来一块更新一次。设已处理部分的运行最大值是 `m`、运行和是 `l = Σ exp(x_j − m)`。新来一块，算出块内最大值 `m_b` 与块内和 `l_b = Σ exp(x_j − m_b)`，更新规则是：

```text
m' = max(m, m_b)
l' = l · exp(m − m') + l_b · exp(m_b − m')
```

为什么这样更新是对的？旧的 `l` 是在旧的 `m` 下算的，新基准是 `m'`，旧的每一项都要乘 `exp(m − m')` 换算到新基准下；新块的 `l_b` 是在 `m_b` 下算的，同样乘 `exp(m_b − m')` 换算。`m` 只会长大不会缩小，`exp(m − m')` 永远 ≤ 1，所以旧状态只会被缩小，不会爆炸。换句话说，rescale 不是近似修正，是精确的换算——换基准不改变比值。

拿一组数走一遍。一行 `x = [1, 2, 3, 4]`，分两块 `[1, 2]` 和 `[3, 4]`。标准做法：`m = 4`，`exp(x − 4) = [e⁻³, e⁻², e⁻¹, 1] ≈ [0.0498, 0.1353, 0.3679, 1]`，求和 `l ≈ 1.5530`，softmax ≈ `[0.0321, 0.0871, 0.2369, 0.6439]`。

分块做法：第一块 `[1, 2]`，块内最大值 `m₁ = 2`，`l₁ = e⁻¹ + e⁰ ≈ 0.3679 + 1 = 1.3679`。第二块 `[3, 4]`，块内最大值 `m_b = 4`，`l_b = e⁻¹ + e⁰ ≈ 1.3679`。更新：`m' = max(2, 4) = 4`，`l' = 1.3679 × e^(2−4) + 1.3679 × e^(4−4) = 1.3679 × 0.1353 + 1.3679 ≈ 0.1851 + 1.3679 = 1.5530`——和标准做法的求和完全一致。`exp(x − 4) / 1.5530` 就是上面那组 softmax。分块没有改变任何一个数，只是换了个顺序算。

把同样的思想搬到 attention 上。设已处理部分的未归一化累加器是 `Õ = Σ_j exp(S_j − m) · V_j`（`S` 是 QK^T 的一行），新来一块 K、V：

```text
α = exp(m_old − m_new)              # 旧状态的 rescale 系数
P_block = exp(S_block − m_new)      # 块内概率（未归一化）
l_new = l_old · α + rowsum(P_block)
Õ_new = Õ_old · α + P_block · V_block
```

最后 `O = Õ / l_new`。注意 `α` 同时乘在 `l` 和 `Õ` 上：旧的求和与旧的累加器是在同一个旧基准下算的，换基准时两个要一起换，只换一个就错了。

同样拿数走一遍。设 `S = [1, 2, 3, 4]`，`V = [10, 20, 30, 40]`（d=1，好手算）。标准做法：softmax(S) ≈ `[0.0321, 0.0871, 0.2369, 0.6439]`，`O = 0.0321×10 + 0.0871×20 + 0.2369×30 + 0.6439×40 ≈ 0.321 + 1.743 + 7.107 + 25.756 = 34.926`。

分块做法：第一块 `[1, 2]`，`m₁ = 2`，`l₁ ≈ 1.3679`，`Õ₁ = e⁻¹×10 + e⁰×20 ≈ 3.679 + 20 = 23.679`。第二块 `[3, 4]`，`m_b = 4`，`α = e^(2−4) ≈ 0.1353`，`l' = 1.3679 × 0.1353 + 1.3679 ≈ 1.5530`，`Õ' = 23.679 × 0.1353 + (e⁻¹×30 + e⁰×40) ≈ 3.205 + (11.037 + 40) = 54.242`。最后 `O = 54.242 / 1.5530 ≈ 34.926`——和标准做法一致。两组数都对上，分块的正确性就不是信念，是算出来的。

证明写成归纳法。每一步的 invariant 是：`Õ = Σ_{已处理 j} exp(S_j − m_current) · V_j`，`l = Σ exp(S_j − m_current)`。第一块来之前 `m = -inf`、`l = 0`、`Õ = 0`，invariant 空成立；每来一块，更新规则恰好保持 invariant（上面那组数就是一次归纳步）；所有块处理完，`Õ` 和 `l` 就是整行的量，除一下得到标准 attention。归纳法走完，“分块不改变结果”就不是口号，是定理。

结论：FlashAttention 是精确算法。分块 + online softmax 改变的只是计算顺序与中间结果的存放位置（S、P 不再写回 HBM），最终输出与标准 attention 在数学上相等，差别只有浮点舍入。这一点后面第 5 周读 FA2/FA3 时还会用到：它们改的都是并行与流水线，不是数学。

一个常见的误解：“分块是近似”——不对。近似的意思是结果变了，分块的结果没变，变的是中间结果存在哪、按什么顺序算。把“分块”等同于“近似”的人，通常是把 rescale 当成了修正项；rescale 是精确换算，不是修正。

对照第 2 周的 fused softmax：那一节的 `row - tl.max(row)` 就是单块版的 online softmax——整行一次性读进来时，`m` 一步到位；分块时 `m` 是逐步长大的，`α` 就是为“长大”付的代价。单块和分块是同一个公式的两种情况，不是两种算法。

**产出**：一份推导笔记，写出 rescale 公式与 invariant，并用一句话回答“为什么分块不改变结果”。

> **论文选读（不读也能学完本周）**：FlashAttention 原论文（Dao 等，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)，2022 年 5 月；链接保留，原文是可选的更深阅读，本周正文已包含它的核心内容）。
>
> - **核心思想**：attention 慢、吃显存，根源在访存。标准 attention 把 N×N 的 S、P 矩阵写回 HBM 再读回，HBM 与片上 SRAM 之间的搬运才是瓶颈。论文提出的原则叫 **IO-aware**：算法设计要把“各级 GPU 显存之间的读写次数”算进去，而不是只算 FLOPs。
> - **核心逻辑链条**：① 用 tiling 把 Q、K、V 分块，一次只把一块读进 SRAM，在 SRAM 里算完再写回；② 用 online softmax 让 softmax 也能分块（就是本节推导的 rescale）；③ 由此 S、P 两个 N×N 中间矩阵**永远不落 HBM**，HBM 访问量从 O(N²) 降到 O(N)；④ 论文给出 IO 复杂度分析，证明在一定 SRAM 容量范围内这个做法是最优的。
> - **关键结论与数字**（论文摘要给出）：BERT-large 训练端到端提速 15%（序列长度 512，对比 MLPerf 1.1 训练速度纪录）；GPT-2 训练提速 3 倍（序列长度 1K）；Long-Range Arena 提速 2.4 倍（序列长度 1K–4K）。更长上下文还带来质量提升：GPT-2 困惑度好 0.7，长文档分类提升 6.4 个点；首次让 Transformer 在 Path-X（序列长度 16K，61.4% 准确率）与 Path-256（序列长度 64K，63.1% 准确率）上超过随机水平。
> - **为什么重要**：它把“算得快”的优化对象从 FLOPs 转向了访存，并且证明了精确 attention 也可以分块算——这正是 M2 整体的底层逻辑：先定位瓶颈（第 1 周的 ncu + Roofline），再按瓶颈改算法，而不是盲目堆算力。
> - **与本节的关系**：本节的推导就是论文第 2–3 节的核心机制；第 5 周会接着讲 FA2/FA3 在这条链上又改了什么（并行、流水线、FP8）。

**延伸**：[Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)的 fused attention（`06-fused-attention`）。

### 第 2 节 · SRAM 容量约束如何决定 block 大小

**目标**：给定你的 GPU，能自己算出 BLOCK_M / BLOCK_N 的上限，而不是抄教程的默认值。

**怎么做**（60 分钟，无 GPU 可做）：

先看两个数：`Br=128, Bc=64` 时一个 block 占 48KB，`Br=128, Bc=128` 时占 80KB。只把 `Bc` 翻了一倍，SRAM 占用就从 48KB 涨到 80KB，涨了三分之二。这一节就是要把这笔账算明白：多出来的 32KB 花在哪了，下一次调参数时该往哪个方向花。

为什么 SRAM 决定 block 大小？第 1 周讲过存储分层：HBM 大但慢，SRAM 快但小。fused attention 的整个卖点是 S、P 不落 HBM——代价是 Q、K、V、S 四块东西要同时塞进 SRAM。塞不下，kernel 就跑不起来。所以 block 大小的上限由 SRAM 容量算出来，公式和表都在上面。换个说法：融合省掉的 HBM 搬运，是用 SRAM 容量换来的，SRAM 就是这笔交易的预算。

列出一个 block 在 SRAM 里要放的东西（FP16，每元素 2 字节，head 维 `d`，Q 块 `Br×d`，K/V 块 `Bc×d`，S 块 `Br×Bc`）：

```text
SRAM ≈ Q + K + V + S
     = 2·d·Br + 2·d·Bc + 2·d·Bc + 2·Br·Bc
     = 2d(Br + 2Bc) + 2·Br·Bc   ≤ 每 block 可用的 shared memory
```

四项各有来头。Q 块是当前程序负责的那 `Br` 行 query，整个内层循环只读一次，在 SRAM 里一直留着等复用；K 块和 V 块是当前扫到的那 `Bc` 列 key 和 value，每扫一块新来一对；S 块是 `Br×Bc` 的 QK^T 分片——第 1 节的 `P_block` 就是从它算出来的。为什么 K 和 V 要各占一份？因为一次内层迭代里两件事都要做：先拿 K 算分数，再拿 V 做加权求和，分数算完之前 V 还没用上，V 读进来之后 K 的分数还在用，两块东西在时间上是交叠的，预算就得按同时在场算。公式里 `Bc` 出现两次，`2·d·Bc + 2·d·Bc`，就是这个意思。

公式里最值得盯着看的是最后一项 `2·Br·Bc`。Q、K、V 三项都只跟 `Br` 或 `Bc` 的一次方成正比，S 这一项是 `Br` 和 `Bc` 的乘积。`Br` 和 `Bc` 各翻一倍，Q、K、V 只翻一倍，S 翻两倍。拿数验证：`Br=64, Bc=64` 时 S 占 8KB，`Br=128, Bc=128` 时 S 占 32KB，整整 4 倍，而 Q 只从 8KB 涨到 16KB。block 开大之后，预算的大头会越来越向 S 倾斜。问一句：瓶颈到底在哪？多数时候答案是 S 这一项先顶到天花板。

手算一张表（`d=64`，A100 每 SM shared memory 164KB，H100 228KB；每 block 的可用上限用 `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` 查，以你所用 CUDA 版本的设备属性为准）。表里每一行的四项加起来就是总数，你可以顺着公式自己验算：

| Br | Bc | Q | K | V | S | 合计 |
|----|----|---|---|---|---|------|
| 64 | 64 | 8KB | 8KB | 8KB | 8KB | 32KB |
| 128 | 32 | 16KB | 4KB | 4KB | 8KB | 32KB |
| 64 | 128 | 8KB | 16KB | 16KB | 16KB | 56KB |
| 128 | 64 | 16KB | 8KB | 8KB | 16KB | 48KB |
| 128 | 128 | 16KB | 16KB | 16KB | 32KB | 80KB |
| 256 | 64 | 32KB | 8KB | 8KB | 32KB | 80KB |
| 256 | 128 | 32KB | 16KB | 16KB | 64KB | 128KB |
| 128 | 256 | 16KB | 32KB | 32KB | 64KB | 144KB |

先看两组花同样钱的搭配。`Br=64, Bc=64` 和 `Br=128, Bc=32` 都是 32KB，一组是方块，一组是瘦高条；`Br=128, Bc=128` 和 `Br=256, Bc=64` 都是 80KB，一组把钱花在更宽的 K/V 扫描上，一组花在更多的 Q 行上。同样 80KB 预算，第二组把钱花在更宽的 K/V 扫描上，第三组花在更多的 Q 行上；哪个更好取决于 N 和卡，得实测，下一节各跑一遍。预算相同不代表性能相同：`Bc` 更宽意味着内层循环步数更少，每步的 rescale 和求和次数也更少；`Br` 更大意味着一个程序算更多行，程序总数减半——N=1024 时，`BLOCK_M=64` 对应 16 个程序，`BLOCK_M=128` 对应 8 个，`BLOCK_M=256` 对应 4 个。程序总数直接由 `N / BLOCK_M` 决定，`BLOCK_M` 翻倍，程序数减半，并行度的结构跟着变。

再看 `d` 的影响。公式里 `d` 只乘在 Q、K、V 三项上，S 这一项没有 `d`。`d=128, Br=128, Bc=64` 时，Q 32KB、K 16KB、V 16KB、S 16KB，合计 80KB——和 `d=64` 时 `Br=128, Bc=128` 的 80KB 撞了个正着。同一个 80KB 可以是“`d` 翻倍”，也可以是“`Bc` 翻倍”，账面上一样，瓶颈可不一样：前者是每个元素的 head 维变宽了，后者是每步扫的列变多了。`d=128, Br=64, Bc=64` 又是另一组数：Q 16KB、K 16KB、V 16KB、S 8KB，合计 56KB。所以换了模型、换了 `d`，上一张表的结论要重算，抄一组参数过去是抄不来的。

理解取舍：block 越大，Q/K/V 的重复读越少（K、V 被更多 Q 行复用），但 SRAM 占用越大、一个 SM 上能同时驻留的 block 越少。只按 shared memory 算一个上限：A100 每 SM 164KB，32KB 的 block 最多驻留 5 个（164 ÷ 32 ≈ 5.1，取整 5），48KB 的最多 3 个，80KB 的最多 2 个，128KB 的只剩 1 个；H100 每 SM 228KB，32KB 的 block 最多 7 个，48KB 的最多 4 个，80KB 的最多 2 个，128KB 的只剩 1 个。80KB 的 block 在两张卡上都只能驻留 2 个——从 48KB 涨到 80KB，SRAM 占用涨了三分之二，并行驻留数却从 3 个掉到 2 个。这个上限只是 shared memory 一项算出来的，寄存器、warp 数这些别的限制只会让它更小，不会更大。这就是第 1 周 occupancy 那一节在 attention 上的回响——第 1 周调的是寄存器和 block 大小，这里调的是 SRAM 和 block 大小，限制因子换了，思路没变：先找出哪项资源先顶到天花板，再决定参数往哪边让。

有两个常见误解。一个是把 164KB 当成整张卡的 SRAM：164KB 是每 SM（流多处理器）的 shared memory 上限，一张卡有几十上百个 SM，每个 SM 各有各的 164KB，block 之间不共享。算预算的时候，分母永远是“每 SM 每 block”的数。另一个是以为 block 越大越好：上面的表已经说明，block 开大之后 S 那一项按乘积膨胀，驻留数跟着往下掉，省下来的重复读未必抵得过并行度的损失。到底哪边赢，没有公式能直接告诉你，公式只负责告诉你哪些组合在预算内，预算内的组合得实测分高下。

Triton 里的对应关系：你不直接管 SRAM，`BLOCK_M` / `BLOCK_N` 就是旋钮；配额超了编译或运行时会直接报错（shared memory 不足），这是最直接的反馈。顺带一提，`BLOCK_M`、`BLOCK_N` 取 2 的幂、16 的倍数，和第 2 周 `tl.dot` 对形状的要求是同一回事：`tl.arange` 要求长度是 2 的幂，`tl.dot` 要求参与矩阵乘的维度至少 16，尺寸取成 64、128 这样的数，两边都对得上。把你算出的两组参数记下来，下一节各跑一遍。

**产出**：一张“block 大小 vs SRAM 占用”计算表（至少三组参数，含你 GPU 的上限），并标出你下一节要先试哪一组。

### 第 3 节 · 动手写 fused attention（单 head、单序列版）

**目标**：写出一个能跑的 fused attention，与 PyTorch 参考数值对齐。

**怎么做**（60 分钟，需 GPU）：

读 Triton 官方教程 `06-fused-attention` 的结构：外层循环按 `BLOCK_M` 分 Q 行，内层循环按 `BLOCK_N` 扫 K/V，每步做一次 online softmax 更新。下面是单 head、单序列、无 causal mask 的版本（多 head、变长、causal 是它的扩展，第 5 节留作练习）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, Out,
              stride_qm, stride_kn, stride_vn, stride_om,
              N, sm_scale,
              D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N
    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :],
             acc, mask=mask_m[:, None])

def fused_attention(q, k, v):
    # q, k, v: [N, D]，fp16，D=64；单 head、单序列、无 causal mask 版
    N, D = q.shape
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M),)
    _attn_fwd[grid](q, k, v, out,
                    q.stride(0), k.stride(0), v.stride(0), out.stride(0),
                    N, 1.0 / (D ** 0.5),
                    D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
```

先把整体结构对上第 1 节的推导。`m_i` / `l_i` / `acc` 就是运行量（running statistics）：`m_i` 是已处理部分的运行最大值，`l_i` 是运行和，`acc` 是未归一化的累加器 `Õ`。`alpha` 就是 rescale 系数 `α = exp(m_old − m_new)`。循环每走一步，三个运行量按第 1 节的更新规则一起往前推一步；循环结束，`acc / l_i` 就是标准 attention 的输出。代码和公式之间没有多余的东西：公式里有的，代码里都有；代码里有的，公式里都能找到对应。

拿一组具体尺寸把执行过程走一遍。`N=128, D=64, BLOCK_M=64, BLOCK_N=64` 时，`grid = (triton.cdiv(128, 64),)` 等于 2 个程序，每个程序负责 64 行 query，内层循环走 2 步（128 ÷ 64）。Q 块是 64×64 的 fp16，一共 8KB，只在循环外读一次；每步读进来的 K 块和 V 块各是 64×64 的 fp16，各 8KB；S 块 `qk` 是 64×64 的 fp32，全程只活在片上，从不写回 HBM。这就是分块的 payoff：Q 读一次、反复用，K、V 每块读一次，S、P 全程不落 HBM。

再算一笔 FLOPs 的账，验证代码算的量和理论算的量对得上。一次 `tl.dot` 是 64×64 乘 64×64 的矩阵乘，乘加次数是 64×64×64 = 262,144，折合 FLOPs（一次乘加算 2 次浮点运算）是 524,288。内层每步有两次 `tl.dot`（一次算 `qk`，一次算 `P·V`），每步 1,048,576 FLOPs；每个程序走 2 步，一共 2,097,152 FLOPs；2 个程序加起来是 4,194,304 FLOPs。理论公式是 `4·N²·d`，N=128、d=64 时正好是 4×16,384×64 = 4,194,304。两边对上了，代码没有多算也没有少算。

逐行看。`pid_m = tl.program_id(0)` 拿程序编号，`offs_m` 是这个程序负责的 64 行行号，`offs_d` 是 0 到 63 的 head 维下标。`mask_m = offs_m < N` 是尾巴处理：N 不是 BLOCK_M 的整数倍时，多出来的行号要拦住。`q` 的那次 `tl.load` 里，`mask=mask_m[:, None]` 把越界的行挡在外面，`other=0.0` 给它们填 0。注意一个细节：被 mask 掉的 Q 行在循环里照样会算，只是最后 `tl.store` 时也被 `mask_m` 拦住，算出来的东西永远写不出去。算了白算，但保证了程序不越界读内存——正确性优先，浪费的那几行是小头。

`m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)`、`l_i = tl.zeros`、`acc = tl.zeros` 是第 1 节归纳法的起点：第一块来之前 `m = -inf`、`l = 0`、`Õ = 0`，invariant 空成立。为什么 `m_i` 的初始值必须是 `-inf`？看第一轮循环就明白了：`m_block` 是第一块的块内最大值，`m_new = tl.maximum(m_i, m_block)` 一定等于 `m_block`，`alpha = tl.exp(m_i - m_new)` 等于 `exp(-inf)`，也就是 0。旧状态是全零，乘 0 还是全零，旧状态被正确清掉。如果 `m_i` 初始值错写成 0，第一轮的 `alpha` 就成了 `exp(0 − m_block)`，一个不等于 0 的数，零状态被莫名其妙地缩放了一轮——虽然初值是零，缩放后还是零，结果碰巧不错，但逻辑已经错了，第二轮的 invariant 就无从谈起。

内层循环里，`offs_n` 是当前 K/V 块的列号，`mask_n` 同样是尾巴处理。`k` 读进来之后，`qk = tl.dot(q, tl.trans(k))` 算出 64×64 的分数矩阵。`tl.trans(k)` 的原因是 `tl.dot` 要的是矩阵乘：`q` 是 `[BLOCK_M, D]`，`k` 是 `[BLOCK_N, D]`，转置之后才是 `[D, BLOCK_N]`，乘出来正好是 `[BLOCK_M, BLOCK_N]` 的 QK^T。

`qk = qk * sm_scale` 里的 `sm_scale = 1/sqrt(D)` 不是随便选的。Q、K 的元素是标准正态分布（`torch.randn` 采出来的，方差为 1），QK^T 的每一项是 D 个独立乘积的和，每个乘积的方差是 1，和的方差就是 D，标准差是 `sqrt(D)`。D=64 时标准差是 8，`qk` 里 typical 的数值量级就是 8 左右。除以 `sqrt(D)` 之后标准差回到 1，softmax 的输入拉回 O(1) 的量级。`sm_scale` 的数值是 1/8 = 0.125，`fused_attention` 里传的就是 `1.0 / (D ** 0.5)`。不除会怎么样？分数的量级是 8，两个分数差 16 再常见不过，权重之比就是 `e^16 ≈ 8.9×10^6`——softmax 还没开始算，最大的一项已经把其他项压得看不见了，attention 退化成只看一个位置的硬选择。缩放的作用就是不让分数在进 softmax 之前就分出胜负。

`qk = tl.where(mask_n[None, :], qk, -float('inf'))` 是尾巴处理：超出 N 的 K 位置被置成 -inf，exp 之后是 0，对求和与累加都没有贡献——和第 2 周的 `other=-inf` 是同一个手法。为什么非 -inf 不可？0 的 exp 是 1，会实实在在地加进分母。拿第 1 节的数算一下就知道代价有多大：`S = [1, 2, 3, 4]` 时 `l = 1.5530`，多一个没 mask 干净的位置贡献 `exp(0) = 1`，`l` 变成 2.5530，凭空涨了 64%，输出全错。-inf 的 exp 是 0，加了等于没加，这才是“不存在的位置”该有的样子。

`m_block = tl.max(qk, axis=1)` 是当前块每行的块内最大值，`m_new = tl.maximum(m_i, m_block)` 是更新后的运行最大值，`alpha = tl.exp(m_i - m_new)` 是 rescale 系数。`alpha` 永远小于等于 1，因为 `m_i` 不可能比 `m_new` 大，指数的输入永远小于等于 0。`p = tl.exp(qk - m_new[:, None])` 是块内概率（未归一化），`m_new[:, None]` 把 `[BLOCK_M]` 的向量拉成 `[BLOCK_M, 1]`，沿着列的方向广播。`l_i = l_i * alpha + tl.sum(p, axis=1)` 就是第 1 节的 `l_new = l_old · α + rowsum(P_block)`，一行代码对一行公式。

`v` 读进来之后，`acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)` 是最容易写错的一行，两个细节都在这里。第一个细节是 `alpha` 必须同时乘在 `l_i` 和 `acc` 上：旧的求和与旧的累加器是在同一个旧基准下算的，换基准时两个要一起换，只换一个就错了。拿第 1 节的数验证：第一块算完 `Õ₁ = 23.679`，第二块的 `α = 0.1353`，正确的更新是 `23.679 × 0.1353 ≈ 3.205` 再加上第二块的贡献 `51.037`，得到 `Õ' ≈ 54.242`，最后 `O = 54.242 / 1.5530 ≈ 34.926`。如果 `acc` 忘了乘 `alpha`，`Õ'` 就成了 `23.679 + 51.037 = 74.716`，`O ≈ 48.11`，和 34.926 差了 13.18——误差大到一眼就能看出来，这正是验收时先查 rescale 的原因。

第二个细节是 `p.to(v.dtype)` 里的 `.to`。`p` 是 `exp` 的结果，fp32；`v` 是 fp16，`tl.dot` 要求两边同类型才进 tensor core，所以 `p` 要先 cast 回 fp16。这个 cast 是 fp16 舍入的来源之一，也是下面容限取 1e-2 的原因之一。fp16 的机器精度（machine epsilon）是 `2^-10 ≈ 9.8×10^-4`，单次舍入的量级大约是这个数的一半，`2^-11 ≈ 4.9×10^-4`。输出是 V 的加权平均，V 的元素量级是 O(1)，单次舍入带来的绝对误差就在 `10^-3` 量级附近。再加上参考实现在 fp32 里算完再 cast 回 fp16、两个 kernel 的累加顺序不一样，偏差攒到 `10^-2` 量级是预期内的。1e-2 这个容限不是随意定的：它比单次舍入的量级高一个数量级，给累加顺序和 cast 留了余地，又比“算错了”的误差小得多——上面那个漏乘 `alpha` 的例子误差是 13.18，尾巴没 mask 干净的例子能让分母涨 64%，真算错了，1e-2 的容限拦不住你，也不会放过你。

循环结束，`acc = acc / l_i[:, None]` 是最后的归一化：`Õ` 除以 `l`，得到标准 attention 的输出。`tl.store` 把结果写回，`mask=mask_m[:, None]` 再次拦住越界的行。整个 kernel 里，Q 只在循环外读一次：一个程序负责的 `BLOCK_M` 行 query 在整个内层循环里复用，K、V 每块读一次，S、P 全程不落 HBM——这就是分块的 payoff。反过来，如果把 `q` 的 `tl.load` 写进循环里，N=1024、`BLOCK_N=64` 时每个程序要走 16 步，8KB 的 Q 块会被重复读 16 次，每个程序多读 128KB。代码的位置决定了访存量，循环外读一次和循环内读 16 次，算出来的数完全一样，搬运量差了一个数量级。

`fused_attention` 这个包装函数里，`BLOCK_M, BLOCK_N = 64, 64` 是第 2 节算出来的那组参数里最小的一档：64×64 的 block 占 32KB，A100 上按 shared memory 算最多能驻留 5 个，留给并行度的余地最大，先拿它跑通正确性，再去试更大的。`grid = (triton.cdiv(N, BLOCK_M),)` 是程序总数，N=128 时是 2。`stride_qm` 这些参数是张量的步长（stride）：`q` 是 `[N, D]` 的连续 fp16 张量，`q.stride(0)` 等于 64，也就是一行 64 个元素、128 字节，`tl.load` 里的 `offs_m[:, None] * stride_qm` 就是靠这个步长找到每一行的起始地址。

正确性检查：与 PyTorch 参考对比（fp16 精度，最大绝对误差应在 1e-2 量级；大了先查 mask 与 rescale，不要先调容限）：

```python
def ref_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    p = torch.softmax(q.float() @ k.float().T * scale, dim=-1)
    return (p @ v.float()).to(q.dtype)

torch.manual_seed(0)
q = torch.randn(128, 64, dtype=torch.float16, device='cuda')
k = torch.randn(128, 64, dtype=torch.float16, device='cuda')
v = torch.randn(128, 64, dtype=torch.float16, device='cuda')
out = fused_attention(q, k, v)
ref = ref_attention(q, k, v)
print('max abs err:', (out.float() - ref.float()).abs().max().item())
```

如果误差大了，先查 mask 与 rescale——最常见的两个错是尾巴没 mask 干净（多出来的位置进了求和）和 alpha 乘错了地方（旧状态没 rescale）；容限是最后才动的东西。上面两个错各给了你一个数：尾巴漏一个位置，分母能涨 64%；`acc` 漏乘一次 `alpha`，输出能偏 13.18。1e-2 的容限筛得出这种错，也筛得出“只是舍入”的情况。调容限之前先问一句：误差是 `10^-2` 量级还是 `10^0` 量级？前者是精度，后者是逻辑。

**产出**：一个能跑的 fused attention（数值对齐）+ 一份“代码行 vs 推导公式”的对照笔记。

### 第 4 节 · 验收：正确性 + 计时 + Roofline 定位

**目标**：给 fused attention 一个完整的验收：对不对、快不快、卡在哪。

**怎么做**（60 分钟，需 GPU）：

验收有三个问题：对不对、快不快、卡在哪。顺序不能乱：先对、再快——拿一个没验证过的 kernel 计时，测出来的数字没有任何意义。第 3 节的正确性检查就是第一关，这一节的后两关才轮到计时和定位。

计时：用 `triton.testing.do_bench` 测 fused attention 与 PyTorch 参考（`torch.nn.functional.scaled_dot_product_attention`，SDPA）在 N=1024、d=64 下的耗时。SDPA 是对的对照：它本身就是融合实现（底层走 flash-attention 后端），比它慢是正常的——先记下差距，下一节再分析。第一版写出来就比 SDPA 快是不现实的，目标是先对、再快。还有一点容易忽略：SDPA 和你的 kernel 比的是实现质量：两个都是 fused attention，算法账是一样的，差出来的时间全在实现上（`tl.dot` 的利用率、softmax 部分的开销、循环的组织方式）。所以差距本身就是信息：它告诉你工业级实现和你这版之间隔了多少。

先算 Roofline，再看实测。标准 attention 计算量 = 4·N²·d：QK^T 占 2N²d（N×N 个输出，每个 d 次乘加），PV 占 2N²d。N=1024、d=64 时，计算量是 4×1,048,576×64 = 268,435,456 FLOPs，大约 0.268 GFLOP。Fused 版搬运量 = Q、K、V、O 各 N·d 个 fp16 元素 = 8·N·d 字节，N=1024 时是 8×1024×64 = 524,288 字节，正好 512 KiB。算术强度 = 4N²d / 8Nd = N/2，N=1024 时 AI=512。注意这个 N/2 的形式：算术强度只跟 N 有关，跟 `d` 无关。`d` 翻倍，计算量和搬运量一起翻倍，比值不动。

朴素版（物化 N×N 的 S 与 P）要多搬约 8N² 字节，N=1024 时是 8×1,048,576 = 8,388,608 字节，正好 8 MiB，是 fused 版搬运量的 16 倍。加上 fused 版自己的 512 KiB，朴素版总共要搬 8,912,896 字节，约 8.5 MiB，算术强度掉到 268,435,456 ÷ 8,912,896 ≈ 30.1 FLOP/字节。以 A100 为例（FP16 tensor core 约 312 TFLOPS，HBM 约 2 TB/s，数字以厂商公开规格为准），机器平衡点约 156 FLOP/字节——AI=512 远高于它，理论上是计算受限；朴素版的 AI≈30 远低于它，理论上是访存受限。同一个 attention，融合前后在 Roofline 图上的位置横跨了平衡点：融合做的事，就是把一个访存受限的问题变成了一个计算受限的问题。这个 16 倍就是融合省掉的东西，先算出来，再看实测。

再把这个对比换成时间下界，更直观。Fused 版如果跑满 312 TFLOPS 的算力，0.268 GFLOP 的计算量最少需要 0.86 微秒；如果只跑满 2 TB/s 的 HBM 带宽，512 KiB 的搬运量最少需要 0.26 微秒。计算的下界比搬运的下界大，瓶颈在计算这边，和 AI=512 高于平衡点 156 的判断一致。朴素版的下界在搬运这边：8.5 MiB 除以 2 TB/s 是 4.46 微秒。两个下界一比，约 5.2 倍——哪怕朴素版的实现完美无缺，它也快不过 4.46 微秒，而 fused 版的天花板在 0.86 微秒。融合省掉的 16 倍搬运，折成时间下界是 5 倍多的差距。实测时间一定比下界长，长出来的部分就是实现上的开销，下一段就是去认这部分开销。

算术强度随 N 变化，这一点决定了验收结论不能只看 N=1024 一个点。AI = N/2，N=256 时 AI=128，低于平衡点 156，同一个 kernel 在短序列下是访存受限的；N=512 时 AI=256，刚过平衡点；N=2048 时 AI=1024，N=4096 时 AI=2048。平衡点对应的序列长度是 N=312（N/2 = 156）。所以“fused attention 是计算受限的”这句话要加定语：在 N=1024 的测试点上是。在 N=256 的短序列上，结论反过来。测完 N=1024 之后，有时间可以顺手测一个 N=256 的点，看 Roofline 上的位置是不是按 N/2 的规律移动——如果两个点都落在你算的 AI 上，说明你对这个 kernel 的理解是对的。

把本周 kernel 的点标到 M0 的 Roofline 图上：横轴是你算的 AI，纵轴是实测算力。实测算力 = 计算量 ÷ 实测耗时，N=1024 时就是 268,435,456 FLOPs 除以 `do_bench` 测出来的时间。举个算术例子（非实测数字，只演示怎么换算）：如果测得耗时 10 微秒，实测算力就是 268,435,456 ÷ 10 微秒 ≈ 26.8 TFLOPS，大约是 312 TFLOPS 峰值的 8.6%。点离 Roofline 有多远，一眼就能看出来。如果点远在 Roofline 下方，说明瓶颈不在访存——理论上计算受限的 kernel 没跑出算力，差在实现上（`tl.dot` 的利用率、softmax 部分的非 matmul 开销）。`tl.dot` 之外的部分值得注意：`exp`、`max`、求和这些操作不走 tensor core，是纯粹的向量与归约开销，`tl.dot` 利用率再高也盖不住它们。第 5 周读 FA2/FA3 会告诉你工业级实现是怎么把它顶上去的。

计时的方法本身也有讲究。第 2 周讲过 `do_bench` 取中位数（median）的道理：计时抖动来自系统噪声，单次测量不可信，中位数扛得住偶尔冒出来的离群值。测 SDPA 和测你的 kernel 要用同一套方法、同一组形状、在同一个进程里跑，否则差距里混进了方法差，分不清是实现差还是测量差。另外，重复跑同一组张量时，512 KiB 的 Q、K、V、O 在两次测量之间很可能还留在片上缓存里，实测的 HBM 搬运量会比 512 KiB 的冷启动账更小——写实测记录时注明是热缓存还是冷启动，比只写一个数字有用。

LeetGPU 验收：上 [LeetGPU](https://leetgpu.com)（50+ 题，浏览器里跑，CUDA 12.8；题库见 [leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)），做至少 3 道 attention / matmul 相关题。LeetGPU 的评测会给你正确性与耗时排名——把它当成“别人写的同一道题”的对照，排名告诉你离别人的实现还有多远。排名的用法是横向对照：同一道题上，别人的实现快你多少倍、用的什么思路，比名次本身更值得看。做完把三道题的通过情况和排名记进实测记录，和 SDPA 的对比放在一起：一个是对照工业界的融合实现，一个是对照其他学习者的手写实现，两边夹着，你这版的位置就清楚了。

有两个常见误解。一个是“算术强度高就一定快”：Roofline 画的是上界，算力能不能跑出来还要看实现。AI=512 只说明访存不是瓶颈，点落在 Roofline 下方很远，就是实现在拖后腿，理论账本身没错。另一个是“比 SDPA 慢就是写错了”：SDPA 背后是调优过的工业实现，第一版比它慢是正常的，慢多少才是信息。先记下差距，下一节再分析——差距本身就是下一节的输入。

**产出**：一份实测记录（fused vs SDPA 耗时、Roofline 点、LeetGPU 3 道题通过）。

### 第 5 节 · 小结：在 Roofline 图上标点 + causal mask 练习

**目标**：收拢本周的结论，并把 causal mask 留作练习。

**怎么做**（60 分钟，推导与总结无卡可做）：

更新 Roofline 图：把 fused attention 的点标上去，并写一句话——它离 Roofline 有多远、差在哪（大概率是 `tl.dot` 的利用率与 softmax 部分的非 matmul 开销，第 5 周会展开工业级实现用什么手段把它顶上去）。标图的动作本身就是验收的收尾：横轴是你第 4 节算的 AI=512，纵轴是实测算力，点的位置把“理论上计算受限”和“实测跑出多少算力”钉在同一张图上。点如果贴着 Roofline，说明实现已经把理论潜力兑现得差不多了；点如果远在下方，说明理论账和实现账之间还有距离，距离就是第 5 周要读的东西。写那一句话的时候要诚实：差在哪是猜测还是测出来的？第 4 节的分析能支撑到哪一步就写到哪一步，撑不到的地方写“待补测”，不要把猜测写成结论。

回头看本周的链条，每一步都咬着上一步。第 1 节证明了分块不改变结果，数学上给融合开了绿灯；第 2 节算出 block 大小的 SRAM 上限，告诉你参数不能抄、得按自己的卡算；第 3 节把公式逐行翻成代码，每一行都能在推导里找到对应；第 4 节用 Roofline 和实测验收，理论账和实测账第一次碰头。链条里任何一环断了，后面的结论都站不住：rescale 没推对，代码的正确性就没根；SRAM 账没算过，block 参数就是抄来的；Roofline 没标点，“快不快”就只剩感觉。

练习：给 kernel 加 causal mask。causal 的意思是 query 第 m 行只能看到 key 的前 m 个位置——后面的位置权重必须是 0。做法是在算 `qk` 之后加一行：`qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float('inf'))`。条件 `offs_m >= offs_n` 恰好保留了 n ≤ m 的位置，其余位置被置成 -inf，exp 之后是 0，对求和与累加都没有贡献——和第 3 节的尾巴 mask 是同一个手法。为什么条件里用 `>=`？query 第 m 行能看到 key 的第 m 个位置本身——对角线上的位置是合法的，`n = m` 必须保留。写成 `>` 会把对角线也 mask 掉，整行 attention 少了一个位置，分母和输出全错一位。这是最容易错的一位，改完先数对角线。

拿 N=128、BLOCK_M=BLOCK_N=64 的尺寸数一数 mask 掉了多少。N×N 的分数矩阵一共 16,384 个位置，causal 保留的是下三角加对角线，一共 N(N+1)/2 = 8,256 个位置，mask 掉 8,128 个，占 49.6%。将近一半的位置算完就被扔掉——这是 causal 版如实算出来的账。再按 block 看，四个 64×64 的 tile 各不相同：行 64–127、列 0–63 的那块，行号最小是 64、列号最大是 63，每一行都满足 `n < m`，整块保留；行 0–63、列 64–127 的那块，行号最大是 63、列号最小是 64，每一行都满足 `n > m`，整块 mask 掉；对角线上的两块（行 0–63 列 0–63、行 64–127 列 64–127）是块内三角，一半保留一半扔掉。整块 mask 掉的那一块，这一版还是老老实实算完再扔——第 5 周读 FA2 时会看到工业实现怎么跳过这种整块，为的是把那 49.6% 的浪费省下来。

注意 mask 的形状是 `[BLOCK_M, BLOCK_N]`，与 `mask_n` 的那行 `tl.where` 可以合并成一个条件。两个条件都是“这个位置合不合法”：尾巴 mask 问的是 `offs_n < N`，causal mask 问的是 `offs_n <= offs_m`，合起来就是“列号既在序列内，又不超出行号”。两个 `tl.where` 写成一个，少一次对 `qk` 的整块遍历，意思也更清楚：合法位置的定义本来就是一句话，不必分两次说。改完之后，第 1 节的归纳法照样成立：被 mask 的位置贡献 `exp(-inf) = 0`，`l` 和 `Õ` 的求和里等于没有这一项，invariant 不变，结论不变。causal 没有改数学，只是把求和的范围从整行缩到了前 `m+1` 个位置。

改完用同样的参考实现验证（参考里用 `torch.tril` 造 mask 或直接调带 `is_causal=True` 的 SDPA）。验证要覆盖两种情况：对角线位置保留了没有，整块 mask 的区域有没有漏网。最简单的检查是拿 N=128 的随机输入跑一遍，最大绝对误差还在 1e-2 量级——如果误差突然变大，先查 `>=` 是不是写成了 `>`，再查合并后的条件是不是把尾巴 mask 挤掉了。causal 版多算的账也值得记一笔：这一版是先算满再 mask，FLOPs 还是 `4·N²·d`，N=128 时还是 4,194,304，但真正有用的只有下三角部分，2,113,536 FLOPs，约 50.4%。另一半算完就扔，是简单写法的代价；causal 本身并不要求算满再扔——区分这两点，第 5 周读 FA2 的跳块优化时才知道它在省什么。

把本周的检查清单并进总清单，加两条：“rescale 的 invariant 写出来了吗”“分块参数是算出来的还是抄的”。这两条正好是本周的两根支柱：一条管数学对不对，一条管参数有没有根据。清单并进去之后，每次写新的 fused kernel 之前，先过一遍这两条：invariant 能不能一句话写出来，block 参数能不能拿出计算表。写不出来、拿不出来，就先回去补，别急着跑。

**产出**：更新后的 Roofline 图 + causal mask 版 kernel（数值对齐）。
## 第 22 章 · 量化 Matmul 深讲：量化方案本身

> 硬件：本周需要 GPU；无卡时先做第 1 节的手算与第 2–3 节的代码编写，实测标注“待补测”。
>
> 本周的主角不是 kernel，是量化方案本身——scale 怎么定、误差从哪来、融合省什么。kernel 只是把方案落地的载体。
>
> 本周主线一句话：量化是一笔交易——用精度换搬运量；五节就是把这笔交易的定价（scale）、执行（INT8 matmul 与 fused dequant）、验收（autotune 实测与 Roofline）一笔一笔算清楚。

### 第 1 节 · Scale 怎么来：三种粒度与误差从哪来

**目标**：理解 per-tensor / per-channel / per-token 三种量化粒度的 scale 是怎么算出来的，以及误差从哪来。

**怎么做**（60 分钟，无 GPU 可做）：

先把量化压缩成一个公式。对称 INT8 量化（symmetric quantization，把一段连续的浮点区间对称地映射到整数区间）：给定一个 scale `s`，`w_q = round(w / s)`，钳制到 `[-127, 127]`，反量化 `w' = w_q · s`。scale 的取法是 `s = max|w| / 127`，量化误差的上界是 `|w − w'| ≤ s / 2`。看这个公式：scale 由最大的那个值定，误差上界由 scale 定——scale 越大，误差越大。整个量化的精度问题就归结为“怎么让 scale 尽可能小”。

为什么误差上界恰好是 `s / 2`？`round` 的舍入最多偏离半个量化单位：`w / s` 与最近的整数最多差 0.5，乘回 scale 就是 `0.5 · s`。上界的推导只有一步，但它的含义值得停一下：这个上界管的是绝对误差（absolute error），不是相对误差（relative error）。一个绝对误差 0.037 的值，如果它本身只有 0.12，相对误差就是 31%——上界没有撒谎，它保的本来就是另一种误差。后面手算的例子会让这一点现形。

为什么是 127，不是 128？int8 能表示的范围是 `[-128, 127]`，对称量化只用 `[-127, 127]`，空出 `-128` 不用。换来的是两点：0 精确对应 0，正负两边完全对称。代价是量化格点从 256 个减到 255 个，少一个格点，换来零点精确。0 这个值在神经网络里到处都是——padding、mask 掉的位置——0 量化不准会引入系统性的偏置。最大值 `max|w|` 恰好落在 127 上，反量化回来分毫不差，下面手算里 10.0 的误差就是 0，这就是 scale 定义本身的性质：被 scale 的那个最大值永远没有误差，误差全落在比它小的值身上。

三种粒度，scale 的“管辖范围”不同：
- **per-tensor**：整个张量一个 scale，`s = max|整个张量| / 127`。最简单，但一个离群值会把整个张量的 scale 顶上去。
- **per-channel**（权重常用）：每个输出通道一个 scale，`s_j = max|第 j 列| / 127`。离群值只影响它所在的通道。
- **per-token**（激活常用）：每个 token（每行）一个 scale，`s_i = max|第 i 行| / 127`。激活的分布随输入变，所以 scale 要运行时现算（dynamic）；也可以事先用校准集定好（static），省了运行时开销，但分布一变就可能失配。

为什么权重用 per-channel、激活用 per-token？权重训练完就固定了，分布不变，per-channel static 一次算好一直用；激活每次前向都不一样，分布跟着输入走，只能 per-token dynamic 现算。粒度不是随便选的，是跟着“分布变不变”走的。

dynamic 的代价可以先算一笔账。per-token dynamic 每次前向都要对激活做一次规约（求每行最大值），多一次访存与同步。以 M=1024、K=4096 的激活为例：fp16 激活本身是 1024×4096×2 字节 = 8.39MB，求每行最大值要把它完整读一遍，再写回 int8 激活 1024×4096×1 字节 = 4.19MB，量化准备这一步一共 12.58MB 的额外搬运。同一形状 FP16 matmul 的总搬运是 50.33MB（第 3 节会逐项算），量化准备就占了约 25%。static 省掉这 12.58MB，但要回答“校准集能不能代表真实输入”——省下的搬运换成了一个假设，假设错了，误差就从舍入误差变成失配误差。推理引擎里常见的做法是权重 per-channel static、激活 per-token dynamic——权重分布固定，激活分布随输入变，各取所需。

手算一个例子，体会离群值的破坏力。一行权重 `[0.1, -0.08, 0.12, 10.0, 0.09]`：per-tensor 的 scale 是 `10 / 127 ≈ 0.079`，`0.1` 量化成 `round(0.1 / 0.079) = 1`，反量化得 `0.079`——21% 的误差。如果 10.0 在别的通道，per-channel 下这一行的 scale 是 `0.12 / 127 ≈ 0.0009`，误差上界 `0.0005`。这就是为什么权重用量化几乎都用 per-channel：它把离群值关进一个通道里。

把这行数的五个元素逐个算完，对比才完整。per-tensor 下 scale `s = 10/127 ≈ 0.0787`，误差上界 `s/2 ≈ 0.0394`：0.1 量化成 1，反量化 0.0787，误差 21.3%；-0.08 量化成 -1，反量化 -0.0787，误差 1.6%；0.12 量化成 2，反量化 0.1575，误差 31.2%；10.0 量化成 127，反量化 10.0，误差 0%；0.09 量化成 1，反量化 0.0787，误差 12.5%。五个绝对误差分别是 0.021、0.0013、0.037、0、0.011，全在上界 0.039 之内——上界没有被违反。看相对误差那一列，画面就变了：最大的值误差 0%，最小的几个值误差 12%–31%。离群值定 scale 的机制决定了受害者永远是小值：scale 是按最大的人定刻度，矮个子的身高就量不准。0.12 那个元素最有说服力：绝对误差 0.037 没有超过上界 0.039，相对误差却到了 31%——同一个数，绝对误差的账本说“合规”，相对误差的账本说“没法用”。评价量化误差的时候，先问一句“绝对还是相对”，两个账本不要混着算。

per-channel 下再算一遍。这一行的 scale 是 `0.12/127 ≈ 0.000945`，误差上界 `s/2 ≈ 0.000472`：0.1 量化成 106，反量化 0.1002，误差 0.16%；-0.08 量化成 -85，反量化 -0.0803，误差 0.39%；0.12 量化成 127，反量化 0.12，误差 0%；0.09 量化成 95，反量化 0.0898，误差 0.26%。四个元素的相对误差全压到 0.5% 以下。两个粒度的误差上界一比：0.0394 除以 0.000472，约 83 倍。本例中 per-channel 把误差上界压低了约 83 倍——“把离群值关进一个通道”这句话的定价就是 83 倍。离群值越大，这个倍数背后的绝对误差差距就越大。

离群值再大 10 倍会怎样？把 10.0 换成 100.0，per-tensor 的 scale 变成 `100/127 ≈ 0.787`，误差上界 `s/2 ≈ 0.394`。0.1 除以 0.787 得 0.127，四舍五入是 0，反量化得 0，误差 100%——这个值在量化后直接消失了。离群值从 10 涨到 100，小值的命运从“量不准”变成“不存在”。误差上界随离群值线性长大（`s/2 = max|w|/254`），离群值每大 10 倍，所有小值的误差上界就大 10 倍。这个线性关系就是 per-tensor 在真实权重上站不住脚的原因：真实权重的分布总有长尾，最大值与典型值之间差一两个量级是常态。

还有一个方向的错误值得算一遍：scale 取小了会怎样？直觉说 scale 越小格点越密、精度越高，这个直觉只在不发生钳制的前提下成立。`s = max|w| / 127` 已经是“不发生钳制的最小 scale”——再小，最大值就装不下。上面那个例子里 scale 取一半（约 0.0394），10.0 除以 0.0394 得 254，四舍五入后钳制到 127，反量化得 5.0，误差 50%。这个误差不受 `s/2` 上界保护：钳制误差（clipping error）可以任意大，它不遵守舍入误差的规矩。scale 的取法卡在两个悬崖中间：大了，舍入误差 `s/2` 跟着大；小了，钳制误差没有上界。`max|w| / 127` 就是卡在悬崖边上的那个点——刚好不钳制，一点不多。所谓“scale 越小越好”，完整说法是“在不钳制的前提下，scale 越小越好”。

回头看三种粒度的选择，逻辑收拢成一句话：scale 的管辖范围越小，离群值的破坏半径就越小，代价是 scale 本身的数量与管理成本上升。per-tensor 只要 1 个 scale，per-channel 要 N 个（权重列数），per-token 要 M 个（token 数，还得每次现算）。精度与成本的权衡不在公式里，在这三个数里。下一节写 kernel 的时候，scale 的数量会直接变成访存量：per-token 的 `scale_a` 是 M 个 float，per-channel 的 `scale_b` 是 N 个 float，epilogue 里每个程序只读自己那一行一列的 scale——这就是为什么 scale 能放在 epilogue 里乘，而不用在累加前乘。

**产出**：一份三种粒度的误差直觉 + 上面的手算（离群值例子 + 误差上界），再加一句话：本例中 per-tensor 的误差上界是 per-channel 的约 83 倍，离群值越大，这个倍数背后的绝对误差差距就越大。

### 第 2 节 · INT8 `tl.dot` 路径：int8 输入、int32 累加、epilogue 再乘 scale

**目标**：写出一个 INT8 matmul：int8 输入、`tl.dot` 得 int32 累加、epilogue 里再乘回 scale。

**怎么做**（60 分钟，需 GPU 做正确性验证；无卡先写代码）：

先想清楚数据流：A（激活，per-token scale）与 B（权重，per-channel scale）都以 int8 存，`tl.dot` 做 int8×int8 → int32 累加；acc 是整数域的，scale 不能在累加前乘——必须等累加完，在 epilogue 里一次乘回：`C = acc_fp32 · s_a[:, None] · s_b[None, :]`。这一步在寄存器里完成，不额外访存。

为什么 scale 能等到最后再乘？因为 scale 不依赖求和下标 k：`s_a` 只跟行有关，`s_b` 只跟列有关，所以它们可以从 K 维的求和里提出来——`Σ_k (a_q · s_a) · (b_q · s_b) = s_a · s_b · Σ_k a_q · b_q`。能提出来，才能先整数累加、后乘 scale；提不出来（比如 scale 跟 k 有关），就得每乘一次都 dequant 一次，那是第 3 节的 fused dequant。

拿两个数验算一下这个提法。设 `s_a = 0.01`、`s_b = 0.02`，K 方向只有两项：`a = [3, -5]`，`b = [7, 2]`。先 dequant 再乘：`(3×0.01)×(7×0.02) + (-5×0.01)×(2×0.02) = 0.03×0.14 + (-0.05)×0.04 = 0.0042 - 0.002 = 0.0022`。先整数累加再乘 scale：`0.01×0.02×(3×7 + (-5)×2) = 0.0002×(21 - 10) = 0.0002×11 = 0.0022`。两边相等，提法成立。关键在第二步里 `s_a` 和 `s_b` 被一次性提到求和外面——K 方向有多少项，这个提法就省了多少次乘法：K=4096 时，逐项乘 scale 要做 4096 次浮点乘，先累加再乘只要 1 次。整数累加省的不只是访存，还有乘法次数。

反过来，什么情况下这个提法会失效？scale 跟 k 有关的时候。比如 per-group 量化（group-wise quantization，每 128 个 k 元素共享一个 scale，推理引擎里很常见的做法）下，scale 是 `s_b[k_group, n]`，求和里每一项的 scale 都不一样，提不出来，只能每组 dequant 一次。本节的 per-token × per-channel 之所以能提出来，是因为两个 scale 都不依赖 k——粒度方案决定了 kernel 的结构，上一节的粒度选择在这里收到了第一笔回报。粒度不是纸面上的分类，它直接决定 epilogue 长什么样。

量化用 PyTorch 做（这部分是数据准备，不是 kernel 本身）：

```python
def quant_per_token(x):
    # x: [M, K] fp16 -> int8 + scales [M]
    amax = x.abs().amax(dim=1)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale

def quant_per_channel(w):
    # w: [K, N] fp16 -> int8 + scales [N]
    amax = w.abs().amax(dim=0)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(w / scale[None, :]).clamp(-127, 127).to(torch.int8)
    return q, scale
```

这两段量化代码里有个容易略过的细节：`clamp_min(1e-8)`。如果某一行（或某一列）全是 0，`amax` 就是 0，scale 会是 0，后面 `x / scale` 是除以 0，得 inf 或 NaN。`clamp_min(1e-8)` 把 scale 的下限钉在 `1e-8`，全零行量化后还是全零（0 除以 `1e-8` 还是 0），scale 本身是个极小的正数，反量化 `0 × 1e-8` 还是 0。1e-8 这个数不是精度参数，是防除零的护栏——它不参与任何精度讨论，只保证代码在全零输入下不炸。顺带一提，`amax / 127.0` 里除的是 127.0 不是 128.0，对应上一节的对称量化约定：最大值落在 127 上。

INT8 matmul kernel：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_int8_kernel(a_ptr, b_ptr, c_ptr,
                       M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       scale_a_ptr, scale_b_ptr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask_c)

def matmul_int8(a_q, b_q, scale_a, scale_b):
    M, K = a_q.shape
    K2, N = b_q.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float16, device=a_q.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_int8_kernel[grid](a_q, b_q, c, M, N, K,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        scale_a.float(), scale_b.float(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
```

读这段 kernel，先对一下它与第 1 周 FP16 matmul 的异同。结构一模一样：算 `pid_m`、`pid_n`，tile 指针沿 K 维步进，`tl.dot` 累加，最后写回。变的只有三处：输入是 int8，acc 是 `tl.int32`，epilogue 多了一行乘 scale。结构不变是好事——第 1 周调通的 tile 逻辑、mask 写法、指针步进，全数复用；变的这三处就是量化方案落地的全部内容。

acc 为什么必须是 int32？算一笔账就明白。int8 的最大值是 127，两个 127 相乘是 16129；K=256 的累加最坏是 `127×127×256 = 4,129,024`，int16 的上限是 32767，4,129,024 是它的约 126 倍。更糟的是，连一个 BLOCK_K=32 的局部和都会溢出：`127×127×32 = 516,128`，是 int16 上限的约 15.8 倍。所以“先用 int16 累加、最后再转 int32”这条退路不存在——第一个 BLOCK_K 跑完 acc 就已经溢出成负数了，错误从第一步就开始累积。int32 的上限是 2,147,483,647，就算 K=4096 全长累加，最坏值也只有 `127×127×4096 = 66,064,384`，只占 int32 上限的约 3.1%，余量约 32 倍。int32 不是“保险起见”，是唯一能装下累加的类型；int16 不是“精度差一点”，是算到一半就错。类型选错与精度选错是两回事：类型错了，结果直接是错的。

epilogue 这一行也值得拆开看：`c = acc.to(tl.float32) * sa[:, None] * sb[None, :]`。`sa` 是 `tl.load(scale_a_ptr + offs_m)` 读进来的，每行一个；`sb` 是每列一个。`sa[:, None]` 把它竖过来（形状 BLOCK_M×1），`sb[None, :]` 把它横过去（1×BLOCK_N），广播（broadcasting）之后每个输出元素 `(i, j)` 恰好乘上 `sa[i] * sb[j]`。广播在这里不是语法糖，是语义本身：scale 的粒度（per-token 管行、per-channel 管列）直接写成了广播的形状。读 scale 的访存量也可以算：每个程序读 BLOCK_M + BLOCK_N 个 float，64+64=128 个，512 字节——相对 tile 的访存可以忽略。scale 的访存从来不是问题，scale 的位置才是：它必须在 epilogue，不在循环里。

注意：`tl.load` 的 mask 必须同时盖住 M/N 维与 K 维余量——只盖 K 维时，M 或 N 不是 BLOCK 倍数会在尾块越界读。第 4 节的 M=32（decode 形状）就会撞上这一点，换形状前先确认 mask 齐全。

M=32 的时候 mask 的问题会变得具体。BLOCK_M=64，M=32，tile 的 64 行里有一半是越界行——`offs_m` 的后 32 个位置指向 A 矩阵之外的内存。没有 M 维 mask，`tl.load` 会把矩阵外面的内存读进来，轻则读到别的数据（结果错得悄无声息），重则越界访问直接报错。K 维的 `k_rem` mask 只管 K 方向的尾巴，管不了 M 方向。两个 mask 是正交的（orthogonal，互相独立、缺一不可）：K mask 管“这一块 K 有没有读完”，M/N mask 管“这一行/列是不是真的存在”。第 1 周的 FP16 matmul 测的是方阵大 batch，M、N、K 全是 BLOCK 的整倍数，mask 的缺失藏得住；M=32 的 decode 形状是第一个让它藏不住的形状。写 kernel 的时候先问一句：这个形状下，哪个维度的尾块会越界？

正确性检查：与 FP16 参考对比。量化有误差，`torch.allclose` 的绝对容限不再适用——改用相对误差，并把它记下来（这是本周的精度账本）：

```python
torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16, device='cuda')
b = torch.randn(256, 128, dtype=torch.float16, device='cuda')
a_q, sa = quant_per_token(a)
b_q, sb = quant_per_channel(b)
c = matmul_int8(a_q, b_q, sa, sb)
ref = a @ b
rel_err = ((c.float() - ref.float()).abs() / ref.float().abs().clamp_min(1)).mean()
print('mean relative error:', rel_err.item())
```

为什么用相对误差？量化误差是按比例走的：值越大，绝对误差越大，固定一个绝对容限没有意义——相对误差才是“误差占结果的几分之几”。预期相对误差在 1% 量级；如果大一个数量级，先查 scale 的粒度（是不是把 per-token 与 per-channel 弄反了）与 `tl.dot` 的累加类型（必须是 int32，int16 会溢出：127×127×256 ≈ 410 万，远超 int16 的 32767）。

“按比例走”可以拿一个三元点积手算验证。取 `a = [0.5, -0.3, 0.8]`，`b = [0.4, 0.9, -0.2]`，精确点积是 `0.5×0.4 + (-0.3)×0.9 + 0.8×(-0.2) = 0.2 - 0.27 - 0.16 = -0.23`。per-tensor 量化：`s_a = 0.8/127 ≈ 0.006299`，`a` 量化成 `[79, -48, 127]`；`s_b = 0.9/127 ≈ 0.007087`，`b` 量化成 `[56, 127, -28]`。整数点积是 `79×56 + (-48)×127 + 127×(-28) = 4424 - 6096 - 3556 = -5228`，乘回 scale：`-5228 × 0.006299 × 0.007087 ≈ -0.2334`。相对误差是 `|-0.2334 - (-0.23)| / 0.23 ≈ 1.47%`——正好落在“1% 量级”的预期里。这个 1.47% 不是测出来的，是手算出来的；它说明 1% 的预期不是随意估计的数，是量化误差本身的量级：每个元素带一个 `s/2` 量级的绝对误差，K 项加总后相对误差收敛到百分之一上下。

`rel_err` 的分母里还有个细节：`ref.float().abs().clamp_min(1)`。参考值接近 0 的位置，绝对误差再小，除以一个接近 0 的分母，相对误差也会爆炸。`clamp_min(1)` 把分母的下限钉在 1，参考值小于 1 的位置按绝对误差计，参考值大于 1 的位置按相对误差计——它在两种误差口径之间划了一条线。没有这行，`rel_err` 的平均值会被少数几个接近 0 的参考值绑架，一个位置的相对误差上千，整张表的平均数就没法看了。

epilogue 乘 scale 的代价也可以先算出来，免得第 4 节实测时算错账。每个输出元素在 epilogue 里做 4 次操作：int32 转 fp32、乘 `sa`、乘 `sb`、fp32 转 fp16。M=32 时输出是 32×4096 = 131,072 个元素，一共 524,288 次操作；M=1024 时输出是 4,194,304 个元素，一共约 1678 万次操作。两档形状下，这个数都是总 FLOPs（2·M·N·K）的约 0.05%——epilogue 的乘法本身便宜得很。如果第 4 节实测发现 INT8 离 2 倍峰值很远，账不要先记在 epilogue 的乘法上，先查 int8 的 `tl.dot` 路径本身（tensor core 的 int8 利用率）与访存结构。算账的时候，先把便宜的部分排除掉，剩下的才是真凶。

一个常见的误解：“INT8 快 2 倍是因为算得快”。算得快只是半句：A100 上 INT8 tensor core 约 624 TOPS，FP16 约 312 TFLOPS，峰值确实是 2 倍。但 INT8 的另一半提速来自搬运——权重从 2 字节/元素降到 1 字节/元素，搬运量直接减半。峰值算力翻倍、搬运量减半，两件事同时发生，缺一边的解释都不完整。第 5 节会把这两边放进同一张 Roofline 图里，那时候“为什么快”才算答完。

**产出**：一个 INT8 matmul + 一份相对误差记录（数值与容限判断），再加一句：三元点积手算的相对误差约 1.47%，落在 1% 量级的预期里。

### 第 3 节 · Fused dequant+matmul：dequant 融进 matmul，不单独物化 FP16 权重

**目标**：写出 fused dequant+matmul，并与“先 dequant 再 matmul”的两步基线做访存对比。

**怎么做**（60 分钟，需 GPU；无卡先写代码）：

先想清楚两步基线的问题：权重以 int8 存，要先跑一个 dequant kernel 把它展开成 FP16（写回 HBM），再跑 FP16 matmul 把它读回来。权重每用一次就多一次“展开→写回→读回”的往返。按字节算，权重每元素要过手 5 字节（读 int8 1 字节、写 fp16 2 字节、再读 fp16 2 字节），融合版只读 1 字节。dequant 融合就是把展开放在寄存器里做：读 int8 tile，乘 scale 转成 fp16，立刻喂给 `tl.dot`，FP16 权重永远不落 HBM。

5 字节对 1 字节，按 4096×4096 的权重矩阵展开成总量更直观。权重一共 4096×4096 = 16,777,216 个元素。两步基线里，dequant kernel 读 int8 权重 16.78MB，写 fp16 权重 33.55MB，matmul 再把 fp16 权重读回来 33.55MB——权重这一项的搬运一共 83.89MB。融合版只读 int8 权重 16.78MB，一次。一笔省下 67.11MB，省掉的搬运是融合版自身搬运的 4 倍。注意省掉的是权重这个最大头的搬运：M=32 的 decode 形状下，FP16 matmul 的总搬运约 34.08MB，权重占 33.55MB（第 5 节会逐项算，占比 98.5%）——省权重的搬运，就是省几乎全部的搬运。两步基线多出来的那两次 fp16 读写，读写的还是同一份权重：先写出去，再读回来，中间没有任何别的 kernel 用它。这就是融合要消灭的往返：数据写出去又读回来，中间没有人消费它，纯粹是格式转换的过路费。

Fused kernel（A 是 fp16 激活，B 是 int8 权重 + per-channel scale）：

```python
@triton.jit
def matmul_fused_dequant_kernel(a_ptr, b_ptr, c_ptr,
                                M, N, K,
                                stride_am, stride_ak, stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                scale_b_ptr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b_int8 = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)
```

读这段 kernel，先看 dequant 发生的位置。循环里 `b_int8 = tl.load(...)` 读进来的是 int8 tile，下一行 `b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)` 在寄存器里把它展开成 fp16，紧接着 `tl.dot(a, b, acc)` 就把它消费掉。`b` 这个 fp16 tile 从生到死只活在寄存器里：它没有对应的 HBM 地址，没有 `tl.store` 写它，也没有下一个 kernel 读它。融合的本质就是这个——中间结果的生命周期缩短到一次循环迭代之内，短到不需要经过 HBM。上一节的 epilogue 乘 scale 也是同样的手法：acc 转 fp32、乘 scale、转 fp16，全在寄存器里完成。两节的差别在乘 scale 的位置：上一节 scale 在循环外乘一次，这一节 scale 在循环内每个 k 步乘一次——因为这一节的 scale 是权重 dequant 的一部分，而权重是分 k 步读进来的。

为什么 dequant 要先转 fp32 再转回 fp16，而不是直接在 fp16 里乘？`b_int8.to(tl.float32) * sb[None, :]` 这一步的乘法发生在 fp32 里：int8 的整数值（最大 127）乘上 fp32 的 scale，乘积的精度由 fp32 兜底；如果直接在 fp16 里乘，乘积要先舍入到 fp16 的精度再进 `tl.dot`，等于在进 tensor core 之前先丢一轮精度。最后的 `.to(tl.float16)` 是为了喂给 `tl.dot`——fp16 的 tensor core 要 fp16 输入。顺序是：fp32 里算准，fp16 里算快。各取所需，分工写在类型转换的顺序里。

`sb` 的加载位置也值得看一眼：`sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)` 写在循环外面，只读一次。scale 不依赖 k（上一节的提法在这里同样成立），所以它可以提到循环外；`b_int8` 依赖 k，必须在循环里分步读。循环外读一次的东西与循环内每步都读的东西，分界线就是“依不依赖 k”。scale 的访存量也可以算：N=4096 个 float，一共 16,384 字节，16KB——相对 16.78MB 的权重可以忽略。融合省的是权重的 67.11MB，不是 scale 的 16KB，账要算在大头上。

注意：这里的 `tl.load` 也要同时盖 M/N 维与 K 维余量（同第 2 节），第 4 步的 M=32 形状下尾块同样会越界。

关键点：`b_int8.to(tl.float32) * sb[None, :]` 是在寄存器里对每个元素做的——dequant 的代价是每个元素几次 ALU 操作，省的是 HBM 往返。融合的收益取决于 matmul 本来卡在哪：小 batch（decode 形状，M 很小）时 matmul 是访存受限，省搬运就是省时间；大 batch 时 matmul 是计算受限，dequant 的 ALU 开销占比上升，收益收窄。这个判断先写下来，下一节实测验证。

dequant 的 ALU 代价可以先算出来，让“占比上升”有个具体的数。每个权重元素在 dequant 里做 3 次操作：int8 转 fp32、乘 scale、fp32 转 fp16。权重一共 16,777,216 个元素，一份权重完整 dequant 一次就是约 5033 万次操作。但注意：每个权重 tile 会被 M 方向的每个程序块各 dequant 一次——权重在 HBM 里只存一份，在寄存器里却要展开 M/BLOCK_M 份。总的 dequant 操作数是 `3 × ⌈M/BLOCK_M⌉ × K × N`，占总 FLOPs（2·M·N·K）的比例是 `3 × ⌈M/BLOCK_M⌉ / (2M)`。M=32、BLOCK_M=64 时，M 方向只有一个程序块，比例是 3/(2×32) ≈ 4.7%；M=1024、BLOCK_M=64 时，16 个程序块各展开一次，比例是 3×16/(2×1024) ≈ 2.3%；BLOCK_M 换成 128，比例降到约 1.2%。M 是 BLOCK_M 的整倍数时，这个比例约等于 `3/(2·BLOCK_M)`——只由 BLOCK_M 定。decode 形状下有个反直觉的结论：BLOCK_M 从 64 换到 128 并不能摊薄 dequant，因为 M=32 时 M 方向本来就只有一个程序块，权重只被展开一次，占比由 M 定，不由 BLOCK_M 定。dequant 的摊薄要靠 M 变大，不是靠 tile 变大。

这句话可以用 Roofline 先算一遍（N=K=4096，FP16）：M=32 时算术强度约 31 FLOP/字节，低于 A100 的 FP16 平衡点约 156，访存受限；M=1024 时算术强度约 680，高于平衡点，计算受限。所以“decode 省搬运就是省时间、prefill 收益收窄”不是感觉，是 Roofline 算出来的——下一节测的就是这个预测准不准。

Roofline 的数可以逐项验算一遍，免得把它当成黑箱。算术强度（arithmetic intensity，FLOPs 除以搬运字节数）= 2·M·N·K / 总搬运字节。M=32 时，FLOPs 是 2×32×4096×4096 = 1,073,741,824（约 1.07 GFLOP）；搬运是权重 fp16 33.55MB + 激活 fp16 0.26MB + 输出 fp16 0.26MB = 34.08MB；算术强度是 1,073,741,824 / 34,078,720 ≈ 31.5。M=1024 时，FLOPs 是 34.36 GFLOP；搬运是权重 33.55MB + 激活 8.39MB + 输出 8.39MB = 50.33MB；算术强度是 34,359,738,368 / 50,331,648 ≈ 682.7。平衡点约 156 的含义是：算术强度低于 156，时间由搬运定；高于 156，时间由算力定。M=32 的 31.5 远在平衡点左边，省搬运直接换算成省时间；M=1024 的 682.7 远在平衡点右边，时间由 FLOPs 定，省下的字节换算不成时间。

融合版的点也可以先标出来。M=32 的 fused dequant：搬运是权重 int8 16.78MB + 激活 fp16 0.26MB + 输出 fp16 0.26MB = 17.30MB，算术强度是 1,073,741,824 / 17,301,504 ≈ 62.1——相对 FP16 的 31.5 向右移了一倍，但还是低于平衡点 156，依然访存受限，所以时间大约减半。M=1024 的 fused：搬运是权重 int8 16.78MB + 激活 fp16 8.39MB + 输出 fp16 8.39MB = 33.55MB，算术强度是 34,359,738,368 / 33,554,432 = 1024——远在平衡点右边，计算受限，省下的 16.78MB 字节换算不成时间，而 dequant 的 ALU（约占 FLOPs 的 2.3%）是实打实多花的。这就是“收益收窄”的机制：融合省的是搬运，搬运不再是瓶颈的时候，省下来的东西就不值钱了，而 dequant 的 ALU 账单照付。两个形状的预测正好相反，下一节一次实测就能同时验证两个。

一个常见的误解：“融合总是更快”。融合是一笔交易：拿 ALU 操作换 HBM 往返。访存受限时，HBM 往返是瓶颈，交易划算；计算受限时，瓶颈是 FLOPs，换回来的字节不值钱，付出去的 ALU 还占地方。上一节的 INT8 matmul 与这一节的 fused dequant 不是谁替代谁的关系：INT8 matmul 把激活和权重都量化了，省得更多，但要付量化误差与 epilogue 的账；fused dequant 只动权重，激活保持 fp16，误差小，但省得少。两个 kernel 的取舍，第 5 节会放进同一张表里。

融合这个手法，本周已经是第三次出现。第 2 周的 fused softmax 把减最大值、exp、求和留在寄存器里，不把中间结果写回 HBM；第 3 周的 FlashAttention 把 S、P 两个 N×N 矩阵留在 SRAM 里，不落 HBM；这一节把 dequant 的 fp16 权重留在寄存器里，不落 HBM。三次融合消灭的是同一种东西：写出去又读回来、中间没有人消费的中间结果。融合的判断标准也可以收拢成一句话：中间结果有没有别的消费者？没有，就融掉；有，就留着。

两步基线（对照用，PyTorch 一行）：`b_fp16 = (b_q.float() * sb).to(torch.float16)`，再 `torch.matmul(a, b_fp16)`。用 `triton.testing.do_bench` 测两边在 M=32（decode 形状）与 M=1024（prefill 形状）下的耗时，填表。

这行基线代码本身就是两步基线的全部成本：`(b_q.float() * sb)` 在 HBM 里物化出一份 fp16 权重（33.55MB 写），`.to(torch.float16)` 再定一次类型，`torch.matmul` 把它读回来（33.55MB 读）。一行代码，三次 HBM 往返——融合要消灭的就是这三次。测的时候注意，基线的 dequant 与 matmul 是两个 kernel，`do_bench` 要把两个加起来算，不要只测 matmul 那一半：只测一半，基线就显得比实际快，融合的收益就被低估了。

**产出**：一个 fused dequant kernel + 一份两步基线 vs 融合版的访存对比表（M=32 与 M=1024 两档），再加一句：M=32 的融合点算术强度约 62.1，仍在平衡点左边；M=1024 的融合点约 1024，已在平衡点右边——两档形状的预测正好相反，下一节一次验证两个。

### 第 4 节 · Autotune + 实测：BLOCK 配置扫一遍，`do_bench` 计时

**目标**：给第 2–3 节的两个 kernel 做 autotune，出一份配置-耗时对照表。

**怎么做**（60 分钟，需 GPU；无卡先写配置表框架，实测标注“待补测”）：

给两个 kernel 套 `@triton.autotune`，`key` 取 `['M', 'N', 'K']`（形状一变就重调——M=32 和 M=1024 想要的 BLOCK_M 不一样）。配置覆盖 `BLOCK_M × BLOCK_N × BLOCK_K` 与 `num_warps` 的组合，至少扫 4 组，例如：`(64, 64, 32, warps=4)`、`(128, 64, 32, warps=4)`、`(64, 128, 32, warps=8)`、`(128, 128, 32, warps=8)`。每组配置第一次调用时都要编译加实测，配置越多调优越久——4 组是起步，不是上限。

autotune 的 `key` 先讲透。`key=['M', 'N', 'K']` 的意思是：autotune 按形状缓存调优结果，M、N、K 任何一个变了，就重新扫一遍配置。为什么形状一变就要重调？因为最优配置跟着形状走：M=32 的 decode 形状与 M=1024 的 prefill 形状，想要的 BLOCK_M 不一样——第 3 节的 Roofline 已经说了，一个访存受限，一个计算受限，tile 的取舍没有理由相同。`key` 里不写 `BLOCK_M` 这些，是因为它们是被扫的变量，不是缓存的条件；`key` 里写的是“什么变了就要重调”，答案是形状。调优的成本也可以算：4 组配置 × 2 档形状 = 8 个（配置，形状）组合，每个组合第一次调用时编译加实测——8 次编译是调优的固定成本，配置翻倍，成本跟着翻倍。4 组是起步，不是上限；但每加一组配置，都要问一句“多扫这一组，换来的信息值不值一次编译”。

4 组配置里每个数都有分工。BLOCK_M 是 tile 的高度：M 方向一次算多少行，决定一个程序块复用 B tile 的次数——BLOCK_M 越大，读进来的 B tile 被越多的行共享。BLOCK_N 是 tile 的宽度：N 方向一次算多少列，决定 A tile 的复用次数。BLOCK_K 固定 32：K 方向每步推进的长度，32 是 int8 `tl.dot` 的常用步长（第 2 节算过，一个 BLOCK_K=32 的局部和最坏是 516,128，int32 装得下）。num_warps 是每个程序块的 warp 数：4 个 warp 是 128 个线程，8 个 warp 是 256 个线程——warp 越多，一个程序块内的并行度越高，但每个线程分到的寄存器与 shared memory 越少。4 组配置扫的其实是两个维度的取舍：tile 大小（复用 vs 占用）与 warp 数（并行度 vs 每线程资源）。没有一组配置在两个维度上同时占优，这就是要实测的原因。

两档形状下，程序块的数量先算出来——它决定了占用率的上限。grid 是 `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`，程序块总数是两个方向的乘积。M=32、BLOCK_M=64、BLOCK_N=64 时，M 方向 1 个程序块，N 方向 64 个，一共 64 个程序块；BLOCK_M 与 BLOCK_N 都换成 128，一共 32 个。A100 有 108 个 SM（NVIDIA 官方文档数据），64 个程序块只能覆盖约 59% 的 SM，32 个只能覆盖约 30%——decode 形状下，剩下的 SM 是空转的，tile 越大，空转的 SM 越多。M=1024 时情况反过来：BLOCK_M=64、BLOCK_N=64，一共 16×64 = 1024 个程序块；BLOCK_M 与 BLOCK_N 都换成 128，一共 8×32 = 256 个——两组都远超 108 个 SM，SM 全有活干，tile 大小的取舍不再受程序块数量卡脖子。decode 与 prefill 的 autotune 结论很可能相反：decode 偏爱小 tile（程序块多，SM 覆盖全），prefill 偏爱大 tile（复用高，dequant 摊薄）。这个预测先写下来，实测的时候对照着看——如果实测说 decode 也偏爱大 tile，那就要回头查：是不是小 tile 的 shared memory 或 launch 开销反而成了瓶颈。

测两档形状：M=32（decode）与 M=1024（prefill），N=K=4096。每组记三列：耗时、换算成的算力（TFLOPS = 2·M·N·K / 耗时）、与 FP16 Triton matmul 基线的比值。基线是用来回答“量化到底买到了多少速度”的——没有基线，TFLOPS 只是一个数。INT8 的理论峰值算力通常是 FP16 的 2 倍（A100 上 INT8 tensor core 约 624 TOPS，FP16 约 312 TFLOPS，数字以厂商公开规格为准），但实测能到多少取决于 `tl.dot` 的 int8 路径与 epilogue 开销（epilogue 里乘 scale、转 fp16，是 FP16 版不付的账）——先测再下结论。

TFLOPS 的换算可以先备好一张换算尺，实测的时候直接套。分子 2·M·N·K 是固定的：M=32 时是 2×32×4096×4096 = 1,073,741,824 FLOP（约 1.07 GFLOP），M=1024 时是 34,359,738,368 FLOP（约 34.36 GFLOP）。耗时以微秒计，TFLOPS = 分子 / 耗时 / 1e12：M=32 时 TFLOPS ≈ 1073.7 / t(µs)，M=1024 时 TFLOPS ≈ 34359.7 / t(µs)。换算尺备好，实测的数字一出来就能换算，不用现推公式。比值那一列也要备好分母：FP16 Triton matmul 基线要在同一形状、同一台卡上测，不要拿别的卡、别的形状的数来比——比值只在“同一形状、同一硬件、同一测量方法”下才有意义。

624 TOPS 对 312 TFLOPS 这个 2 倍是天花板，不是承诺。天花板的意思是：INT8 的算力上限是 FP16 的 2 倍，实测能到多少，取决于 int8 的 `tl.dot` 路径能不能把 tensor core 喂饱。第 2 节算过一笔相关的账：epilogue 的乘 scale 与类型转换只占总 FLOPs 的约 0.05%——如果实测离 2 倍很远，账不要先记在 epilogue 上。更可能的缺口在两处：一是 int8 tensor core 路径本身的利用率（tile 形状、访存对齐、流水线），二是 decode 形状下程序块太少、SM 覆盖不满（上面算的 59% 与 30%）。两个缺口，第 4 节的表一次能看出来：同一形状下换配置，TFLOPS 跟着动，说明缺口在配置；换什么配置都不动，说明缺口在形状本身。

验证第 3 节的判断：fused dequant 在 M=32 时应该明显快于两步基线（省了权重展开的 HBM 往返），在 M=1024 时差距收窄。如果实测与判断不符，记下来——这是第 6 周“负面结果分析”的素材。第 3 节的预测是可证伪的，测出来不对也是结果，藏起来才是浪费。

“明显快”与“差距收窄”在填表之前先定好标准，免得实测完了各说各话。M=32 时 fused 的算术强度约 62.1，仍在平衡点左边，时间由搬运定：搬运从两步基线的 83.89MB（权重那一项）降到 16.78MB，时间应该跟着降七八成——如果实测只快了一点点，先查两步基线的 dequant 是不是被 L2 缓存兜住了：33.55MB 的 fp16 权重在 40MB 的 L2 里理论上塞得下，matmul 第二次读可能命中 L2，没走 HBM——这个可能性要记下来，实测的时候对照 ncu 的 DRAM 读数判断，待补测）。M=1024 时 fused 的算术强度约 1024，已在平衡点右边，时间由 FLOPs 定：两步基线的总时间里 matmul 占大头，dequant 的 83.89MB 搬运只占一小部分，fused 省掉它，差距收窄是预期的。判断的标准写在表头下面，实测的数字填进去，对得上就是验证，对不上就是第 6 周的素材——两种结果都有去处，测之前先把去处想好。

**产出**：一份配置-耗时对照表（两档形状 × 至少 4 组配置）+ 一句“判断是否成立”的结论。表头先备好，数字待补测：

| 形状 | kernel | 配置（BLOCK_M，BLOCK_N，BLOCK_K，warps） | 耗时（待补测） | TFLOPS（待补测） | / FP16 基线（待补测） |
|---|---|---|---|---|---|
| M=32 | INT8 matmul | （64，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=32 | INT8 matmul | （128，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=32 | INT8 matmul | （64，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=32 | INT8 matmul | （128，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=32 | fused dequant | （64，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=32 | fused dequant | （128，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=32 | fused dequant | （64，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=32 | fused dequant | （128，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=32 | 两步基线 | — | 待补测 | 待补测 | 待补测 |
| M=1024 | INT8 matmul | （64，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=1024 | INT8 matmul | （128，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=1024 | INT8 matmul | （64，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=1024 | INT8 matmul | （128，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=1024 | fused dequant | （64，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=1024 | fused dequant | （128，64，32，4） | 待补测 | 待补测 | 待补测 |
| M=1024 | fused dequant | （64，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=1024 | fused dequant | （128，128，32，8） | 待补测 | 待补测 | 待补测 |
| M=1024 | 两步基线 | — | 待补测 | 待补测 | 待补测 |

### 第 5 节 · 小结：量化省的是什么（搬运量），代价是什么（精度），在 Roofline 图上标点

**目标**：收拢本周的结论，把两个 kernel 的 Roofline 点标到图上。

**怎么做**（60 分钟，总结无卡可做）：

算术强度对比：同一个 4096³ matmul，FP16 版搬运量是权重与激活各 2 字节/元素，INT8 版权重只剩 1 字节/元素——量化首先省的是搬运量，算术强度向右移；同时 int8 的 `tl.dot` 峰值算力更高，Roofline 的顶也抬高了。两边一起动，这就是量化提速的完整解释。省多少取决于权重占搬运的比重：M=32 的 decode 形状下，FP16 版总搬运约 34MB 里权重占 33.5MB——九成八以上，INT8 把总搬运几乎减半；方阵大 batch 下激活和输出也占大头，省的比例就小。量化最划算的形状，是权重占主导的形状。

“向右移”与“顶抬高”这两句话可以落实成八个点，标在同一张 Roofline 图上。先算方阵 4096³ 这一档：FP16 版搬运是权重、激活、输出各 4096×4096×2 字节 = 33.55MB，三项一共 100.66MB，FLOPs 是 2×4096³ = 137.44 GFLOP，算术强度是 137,438,953,472 / 100,663,296 ≈ 1365；INT8 版（激活与权重 int8、输出 fp16）搬运是 16.78 + 16.78 + 33.55 = 67.11MB，算术强度是 137,438,953,472 / 67,108,864 = 2048。两个点都在平衡点 156 的右边，都是计算受限——方阵大 batch 下，量化省的 33.55MB 字节换算不成时间，提速只能来自顶的抬高（312 → 624）。再算 decode 这一档：M=32 的 FP16 点算术强度约 31.5，INT8 点约 62.5，fused 点约 62.1——三个点全在平衡点左边，都是访存受限，提速来自搬运减半（总搬运 34.08MB → INT8 的 17.17MB，约 0.504 倍）。同一张图上，两档形状的提速机制正好相反：decode 靠向右移，方阵靠顶抬高。Roofline 图的价值就在这里：它把“为什么快”拆成两个方向，哪个形状靠哪边，一眼就能看出来。

fused 点与 INT8 点在 M=32 时几乎重合（62.1 对 62.5），这个重合本身就是一道检查题。两个 kernel 的搬运只差激活那一项：fused 的激活是 fp16（0.26MB），INT8 的激活是 int8（0.13MB），差 0.13MB；总搬运 17.30MB 对 17.17MB，差 0.8%。点重合说明 fused 与 INT8 在 decode 形状下的搬运账几乎一样，差别在别处：fused 的激活没有量化误差，INT8 的激活有。搬运账一样、精度账不一样——这就是第 3 节说的“fused 省得少、误差小”的定量版。

省多少取决于权重占搬运的比重，这句话也可以落实成三个数。M=32 时，FP16 总搬运 34.08MB 里权重占 33.55MB，占比 98.5%——九成八以上，INT8 把总搬运压到 17.17MB，几乎减半。M=1024 时，FP16 总搬运 50.33MB 里权重占 33.55MB，占比 66.7%，激活与输出占了剩下的 33.3%，INT8 总搬运 29.36MB，是 FP16 的约 0.583 倍——省了，但没有减半。方阵 4096³ 时，权重只占 FP16 总搬运的 33.3%，INT8 总搬运 67.11MB，是 FP16 的约 0.667 倍。权重占比从 98.5% 掉到 66.7% 再掉到 33.3%，省的比例跟着从 0.504 倍走到 0.583 倍再走到 0.667 倍。“量化最划算的形状，是权重占主导的形状”——这句话的三个数就是 98.5%、66.7%、33.3%。decode 之所以是量化的主场，不是因为它小，是因为它的搬运里权重占了九成八。

代价：精度。第 2 节的相对误差就是代价的度量。把“相对误差”与“提速比”并排放进同一张表——量化方案的取舍永远是这两个数的权衡，不是单看速度。快 2 倍、误差 5% 的方案可能没法上线；快 1.2 倍、误差 0.5% 的方案可能就上线了。数字说了算。

这张表的两列数字来源不一样，要分清楚。相对误差那一列是第 2 节的精度账本：三元点积手算约 1.47%，预期在 1% 量级——这是本周唯一能量化精度的实测口径（手算也是实测的一种）。提速比那一列是第 4 节的配置-耗时对照表：INT8 matmul 与 fused dequant 各自除以 FP16 基线——实测标注“待补测”，表先备好，数字后填。两个数并排之后，取舍才有得谈：快 2 倍、误差 5% 的方案可能没法上线；快 1.2 倍、误差 0.5% 的方案可能就上线了。这两个例子里的数字是“可能”的举例，不是本周的实测结论——本周的实测结论要等第 4 节的表填满。取舍的判断标准也可以先定下来：误差的容忍线由下游任务定（分类任务与生成任务的容忍线不一样），提速比的及格线由部署定（省下来的时间值不值一次上线评审）。标准先定，数字后填，填完再下结论——顺序不要反过来。

更新 Roofline 图：把 INT8 matmul 与 fused dequant 的点标上去，并写一句话——它们离 Roofline 有多远、差在哪。

标图的时候，八个点一次标全，免得漏掉对照组：

| 点 | 形状 | 算术强度（FLOP/字节） | 区间 | 顶 |
|---|---|---|---|---|
| FP16 matmul | M=32 | 约 31.5 | 访存受限 | 312 TFLOPS |
| INT8 matmul | M=32 | 约 62.5 | 访存受限 | 624 TOPS |
| fused dequant | M=32 | 约 62.1 | 访存受限 | 312 TFLOPS |
| FP16 matmul | M=1024 | 约 682.7 | 计算受限 | 312 TFLOPS |
| INT8 matmul | M=1024 | 约 1170.3 | 计算受限 | 624 TOPS |
| fused dequant | M=1024 | 约 1024.0 | 计算受限 | 312 TFLOPS |
| FP16 matmul | 4096³ 方阵 | 约 1365.3 | 计算受限 | 312 TFLOPS |
| INT8 matmul | 4096³ 方阵 | 约 2048.0 | 计算受限 | 624 TOPS |

“离 Roofline 有多远”那句话，实测之前只能写一半：点的横坐标（算术强度）是算出来的，纵坐标（实测算力）要等第 4 节的表。横坐标先标上去，纵坐标待补测——图上先有位置，后有高度。差在哪的判断也分两档写：访存受限的三个点（M=32），差在搬运量与带宽利用率；计算受限的五个点，差在 tensor core 利用率与 tile 配置。第 4 节的实测数字填进来之后，这句话才能写完。

把本周的检查清单并进总清单，加两条：“scale 的粒度与误差上界能手算吗”“dequant 融合的收益在什么形状下成立、什么形状下收窄”。

这两条检查项各对应本周的一次手算。第一条对应第 1 节的离群值例子：不看笔记，能不能现场算出 per-tensor 的 scale 约 0.079、0.1 的误差 21%、per-channel 的误差上界 0.0005？算不出来，说明粒度的直觉还没长在手上。第二条对应第 3 节的 Roofline 预测：能不能说出 M=32 时融合省时间（算术强度 31.5 → 62.1，仍在平衡点左边）、M=1024 时收益收窄（算术强度 682.7 → 1024，已在平衡点右边）？说不出来，说明融合的判断还停留在“感觉融合更快”的层面。两条都过，本周才算收拢。

**产出**：更新后的 Roofline 图 + 一张“相对误差 vs 提速比”对照表。相对误差一列来自第 2 节（手算约 1.47%，预期 1% 量级）；提速比一列来自第 4 节（待补测）。
## 第 23 章 · FlashAttention 2/3 与 CUTLASS：读工业级实现

> 硬件：以读代码为主，无卡可做；第 4 节改配置实测需要 GPU（有 nvcc 的机器）。
>
> 第 3 周你写了一个能跑的 attention，第 4 周你写了量化的 matmul。这一周不写新 kernel——读别人写的工业级实现，看它们把第 3 周的点离 Roofline 的差距是怎么补上的。
>
> 本周主线：每一代工业实现都在回答同一个问题——利用率（实际算力占理论峰值的比例）为什么上不去，补上之后数字变成多少。

### 第 1 节 · FlashAttention 1→2→3 的工程演进

**目标**：讲清三篇论文是一条链：每一代都先点名上一代的瓶颈，再改它。学完能说出每一代“改了什么、为什么改、改出多少”。

**怎么做**（60 分钟，无 GPU 可做）：

先把链条立起来（第 3 周已讲 FA1 的 IO-aware 与 tiling，这里只补它留下的尾巴）：FA1 只解决了访存，没解决算力利用率。利用率这个词值得先抠一下：它等于实测 FLOPs 除以理论峰值 FLOPs，是一个比值。FA2 的摘要直接点名：FA1 只达到理论峰值 FLOPs 的 25–40%，远低于优化过的 GEMM。25–40% 是什么概念？拿区间中点算一道题：（25 + 40）÷ 2 = 32.5%，也就是说 FA1 平均只用出了三分之一左右的算力，三分之二的 Tensor Core 时间在等数据、等同步、等别的 warp。数学对了，利用率还是低——瓶颈从算法挪到了并行结构上。

为什么数学对了还会低？第 3 周的 online softmax 推导证明的是正确性：分块算出的结果和标准 attention 相等。但正确性只保证结果对，不保证硬件忙。硬件忙不忙，取决于工作怎么分：一个 thread block（线程块，一组跑在同一个 SM 上的线程）算多少、一个 warp（32 个线程的执行单位）算多少、warp 之间要不要等对方把数据写进 shared memory（片上共享内存，同一 block 内线程共用的高速暂存）再读出来。分工不合理，Tensor Core 就算得断断续续，利用率自然上不去。FA1 的贡献在访存，FA2 的贡献就在分工。

FA2 的三处改动（按摘要的原意），每一处都可以对着“利用率低”这个诊断去理解。

减少非 matmul 的 FLOPs：softmax、rescale 这些非矩阵乘的开销占比降下来，让 matmul 的比重上去。为什么这能提利用率？Tensor Core 只吃 matmul，softmax 和 rescale 跑在 CUDA core（通用计算单元）上。一个 attention 前向里，matmul 的 FLOPs 本来就占大头，但 softmax、rescale 这些逐元素操作如果反复做、每次都走一遍完整的读写，就会把流水线拖慢：Tensor Core 算完一块 QK^T，得等 CUDA core 把 softmax 算完才能继续。把非 matmul 的 FLOPs 压下去，等于把 Tensor Core 的等待时间压下去，同样的时间里 matmul 的占比上去了，利用率也就上去了。打个比方：一条流水线上真正赚钱的工位是 matmul，非 matmul 工位是绕不过去的质检，质检越快，赚钱工位的空闲就越少。

跨 thread block 并行：哪怕是单个 head 的 attention，也拆到多个 thread block 上并行，提高 occupancy（占用率，SM 上同时驻留的 warp 占硬件上限的比例）。单个 head 的 attention 本来可以只交给一个 block 算，但一个 head 的计算量再大，一个 block 也只占一个 SM，别的 SM 全闲着。拆到多个 block 上，多个 SM 同时开工，总的并行度上去了。occupancy 高低直接决定延迟能不能被藏住：第 1 周讲过，一个 warp 等数据的时候，SM 要切到另一个 warp 去算，驻留的 warp 越多，能顶上去的就越多。跨 block 并行就是给 SM 多备几个能随时顶上去的 warp。

warp 间分工：同一个 block 内把工作分给不同的 warp，减少 warp 之间经由 shared memory 的通信。原来一个 warp 算完要把中间结果写进 shared memory，另一个 warp 再读出来算下一步，一写一读之间还要同步。分工改了之后，每个 warp 手里的活更完整，中间结果尽量留在寄存器里，shared memory 的往返少了，同步的次数也少了。shared memory 的读写本身不慢，慢的是“写完要等别人读”这个同步点：同步点一多，所有 warp 都在等最慢的那个。

FA2 的结果（摘要数字）：相对 FA1 提速约 2 倍，A100 上达到理论峰值的 50–73%，接近 GEMM 的效率；端到端训练 GPT 风格模型时，单张 A100 达到 225 TFLOPs/s（72% 的模型 FLOPs 利用率）。这几个数可以互相验算。先看区间中点：（50 + 73）÷ 2 = 61.5%，FA1 的中点是 32.5%，61.5 ÷ 32.5 ≈ 1.89，约等于 2，和“相对 FA1 约 2 倍提速”对得上。再看端到端那个数：225 ÷ 0.72 ≈ 312.5 TFLOPs/s，这是摘要数字反推出的 A100 理论峰值——225 TFLOPs/s 是 72% 的利用率，峰值就是 225 除以 0.72。两个独立的数字（50–73% 的区间和 225 TFLOPs/s 配 72%）讲的是同一件事：FA2 把利用率从三分之一左右抬到了六成以上。还有一个细节值得留意：50–73% 是一个区间，不是单点，因为利用率随序列长度、head 维这些配置变化；读论文数字时，区间比单点诚实，单点往往是挑过的最好情况。

> **论文选读（不读也能学完本周）**：FlashAttention-2（Dao，[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)，2023 年 7 月；原文可选，正文已包含核心内容）。
>
> - **核心思想**：FA1 的问题不在算法数学，在**并行结构**。同样的 tiling 与 online softmax，换一种工作分配就能多榨一倍性能。
> - **核心逻辑链条**：诊断（25–40% 利用率 → 分工问题）→ 三处改动（减非 matmul FLOPs、跨 block 并行、warp 间分工）→ 验证（50–73% 利用率、端到端 225 TFLOPs/s）。
> - **关键结论与数字**：相对 FA1 约 2 倍提速；A100 上 50–73% 理论峰值；GPT 风格模型端到端训练 225 TFLOPs/s（72% 利用率）。摘要同时确认了 FA1 的基线：相对优化过的基线 2–4 倍提速、显存从平方降到线性、无近似。
> - **为什么重要**：它证明了“瓶颈诊断”本身的价值——第 1 周的 ncu 定位、第 3 周的 Roofline，都是在练同一种能力：先找到利用率低的原因，再决定改哪里。

FA3 的转向：FA2 在 H100 上只有 35% 利用率——Hopper 有了新硬件能力（Tensor Core 异步、TMA），FA2 没用上。35% 这个起点要和 FA2 在 A100 上的 50–73% 对着看：同一套代码，换一块更强的卡，利用率反而掉了。这说明瓶颈跟着硬件走了：H100 的算力涨得比 FA2 的搬运与同步能力快，旧结构喂不饱新硬件。第 1 周“先看硬件再写 kernel”的道理，在这里是论文级别的演示：硬件升级不会自动带来提速，结构不跟着改，利用率反而会跌。

FA3 的三项技术（摘要原意），每一项都是冲着 Hopper 的新能力去的。

warp specialization（warp 特化，让不同的 warp 专门干不同的活）+ 异步：利用 Tensor Core 与 TMA（Tensor Memory Accelerator，张量内存加速器，Hopper 上专门负责搬运的硬件单元）的异步性，让计算与数据搬运重叠——一部分 warp 专做搬运（生产者），一部分专做计算（消费者），流水线不断。关键在“专做”：同一个 warp 不再又搬数据又算，搬运和计算各自有人管，TMA 异步搬运时 tensor core 不用等。为什么“专做”比“轮流做”好？一个 warp 又搬又算，搬的时候算的单元闲着，算的时候搬的通路闲着，两种闲置轮流发生；分开之后，搬运 warp 只管把下一块 tile 提前搬进 shared memory，计算 warp 只管在 Tensor Core 上算，两边同时进行，闲置就被重叠吃掉了。重叠能省多少，可以拿一组假设数字演示（假设数字，只为讲清机制，不代表实测）：一块 tile 的搬运要 6 个单位时间，计算要 10 个单位时间，串行是 6 + 10 = 16，重叠之后每块 tile 只要 max（6，10）= 10，16 ÷ 10 = 1.6 倍——这就是流水线重叠的算法，省掉的是“等”的时间。

matmul 与 softmax 交错：把分块 matmul 与 softmax 操作交错排布，不让 Tensor Core 等 softmax。softmax 不是 matmul，跑在 CUDA core 上；流水线串行排布时，tensor core 在 softmax 阶段是闲着的，交错排布就是给它找活干。同样拿一组假设数字演示：每步 matmul 要 10 个单位时间，softmax 要 4 个单位时间，串行是 10 + 4 = 14；交错之后，上一块的 softmax 和下一块的 matmul 同时进行，理想情况下每步只要 10，14 ÷ 10 = 1.4 倍。两组假设数字都不代表 FA3 的实测提速，它们只说明一件事：FA3 的提速来自把“等”的时间从流水线里抠掉，单步本身并没有变快。FA2 减非 matmul FLOPs 是把 softmax 的活变少，FA3 的交错是把剩下的 softmax 藏到 matmul 的时间里——两代的思路是接力关系。

FP8 的块量化与 incoherent processing（非相干处理，一种打散数据相关性以控制量化误差的手段）：用硬件的 FP8 支持做低精度计算，同时控制误差。FP8 的算力是 FP16 的约 2 倍，但量化误差更大，块量化和 incoherent processing 是把误差控制住的手段。块量化好理解：每块各自定 scale（缩放因子）来量化，一个全局 scale 不再管整张量，数值范围差异大的块不会被同一个 scale 拖垮；incoherent processing 的直觉是把数据的结构打散，让量化误差均匀摊开，不集中在少数几个数值上。低精度真正的风险不是“精度低了”，是误差集中爆发——控制手段的作用就是把误差摊薄。

FA3 的结果（摘要数字）：H100 上相对 FA2 提速 1.5–2.0 倍；FP16 达到 740 TFLOPs/s（75% 利用率）；FP8 接近 1.2 PFLOPs/s；FP8 版的数值误差比基线 FP8 attention 低 2.6 倍。注意最后一项：FP8 不仅更快，误差反而更低——低精度不等于低质量，关键在误差控制的手段。这几个数也可以互相验算。740 ÷ 0.75 ≈ 986.7 TFLOPs/s，这是摘要数字反推出的 H100 FP16 理论峰值。再看起点：75% ÷ 35% ≈ 2.14，略高于 1.5–2.0 倍的提速区间——这并不矛盾，因为利用率是峰值占比，提速是实测耗时比，两者口径不同，不能直接划等号；读论文时把不同口径的数字直接相除，本身就是常见的误读。FP8 的账也值得算一笔：1.2 PFLOPs/s 就是 1200 TFLOPs/s，1200 ÷ 740 ≈ 1.62，即 FP8 版比 FP16 版快约 1.62 倍；而 FP8 的理论算力是 FP16 的约 2 倍，1200 ÷（740 × 2）≈ 0.81，说明 FP8 只兑现了理论翻倍的约 81%，剩下的 19% 被量化、搬运这些额外开销吃掉了——低精度翻倍的是峰值，不是实测。误差的账同样可以算：误差低 2.6 倍，就是新误差只有基线的 1 ÷ 2.6 ≈ 0.385，即约 38.5%，误差降了约 61.5%。“更快且误差更低”这句话，数字上就是 1.62 倍速度配 38.5% 的误差。

> **论文选读（不读也能学完本周）**：FlashAttention-3（Shah 等，[arXiv:2407.08608](https://arxiv.org/abs/2407.08608)，2024 年 7 月；原文可选，正文已包含核心内容）。
>
> - **核心思想**：新硬件要有新打法。Hopper 的异步能力（TMA、warp specialization）与 FP8 支持，是 FA1/FA2 的结构用不上的——FA3 围绕这两点重排了流水线。
> - **核心逻辑链条**：诊断（FA2 在 H100 上仅 35% 利用率）→ 三项技术（warp specialization 重叠计算与搬运、matmul/softmax 交错、FP8 块量化）→ 验证（FP16 740 TFLOPs/s、FP8 近 1.2 PFLOPs/s、误差反而更低）。
> - **关键结论与数字**：H100 上 1.5–2.0 倍于 FA2；FP16 740 TFLOPs/s（75% 利用率）；FP8 接近 1.2 PFLOPs/s；FP8 数值误差比基线低 2.6 倍。
> - **为什么重要**：它把“硬件特性决定算法结构”这句话落到了实处——第 1 周学的“先看硬件再写 kernel”，在这里是论文级别的实践。M3 会回到这个主题：换一块芯片（Ascend），同样的重排要再做一遍。

把三代画成一张演进表：每代一行，写“瓶颈诊断 → 改动 → 结果数字”。这是本周的核心产出，第 5 节会再用它。填好的样子：

| 代 | 瓶颈诊断 | 改动 | 结果数字 |
|---|---|---|---|
| FA1 | 访存：S、P 矩阵反复过 HBM | tiling + online softmax，IO-aware | 显存从平方降到线性；GPT-2 训练 3 倍 |
| FA2 | 并行结构：利用率只有 25–40% | 减非 matmul FLOPs、跨 block 并行、warp 间分工 | 相对 FA1 约 2 倍；A100 上 50–73% 利用率 |
| FA3 | 新硬件没用上：H100 上只有 35% | warp specialization、matmul/softmax 交错、FP8 | 相对 FA2 1.5–2.0 倍；FP16 740 TFLOPs/s（75%） |

读这张表要看出的规律：每一代的“改动”都是上一代“瓶颈诊断”的直接回答，没有一代是在漫无目的地优化。先诊断、再改动、最后用数字验证——这就是第 1 周 ncu 定位、第 3 周 Roofline 的同一套方法，只是搬到了论文级别。还有一个容易忽略的规律：三代改的东西一层比一层靠近硬件。FA1 改的是访存模式（算法层），FA2 改的是并行分工（执行层），FA3 改的是流水线与硬件特性的用法（硬件层）。越往后，改动离数学越远，离硬件越近——这也是第 3 周你写的 Triton 版和第 5 周读的工业版之间的差距所在：Triton 帮你管了访存和分块，工业版还要亲手管 warp 分工和流水线重叠。

**产出**：一份“论文做法 vs 工程做法”对照笔记，核心是一张 FA1→FA2→FA3 演进表（每代的瓶颈、改动、数字）。

### 第 2 节 · 读 flash-attn 源码：主循环长什么样

**目标**：在真实实现里定位第 3 周学过的 online softmax 循环，画出前向 kernel 的结构图。

**怎么做**（60 分钟，无 GPU 可做）：

先说为什么读源码：第 3 周的 kernel 和 flash-attn 解的是同一道题，差别就是 FA2/FA3 的工程手段。读源码不是为了背实现，是为了看清“你的版本和工业版差在哪”——差的地方，就是第 1 节那张演进表里的改动落点。第 3 周你写的 Triton 版，编译器帮你处理了分块、流水线和内存层级；flash-attn 的 CUDA 版把这些全部摊开写在代码里。读它等于把编译器替你做的事重新看一遍，看完就知道第 1 节那张演进表里的每一行改动在代码里长什么样。

打开 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)，定位这三个文件：`csrc/flash_attn/src/flash_fwd_kernel.h`（前向主 kernel）、`csrc/flash_attn/src/flash_fwd_launch_template.h`（按 head 维与数据类型选 kernel 变体的分发模板）、以及按 head 维/数据类型实例化的文件（如 `flash_fwd_hdim64_fp16_sm80.cu`）。读的顺序反过来：先看 launch template——它告诉你给定 head 维和数据类型，实际跑的是哪个 kernel 变体；再进 kernel 本体；先看循环骨架，再看头尾细节。为什么顺序要反过来？因为工业代码是“先分发、后计算”的结构：同一个算法针对不同的 head 维、不同的数据类型、不同的 GPU 代际，会实例化出很多个 kernel 变体，直接钻进某一个 `.cu` 文件，你不知道它代表的是哪种情况；先看 launch template，等于先看目录再看正文，知道每个变体管的是哪种输入。

在 `flash_fwd_kernel.h` 里找第 3 周的对应物：外层按 K/V 分块的循环、每步的 `m` / `l` / `acc` 更新（online softmax 的 rescale）、Q/K/V 的 tile 加载。代码是 CUDA，数学是第 3 周的推导——你读源码时要做的就是把每一行 CUDA 对回推导里的一行公式。对上了，说明第 3 周的东西是真的；对不上，先怀疑自己读错了位置，再怀疑实现用了别的等价形式。对照的方法可以具体一点：第 3 周样稿里的更新规则是 `α = exp(m_old − m_new)`，`l_new = l_old · α + rowsum(P_block)`，`Õ_new = Õ_old · α + P_block · V_block`。在 kernel 里找这三件事：`m` 是不是每来一块就取新最大值，`l` 是不是乘了 `exp(m_old − m_new)` 再加块内和，累加器 `acc` 是不是同样先乘 `α` 再加 `P_block · V_block`。三个对上两个，骨架就确认了；第三个如果长得不一样，很可能是实现把 rescale 和 matmul 的顺序换了——比如先算 `P_block · V_block` 再统一乘 `α`，数学等价，只是写法不同。Q 的加载也值得留意：Q 只加载一次、驻留在寄存器里反复用，K、V 则是每步换一块——这正是第 3 周 tiling 的结构：外层循环走 K/V，Q 不动。读到 Q 的加载只出现一次、K/V 的加载出现在循环里，结构图就对上了。

再找 FA2 的痕迹：工作是怎么在 thread block 与 warp 之间分配的（第 1 节的三处改动在代码里落在哪些地方）。不用全读懂，定位到位置并写下来就是本节的目标。定位有三个抓手：跨 block 并行看 block 的划分方式——单个 head 的 attention 是不是拆到了多个 block 上，每个 block 算哪几行输出；warp 间分工看同一个 block 内的 warp 是不是各算各的一块、中间结果经不经过 shared memory；减非 matmul FLOPs 看 softmax 与 rescale 的代码是不是被收敛到了最少的几步，有没有把能合并的逐元素操作合并到 matmul 的间隙里。三个抓手对应第 1 节的三处改动，定位到一个就是一处落点。读的时候可以带着第 1 节的演进表去找：表里每一行改动，都应该在代码里有一个对应的位置，找不到的先记下来，到第 5 节再回头对。

注意边界：flash-attn 是工业级实现，里面有大量为特定硬件与 head 维特化的分支——读源码的目标是“定位结构”，不是“背下实现”。遇到看不懂的分支，先记下它在管什么（比如某个 `#ifdef` 是为哪代 GPU 开的），不要卡住。特化分支的存在本身就是一个知识点：head 维 64 和 head 维 128 的 kernel 是不同的实例化文件，文件名里的 `hdim64`、`sm80` 就是分发的条件——head 维和 GPU 代际（sm80 对应 Ampere 架构）决定了用哪个变体。工业代码里这种分支越多，说明算法对硬件形状越敏感；第 3 周的 Triton 版没有这些分支，是因为编译器把特化藏起来了，不是特化不存在。

一个常见的误解：“读不懂源码就是基础差”——读工业级 CUDA 代码卡住是常态，卡住的位置往往是特化分支和模板技巧，不是 attention 的数学。正确的做法是先画骨架再抠细节：外层循环、rescale 更新、tile 加载这三件确认了，主循环就读完了；剩下的分支是“这份代码为了跑得快多做的事”，不是“这道题必须有的事”。反过来，把骨架和分支混在一起读，每一行都像在讲新东西，读一小时也画不出结构图。

读完之后怎么验证自己真的读懂了？合上代码，凭记忆回答四个问题：Q、K、V 各自在什么位置加载，Q 加载几次、K/V 加载几次；`m` / `l` / `acc` 的更新发生在循环内的第几步，rescale 乘在哪些量上；外层循环走完一块之后，哪些量留到了下一块、哪些量被写回；FA2 的三处改动里，至少一处在代码里的落点在哪。四个问题答得出来，结构图就是你自己的；答不上来，说明刚才读的是文字，不是结构。这四个问题也可以当成结构图的检查清单：图上每个框都应当能回答其中至少一个问题，答不上任何问题的框就是装饰，应当删掉。最后把这份结构图和第 3 节的 CUTLASS 三级结构图并排看：flash-attn 的分发模板按 head 维和数据类型选 kernel 变体，CUTLASS 的模板参数按 tile 形状生成 kernel，两边都在做同一件事——把“针对某种输入形状特化”的决定交给编译期，运行期的 kernel 只负责算。读懂了这一层，第 3 节读 CUTLASS 模板参数时就不会陌生。

和第 3 周的联系可以直接写进结构图：外层循环对应第 3 周的 K/V 分块，`m` / `l` / `acc` 对应 online softmax 的三个运行量，Q 常驻、K/V 轮换对应 tiling 的数据流向。第 1 节的演进表也可以画进去：在结构图的边上标出 FA2 的三处改动各落在哪个位置——跨 block 并行落在 block 划分上，warp 分工落在 block 内的 warp 划分上，减非 matmul FLOPs 落在 softmax 与 rescale 的写法上。一张图同时装下数学结构和工程改动，就是本节要的产出。

**产出**：一张前向 kernel 的结构图（循环、分块、rescale 更新、warp 分工的位置）。

### 第 3 节 · CUTLASS 三级结构：threadblock / warp / thread + MMA pipeline

**目标**：能画出 CUTLASS GEMM 的三级分工与数据流向，知道每一级在管什么。

**怎么做**（60 分钟，无 GPU 可做）：

打开 [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)，读 `examples/00_basic_gemm/basic_gemm.cu`：它用 `cutlass::gemm::device::Gemm<float, ColumnMajor, ...>` 搭一个 GEMM，默认 threadblock tile 是 128×128×8；调用方式是先构造 `CutlassGemm::Arguments` 再调 `gemm_operator(args)`；文件里还带了一个 naive 参考 GEMM 供对照。先把这个最小例子跑通结构：输入怎么进、tile 怎么分、结果怎么出。naive 参考 GEMM 的作用是当“对照组”：同样的输入，naive 版逐元素算，CUTLASS 版分块算，结果应当一致；对照组存在的意义是让你改配置之后有东西可比——第 4 节改 tile 配重编之后，第一件事就是拿 naive 版验算结果对不对，再谈快慢。

先把默认 tile 128×128×8 的账算清楚，这是理解三级结构最实在的入口。128×128×8 的意思是：一个 threadblock 一次算输出矩阵里 128 行 × 128 列的一块，K 维每次推进 8。A 的一块 tile 是 128×8 = 1024 个元素，B 的一块 tile 是 8×128 = 1024 个元素，两块加起来 2048 个元素；例子用的是 float（单精度浮点，4 字节），2048 × 4 = 8192 字节，即 8 KB——这就是一个 K 步里从 global memory（全局内存，GPU 上容量最大、速度最慢的那层）搬进 shared memory 的数据量。这一 K 步的计算量是 2 × 128 × 128 × 8 = 262144 次浮点操作（乘加各算一次，所以乘 2），262144 ÷ 8192 = 32，即每从 global memory 搬 1 字节，就做 32 次浮点操作——这个比值就是这一 tile 的算术强度（每字节搬运对应的计算量），第 1 周 Roofline 的语言在这里直接用得上。输出的一块 tile 是 128 × 128 = 16384 个元素。K 维如果一共是 4096，K tile 是 8，外层循环就要走 4096 ÷ 8 = 512 步；每一块输出 tile 都要经过这 512 步的累加。再算 block 的数量：一个 4096×4096 的输出矩阵，用 128×128 的 tile 切，（4096 ÷ 128）×（4096 ÷ 128）= 32 × 32 = 1024 个 block。1024 个 block、每块 512 步、每步 8 KB 搬运和 262144 次操作——这就是默认配置下一张 GEMM 的完整工作量画像。

三级分工（对照例子里的模板参数）：
- **threadblock 级**：一个 block 算输出矩阵的一块 tile（默认 128×128），负责从 global memory 把 A/B tile 搬进 shared memory；
- **warp 级**：一个 block 内的多个 warp 再分 tile（WarpShape），每个 warp 算自己的一块；
- **thread 级**：每个线程算一小块（InstructionShape），最终落到 Tensor Core 的 MMA 指令上。
- 数据流向：global → shared（block 级搬运）→ registers（warp/thread 级）→ Tensor Core MMA → 写回 global。每一级都在回答“这块数据现在放在哪、谁来算”。

为什么是三级？因为硬件就是三级：第 1 周讲过，block 对应 SM 和 shared memory，warp 对应 warp 调度器和 MMA，thread 对应寄存器和 MMA 的分片。CUTLASS 的结构不是凭空设计的，是硬件结构的映射——每一级管的就是它对应的那层硬件。这个映射关系可以反过来，如果只有一级分工，比如所有线程直接从 global memory 读数据算 MMA，global memory 的带宽会被几千个线程同时打爆；分了三级之后，global memory 只需要喂饱 block 级的 tile 搬运，shared memory 喂 warp，寄存器喂 thread，每一级的带宽需求都比上一级小一个量级。分级的本质是把一次大搬运拆成三级小搬运，每一级的搬运量都刚好配得上那一级硬件的带宽。MMA（Matrix Multiply-Accumulate，矩阵乘累加指令，Tensor Core 一次算一小块矩阵乘加的硬件指令）落在最底层，是因为 Tensor Core 吃的就是寄存器里的小块数据——thread 级的分片形状，最后要对齐 MMA 指令一次能吃的数据形状。

另外注意 CUTLASS 是模板库：“代码”主要是编译期生成 kernel 的模板，读它要读模板参数，不是读生成出来的代码；改一个 Shape 参数就是换一个实例化，所以下一节每改一次都要重编。模板库这个概念值得展开：你写的 `Gemm<float, ColumnMajor, ...>` 不是一段跑起来的代码，是一张“生成代码的图纸”；编译器读到这张图纸，才生成真正跑在 GPU 上的 kernel。改 ThreadblockShape 不会改任何一行“代码”，它改的是图纸上的一个尺寸，编译器按新尺寸重新生成 kernel——所以第 4 节每改一次配置都要重编，重编不是麻烦，是模板库的工作方式。读 CUTLASS 也因此和读 flash-attn 不一样：flash-attn 读的是已经写出来的 kernel，CUTLASS 读的是生成 kernel 的参数；前者找循环骨架，后者找三个 Shape 参数。

MMA pipeline：理解“流水线”的意思——下一个 tile 的搬运与当前 tile 的计算重叠，靠的是 shared memory 的双缓冲（甚至多级缓冲）与异步拷贝。画一张两级缓冲的时序图：tile k 在 buffer 0 里算的时候，tile k+1 正往 buffer 1 里搬；算完切换，搬完的那个 buffer 接着算。异步拷贝（cp.async）是让重叠成立的关键——搬运不用等算完，算也不用等搬完。先画出时序图，不用深究每一级缓冲的同步原语。双缓冲的账也可以用上一段的数字算：默认配置下一 K 步搬 8 KB、算 262144 次操作，双缓冲就是两块 buffer 各 8 KB，一共 16 KB 的 shared memory 被流水线占着。16 KB 这个数本身不大，关键在它让“搬”和“算”同时发生：没有双缓冲，每步都是先搬 8 KB 再算，搬运时间全暴露在关键路径上；有了双缓冲，搬运被藏到了上一块的计算时间里。同步原语（保证“搬完再算、算完再覆盖”的机制）这一节不用深究，知道它的作用是“防踩”就够了：buffer 1 没搬完不能开始算，buffer 0 没算完不能往里搬下一块——流水线的正确性就靠这两条规则。

定位改配置的入口：tile 配置由 ThreadblockShape / WarpShape / InstructionShape 三个模板参数决定，在 `cutlass/gemm/device/gemm.h` 与 `cutlass/gemm/device/default_gemm_configuration.h` 里找它们的定义与默认值。下一节要改的就是它们。三个参数的分工正好对应三级结构：ThreadblockShape 定 block 级 tile 的大小，WarpShape 定 warp 级的分块，InstructionShape 定 thread 级落到 MMA 指令上的小块形状。改配置就是改这三张图纸上的尺寸，改完重编，编译器生成新的 kernel。第 4 节会先只动 ThreadblockShape，再动 WarpShape，一次只改一个——这既是第 1 周阶梯优化的老规矩，也是模板库的特性决定的：一次改两个参数，性能变了你不知道是哪个参数的功劳。

**产出**：一张三级分工与数据流向图 + 改配置的入口位置（三个 Shape 参数的文件与默认值）。

### 第 4 节 · 动手改一个 CUTLASS GEMM：改 tile 配置，重编，测出变化

**目标**：改一个 CUTLASS GEMM 的 tile 配置，重编，测出变化。

**怎么做**（60 分钟，需 GPU + nvcc；无卡时先完成第 3 节的结构图，实测标注“待补测”）：

按 CUTLASS 仓库 README 的构建说明（CMake）编出 `00_basic_gemm`，先跑默认配置，记下耗时（例子自带计时输出；也可以用第 1 周的 CUDA Events 自己包一层）。记的是 kernel 的耗时，不是整条命令的 wall-clock——编译时间和进程启动不算在内。这个区分值得强调：`00_basic_gemm` 的一次运行里，编译可能占几分钟、进程启动占毫秒级，kernel 本身可能只有毫秒级；把 wall-clock 当成 kernel 耗时，结论会差出几个数量级。CUDA Events（第 1 周学过的 GPU 端计时方法，在 kernel 前后各打一个时间戳）量的才是 kernel 的时间；例子自带的计时输出如果已经是 kernel 计时，直接用它就行。

改配置之前，先把三组配置的账算好，带着预测去测，而不是测完再想。默认配置 128×128×8 的账第 3 节算过：每 K 步搬 8 KB、算 262144 次操作、算术强度 32 次操作每字节、双缓冲占 16 KB shared memory、4096×4096 的输出切成 1024 个 block。往小改的 64×128×8：A tile 是 64×8 = 512 个元素，B tile 还是 8×128 = 1024 个元素，两块加起来 1536 个元素，1536 × 4 = 6144 字节，即 6 KB；计算量是 2 × 64 × 128 × 8 = 131072 次操作，131072 ÷ 6144 ≈ 21.3，即算术强度约 21.3 次操作每字节；双缓冲占 12 KB；4096×4096 的输出切成（4096 ÷ 64）×（4096 ÷ 128）= 64 × 32 = 2048 个 block，是默认的 2 倍。往大改的 256×128×8：A tile 是 256×8 = 2048 个元素，B tile 1024 个元素，两块加起来 3072 个元素，3072 × 4 = 12288 字节，即 12 KB；计算量是 2 × 256 × 128 × 8 = 524288 次操作，524288 ÷ 12288 ≈ 42.7，即算术强度约 42.7 次操作每字节；双缓冲占 24 KB；4096×4096 的输出切成（4096 ÷ 256）×（4096 ÷ 128）= 16 × 32 = 512 个 block，是默认的一半。三组数排在一起，取舍就清楚了：tile 越大，每字节搬运对应的计算越多（42.7 对 32 对 21.3），数据复用越好；但 block 越少（512 对 1024 对 2048），并行度越低，同时每个 block 占的 shared memory 越多（24 KB 对 16 KB 对 12 KB），SM 上能同时驻留的 block 数越少。改 tile 配置不是在调一个“越大越好”的旋钮，是在复用和并行度之间找平衡点——这个平衡点没有公式可算，只能实测。

改配置：把 ThreadblockShape 从默认的 128×128×8 改成另一组（例如 64×128×8 或 256×128×8），只改这一处，重编，重跑，记下耗时变化。一次只改一个参数，否则变化归因不清——两个地方一起动，时间变了你不知道是谁的功劳，这是第 1 周阶梯优化的老规矩。64×128×8 是往小改：每个 block 算得少了、shared 占用降了、同时驻留的 block 多了，但每块 tile 的复用也少了；256×128×8 是往大改：反过来，还可能撞上 shared memory 上限、编不过或跑不起来——那也是结果，记下来。编不过这件事要特别说明：它不是“你改错了”，是配置本身越过了硬件的硬限制——一个 block 的 shared memory 占用超过上限，编译器会直接拒绝生成这个 kernel。记下“编不过”时，要把配置和报错一起记下来，这条记录本身就是结论：这条路在当前硬件上走不通。跑不起来也一样：如果程序编过了但跑起来结果不对，先拿 naive 参考 GEMM 验算——结果错了是正确性问题，耗时数字没有意义；结果对了耗时变了，才是性能结论。

再试 WarpShape：改一档，重编，重跑，记下变化。把三次结果填进同一张表：配置、耗时、相对默认的变化。WarpShape 的改动和 ThreadblockShape 的改动不在同一个层级：ThreadblockShape 改的是 block 级 tile 的大小，WarpShape 改的是 block 内 warp 之间的分工。改 WarpShape 时 ThreadblockShape 保持默认，这样两次改动的归因才分得开。表里的“相对默认的变化”建议写成百分比：比如某配置耗时是默认的 1.2 倍，或者 0.85 倍——百分比比绝对毫秒数更容易跨机器比较，绝对毫秒数换一台卡就变了，相对变化的趋势才有参考价值。

用第 1 周的分析框架解释结果：tile 变大 → 每个 block 算得更多、shared memory 占用上升、同时驻留的 block 数下降；tile 变小则反之。实测数字与这个分析对不上时，先怀疑测量（是不是只跑了一次、是不是含了编译时间），再怀疑分析。测量上最常见的坑是只跑一次：第一次跑常常包含预热（缓存冷、时钟没拉起来），单次数字的抖动可能比配置带来的变化还大；第 1 周的做法是跑多次取中位数，同样适用于这里。另一个坑是把编译时间算进去：CUTLASS 的编译不快，改一次配重编一次是正常的——把等待时间用来写第 5 节的对比表。如果实测和预测反着来，比如 tile 变大反而变慢了，先对一下账：是不是 block 数从 1024 掉到 512 之后，并行度不够填满所有 SM 了；是不是 shared 占用从 16 KB 涨到 24 KB 之后，SM 上同时驻留的 block 数少了，延迟藏不住了。预测错了不丢人，错的位置就是新的认识：Roofline 和第 1 周的分析框架给的是方向，具体的平衡点永远在实测里。

注意：CUTLASS 的编译不快，改一次配重编一次是正常的——把等待时间用来写第 5 节的对比表。

**产出**：一份“改前改后”实测记录（至少两组配置变化 + 耗时 + 一句归因）。

### 第 5 节 · 小结：Triton 版 vs CUTLASS 版 attention 的性能与可维护性

**目标**：写出“什么时候自己写 kernel，什么时候直接用库”的判断。

**怎么做**（60 分钟，无 GPU 可做）：

对照第 3–4 周的实测数据，填这张表：

| 维度 | Triton 版（第 3 周） | CUTLASS / flash-attn 版 | 结论 |
|---|---|---|---|
| 代码量 | 短，几十行 | 长，模板层层嵌套 | 快速验证选 Triton |
| 性能天花板 | 接近但到不了顶（缺 warp specialization 等精细控制） | 最高（FA3 的 75% 利用率是这么来的） | 榨干性能选 CUDA/CUTLASS |
| 可维护性 | 好读好改 | 难读难改，改一处牵全身 | 长期维护选 Triton 或直接用库 |
| 硬件适配 | 编译器兜底一部分 | 每代硬件要重写特化 | 换芯片时两边都要重做（M3 的主题） |

这张表每一行都可以对回本周的具体内容。代码量一行：第 3 周的 Triton 版几十行，flash-attn 光是前向 kernel 加分发模板加各 head 维的实例化文件就有多个文件，CUTLASS 更是模板套模板——代码量的差距不是风格问题，是控制粒度的问题：Triton 把分块、流水线、内存层级交给了编译器，代码短是因为编译器替你写了那部分；CUDA/CUTLASS 把控制权拿回自己手里，代码长是因为每一层分工都要亲手写出来。性能天花板一行：第 1 节的数字就是证据——FA3 在 H100 上做到 75% 利用率、740 TFLOPs/s，靠的是 warp specialization、matmul/softmax 交错、FP8 这些 Triton 语言里表达不出来的精细控制；Triton 版能接近这个天花板，但缺了这些控制手段就到不了顶。75% 和 35% 之间差了 40 个百分点，这 40 个百分点就是“精细控制”的价值，也是 Triton 版和工业版之间的性能差距。可维护性一行：第 2 节读 flash-attn 的体会可以直接写进来——定位一个结构要先看分发模板再进 kernel 本体，特化分支按 head 维和 GPU 代际展开，改一处要确认它影响的是哪个变体；第 4 节的体会也一样：改一个 Shape 参数就要重编整个 kernel，编译一次不快。硬件适配一行：第 1 节 FA2 在 H100 上只剩 35% 利用率就是现成的例子——换一代硬件，旧的特化就失效一部分；M3 换一块芯片（Ascend）时，同样的重排要再做一遍，Triton 版和 CUTLASS 版谁也躲不掉。

判断标准：瓶颈在访存模式与融合时，Triton 够用；瓶颈在硬件细节（bank conflict、流水线、异步拷贝）时，回 CUDA/CUTLASS；瓶颈已经是“别人写好的库比你快”时，直接用库。三条标准对应三种情况：优化能说成“哪个 block 读哪块”时用 Triton；优化必须说到“哪个 warp 干什么、流水线第几级”时回 CUDA/CUTLASS；优化到最后发现库已经比你快时，别较劲，直接用库——第 6 周的 CustomOp 收口就是第三种情况的实践：自己写的 kernel 要接进引擎，而不是停在 standalone。

三条标准可以各配一个本周的具体场景，判断就不再是口号。访存模式与融合的场景：第 3 周的 fused attention——把 QK^T、softmax、乘 V 融合成一个 kernel，省掉 S、P 矩阵过 HBM 的往返，这件事用 Triton 几十行就能写出来，瓶颈在“数据走哪条路”，不在“哪个 warp 在哪个周期干什么”，Triton 够用。硬件细节的场景：第 1 节 FA3 的 warp specialization 和 matmul/softmax 交错——瓶颈在“搬运 warp 和计算 warp 怎么重叠”“Tensor Core 在 softmax 阶段闲不闲”，这些要精确到 warp 分工和流水线级数，Triton 表达不出来，回 CUDA/CUTLASS。直接用库的场景：第 1 节 FA3 的 FP8 版——740 TFLOPs/s 的 FP16 和接近 1.2 PFLOPs/s 的 FP8，配上比基线低 2.6 倍的误差控制，自己从头写一个同等水平的 FP8 attention，工作量是数周起步，库已经摆在那里，直接用就是最优解。三个场景的区别不在“难不难”，在瓶颈落在哪个层级：访存层用 Triton，硬件层用 CUDA/CUTLASS，库已经封顶时用库。

还有一个判断容易被忽略：时间成本。Triton 版几十行，改一次重编快、验证快，适合“先验证想法对不对”；CUTLASS 版改一个 Shape 参数就要重编，第 4 节已经体会过编译等待的时间，适合“想法已经验证过，现在要榨干最后一点性能”。第 6 周的 CustomOp 收口把这个判断落到了工程上：自己写的 kernel 接进引擎，跑在真实推理链路里，才算完成；停在 standalone 的 kernel 再快，也只是个 benchmark 数字。换句话说，选工具的标准不只有“能跑多快”，还有“多久能跑起来”和“接进系统要多少成本”——本周的表里，“可维护性”和“硬件适配”这两行，讲的就是后两个成本。

把本周的检查清单并进总清单，加两条：“FA1→2→3 每一代的瓶颈与改动能说出来吗”“CUTLASS 的三个 Shape 参数改过并测出变化了吗”。这两条正好对应本周的两个核心产出：第一条对的是第 1 节的演进表，能说出来才算链条立住了；第二条对的是第 4 节的实测记录，改过并测出变化才算三级结构理解了。无卡的同学第二条先标“待补测”，有卡之后补上——清单的作用是记下“什么还没做”，不是假装“都做完了”。

**产出**：一页对比表 + 取舍笔记。
## 第 24 章 · 收口：把自写 kernel 接入推理引擎

> 注意：vLLM 的 attention 走 Attention 后端，不走 CustomOp（CustomOp 的可替换面是激活、归一化这类算子），所以接入引擎的载体是 Triton 版 SiluAndMul（与第 3 周的 fused attention 共享同一套“写 kernel → 验证 → 接入”流程）；fused attention 仍在第 3 周做 standalone 验证。

> 硬件：本周需要 GPU（租卡）。无卡时先完成第 1–3 节的机制学习与代码编写，端到端验证标注“待有卡后补测”。
>
> 本周是收口：前面五周的东西（写 kernel、验证、测速）要在这周接进一个真实引擎，跑出端到端的数据。

本周主线只有一句话：让第 2 节写出来的 kernel 在真实引擎里跑起来，再用同一套数字回答它到底快了多少。
### 第 1 节 · vLLM 的 CustomOp 机制：注册、分发、启用三件事，及它的边界

**目标**：能讲清插件算子的注册与调用路径，并讲清它的边界（attention 不走这条路）。

**怎么做**（60 分钟，无 GPU 可做）：

读 [vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)，再对照源码 `vllm/model_executor/custom_op.py`，搞清三件事。先把数摆出来：三件事、一个边界、一个只加载一次的 guard。注册、分发、启用是三件不同的事，发生在三个不同的时间点：注册在 import 时，分发在每次前向时，启用在启动配置里。把三件混成一件，是这一节最常见的糊涂——后面每一节的坑，都是在这里埋下的。

注册说的是名字怎么进表。`CustomOp` 的子类通过 `register_oot` 做 out-of-tree 注册（out-of-tree，树外注册：实现不放在 vLLM 自己的代码树里，放在你的插件包里），按类名替换；无参数时注册名就是类名，如 `"SiluAndMul"`。为什么按类名？因为模型代码里调用的就是这个类名，分发时按名字查表，名字就是查表的键。`CustomOp.register` 会断言不重名——同一个名字不能注册两次。这条断言不是刁难：同一个名字对应两个实现，分发时该用哪一个就含糊了，所以注册表一个名字只允许占一个位置。打个比方：注册表像点名册，一个名字只能对应一个人，重名就没法点名。第 3 节要避的坑就是这个：不要在同一个进程里把同一个名字注册两次。

分发说的是运行时怎么找到你的实现。模型代码里调用的是 `SiluAndMul()` 这个类，运行时分发到注册的实现上；`forward_cuda` 走 GPU 实现（你的 Triton kernel），`forward_native` 是 PyTorch 后备实现（CPU / 非 CUDA 路径走它）。后备为什么存在？同一份模型代码要在有卡和没卡的机器上都跑得起来，后备保证语义正确，代价是速度慢一点。这一句值得停一下：后备在语义上是对的，数值一样，只是慢一点。不报错、不警告——这正是第 3 节静默失败的根。替换没生效，模型照样跑，只是悄悄用回默认实现，分发检查或 profiler 之外没有任何东西会告诉你。

启用说的是开关怎么打开。编译配置里用 `--compilation_config.custom_ops '["+SiluAndMul"]'` 打开，`+` 表示“在默认集合之外再加这一个”，不是把整个集合换掉。注意 inductor 后端默认会禁用 custom ops，启用时要确认这条配置真的生效了。为什么 inductor 要禁？inductor 要把计算图抓下来融合重写，custom op 对它来说是个黑盒子，图优化器看不懂里面是什么，所以默认不让它进图。确认配置生效的办法，不在这一节空想：第 3 节的进程内检查就是确认的办法，注册上了、分发到了，才算启用真的生效了。

分发里还有一层意思值得注意：模型代码根本不知道 TritonSiluAndMul 存在。它只认 SiluAndMul 这个名字，查表才找到你的实现。注册表就是模型代码和你的 kernel 之间的解耦层：同一份模型代码，装了插件走 Triton 实现，没装插件走默认实现，模型代码一个字都不用改。配置里的名字也是同一张表里的键：`'["+SiluAndMul"]'` 里的 SiluAndMul 必须和注册名对上号，写错名字就对不上，分发还是走默认实现——名字对不上和插件没装，在输出上同样安静。

再讲清边界：attention 不走 CustomOp。vLLM 的 attention 走 Attention 后端（backend 选择发生在 attention 层，不是 CustomOp 分发）。CustomOp 的可替换面是激活、归一化这类“逐元素 / 小规约”算子。为什么边界划在这里？attention 有自己的后端选择机制（flash-attn、SDPA 等）和 KV cache 的集成：KV 的布局、显存分配、跨步长的复用，这些事一个 forward 函数装不下；激活算子就是一个 forward 函数，输入进来、输出出去，没有跨层的状态，没有后端要选，没别的。打个比方：CustomOp 是单个插座，只接得上单个电器；attention 是一间配电房，有自己的开关柜和线路，接不到单个插座上。选 SiluAndMul 做载体正是因为它在边界内：它是 MLP 里的激活算子，形状规整、易于验证——第 2 节会拿它当载体。

再读插件加载机制：vLLM 启动时通过 `vllm.general_plugins` entry point 加载插件包（源码见 `vllm/plugins/__init__.py`，`load_general_plugins` 在进程内只加载一次，有防重复加载的 guard）。关键时序是：插件模块被 import 时，`register_oot` 装饰器执行，替换即完成——注册发生在 import 时。反过来说：插件没被 import，什么替换都不会发生，模型照样跑，只是跑的是默认实现——这是第 3 节要防的静默失败。设问：如果注册发生在 import 时，那么一个插件“写好了但没被加载”和“根本不存在”，从模型输出上看有什么区别？没有区别。输出一样、速度一样、日志里一样安静。这就是为什么同一个问题（“我的 kernel 真的跑了吗？”）要问三遍：第 1 节问 import 了没有，第 3 节问分发到了没有，第 4 节问 profiler 里有没有——每一遍都有不同的东西可能掉链子。

两个常见的误解，单拎出来。“注册了就等于启用了”：错。注册是把名字写进表，启用是启动时把开关打开，缺一个都不行。“CustomOp 能包一切算子”：错。边界划在激活、归一化这类单 forward 算子，attention 在界外。结论：这一节的目标不是写代码，是拿到一张地图——注册、分发、启用三件事各在何时发生，边界划在哪里，静默失败长什么样。后三节就是沿着这张地图走一遍：写 kernel、包成插件、端到端验证，每一步都有对应的检查。

**产出**：能讲清插件算子的注册与调用路径 + 边界（attention 为什么不走这条路），并说出注册、分发、启用三个环节各自发生的时间点和各自失效时的表现。
### 第 2 节 · 动手：写 Triton 版 SiluAndMul，standalone 正确性 + 微基准

**目标**：写出可独立验证的 Triton 版 SiluAndMul，并算清融合到底省了多少访存。

**怎么做**（60 分钟，需 GPU 做正确性验证；无卡先写代码）：

先看 vLLM 自带的对照：`vllm/model_executor/layers/activation.py` 里的 `SiluAndMul` 类——`forward_native` 是 `F.silu(x[..., :d]) * x[..., d:]`，文件里还含有一个 Triton 版实现可对照写法。你的 kernel 要与这个语义完全一致：输入 `[M, 2D]`，前一半过 SiLU，再乘以后一半，得 `[M, D]`。SiLU（Sigmoid Linear Unit，`silu(x) = x · sigmoid(x)`）先对前一半逐元素作用，再和后一半逐元素相乘。为什么先读对照再写？对照就是正确性标准：这一节末尾的检查，就是拿你的 kernel 和它逐元素比，比的就是这一个语义。

先把融合的账算清楚。融合省的是遍数，不是计算量。拿 fp16（每个元素 2 字节）算一笔总账：融合版读输入 `[M, 2D]` 共 `4MD` 字节，写输出 `[M, D]` 共 `2MD` 字节，合计 `6MD` 字节。eager 版是两个 kernel：silu 读前一半 `2MD` 字节、写中间结果 `2MD` 字节；乘法读中间结果 `2MD` 字节、再读后一半 `2MD` 字节、写输出 `2MD` 字节，合计 `10MD` 字节。省掉的是 `4MD` 字节，占 eager 版访存的 40%。拿正文的算例走一遍：`M = 128`、`D = 128`，输入 `128 × 256` 个 fp16 元素共 64 KiB，输出 `128 × 128` 共 32 KiB，融合版一共 96 KiB；eager 版 160 KiB；省 64 KiB，正好 40%。exp 和 sigmoid 的 FLOPs 一点没少，省掉的是中间结果写回 HBM 再读回来的一次往返——和第 3 周 fused attention 省掉 S、P 落 HBM 是同一笔账，换了个算子而已。

再看这个算子快不快得起来。每个输出元素，融合版读 4 字节（`x1`、`x2` 各一个 fp16）、写 2 字节，一共 6 字节；每个输出元素只做几次 FLOPs。按正文给出的约 1 FLOP/字节，时间由带宽定，不由算力定。结论先摆出来：别指望数量级提速，融合的全部收益就是上面那 40% 的访存。先测再判断——这一节末尾的微基准就是量这个的。

写 kernel（二维 grid：行 × 列块；SiLU 在 fp32 里算，再 cast 回输入精度——cast 回去再乘是 load-bearing 的，直接拿 fp32 的 SiLU 结果乘 fp16 的另一半会丢精度；这个写法与 TritonAscendBench 的 `L2_04_silu_and_mul` 参考 kernel 一致，M3 第 5 周移植它时会再遇到）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _silu_and_mul_kernel(x_ptr, y_ptr, D,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    x1 = tl.load(x_ptr + row * 2 * D + col, mask=mask, other=0.0)
    x2 = tl.load(x_ptr + row * 2 * D + D + col, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    s = (x1_f32 * tl.sigmoid(x1_f32)).to(x1.dtype)
    y = s * x2
    tl.store(y_ptr + row * D + col, y, mask=mask)

def silu_and_mul_triton(x, BLOCK=1024):
    # x: [M, 2*D] fp16；y: [M, D] fp16
    M, D2 = x.shape
    D = D2 // 2
    y = torch.empty((M, D), dtype=x.dtype, device=x.device)
    grid = (M, triton.cdiv(D, BLOCK))
    _silu_and_mul_kernel[grid](x, y, D, BLOCK=BLOCK)
    return y
```

注意 `.to(x1.dtype)` 的位置：SiLU 在 fp32 里算完，先 cast 回输入精度再乘。这个顺序是 load-bearing 的——改了顺序，max abs err 就会变，下一段的正确性检查就是量这个的。为什么这个顺序关键？sigmoid 里有 exp，第 2 周算过，fp32 下 exp 的输入超过约 89 就会溢出成 inf；先在 fp32 里把 SiLU 算完，中间结果的精度余量大得多，再 cast 回去，误差只来自最后一次舍入。如果拿 fp32 的 SiLU 结果直接乘 fp16 的另一半，乘法在混合精度下做，误差的来源就变了，max abs err 就会变。这个写法与 TritonAscendBench 的 `L2_04_silu_and_mul` 参考 kernel 一致，M3 第 5 周移植它时会再遇到。grid 是二维的（行 × 列块），每个程序负责一行里的一块列，这和第 2 周的向量加法是一维 grid 的推广：程序的组织方式跟着数据的形状走。

逐行看地址算术，这是最容易写错的地方。行内偏移：`x1` 在 `row * 2 * D + col`，`x2` 在 `row * 2 * D + D + col`，输出在 `row * D + col`。行跨度是整行 `2D`，不是 `D`——写成 `row * D`，读到的就是错行的数。`mask = col < D` 配 `other=0.0`：mask 掉的位置读进来是 0，`silu(0) = 0`，store 时又被同一个 mask 拦住，写不进去。mask 在这里既防越界读，也防越界写。再看 grid 的数：正文算例 `M = 128`、`D = 128`、`BLOCK = 1024`，列块数 `cdiv(128, 1024) = 1`，grid 是 `(128, 1)`，一共 128 个程序。每个程序手里有 1024 条 lane，真正干活的只有前 128 条，剩下的 896 条被 mask 掉，占 87.5%。这说明什么？这个算例是正确性算例，不是性能算例：`BLOCK = 1024` 配 `D = 128`，大部分 lane 在空转。换一组数算算（纯算例，不对应某个模型）：`D = 2048` 时列块数是 2，`D = 4096` 时是 4。BLOCK 该取多大，要跟着 D 走，答案在微基准里扫出来，不在纸上定。

正确性检查（与 `F.silu` 参考对比）与微基准（`triton.testing.do_bench`，对比 PyTorch 版）：

```python
def ref_silu_and_mul(x):
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

torch.manual_seed(0)
x = torch.randn(128, 256, dtype=torch.float16, device='cuda')
y = silu_and_mul_triton(x)
ref = ref_silu_and_mul(x)
print('max abs err:', (y.float() - ref.float()).abs().max().item())
```

先管理预期：这个算子是访存受限的，别指望数量级提速。逐元素算子每个输出元素只做几次 FLOPs，却要读两个、写一个，算术强度约 1 FLOP/字节，远低于机器的平衡点——时间由带宽定，不由算力定。融合能省的是遍数：eager 模式下 silu 和乘法是两个 kernel，中间结果要过一次 HBM；融合版是一个 kernel，中间结果在寄存器里。省一遍 HBM 就是这一节的全部收益，先测再判断。

检查要怎么读结果？`torch.manual_seed(0)` 固定随机数，`x = torch.randn(128, 256)`，max abs err 量的就是你的 kernel 和 PyTorch 参考逐元素差多少。预期：误差不是 0——求值顺序不同，fp16 舍入路径不同，全等本来就不该出现；要看的是误差的数量级。误差小到和 fp16 舍入一个量级，就是对齐了；误差大到和输出值本身差不多，就是写法错了。三个典型错误示范：两半切反了（silu 加在后一半上），误差立刻变大；行跨度写成 `row * D`，读错行，误差立刻变大；cast 顺序颠倒，误差变大但不一定大到一眼看出来——所以这个检查要和逐行对照一起做，不要只看一个数就下结论。微基准用 `triton.testing.do_bench`，对比 PyTorch 版，同一形状、同一精度；第 2 周讲过 do_bench 取中位数的道理，测出来的数要按那个读法读。怎么读结果：如果时间省下来的比例接近访存省下来的 40%，说明访存确实是瓶颈；如果时间没怎么动，先看是不是形状太小、kernel 启动开销占了大头——这是个待验证的假设，数字要测出来再写，不要先下结论。

**产出**：可独立验证的 Triton kernel（数值对齐 + 一份微基准），附一笔融合省访存的字节账。
### 第 3 节 · 包成 CustomOp 子类并注册：`register_oot` 按类名替换，进程内验证替换真的发生了

**目标**：做出可加载的插件包，并验证替换真的发生了。

**怎么做**（60 分钟，需 GPU 做加载验证；无卡先写插件包骨架）：

写插件包。先把骨架的三部分说清楚：子类、注册、入口。子类是实现，分发时按类名找的就是它；注册是把名字写进表；入口是让 vLLM 启动时找到你的包。三部分缺一个，链条就是断的。骨架如下（类名、注册方式与签名以你所装 vLLM 版本的 `vllm/model_executor/custom_op.py` 与 [CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)为准，版本变了先核对再抄）：

```python
# my_plugin/silu_and_mul_plugin.py
import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul

@SiluAndMul.register_oot
class TritonSiluAndMul(SiluAndMul):
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul_triton(x)
    # forward_native 从父类继承（PyTorch 实现），CPU / 非 CUDA 路径走它
```

子类只重写 `forward_cuda`，里面调第 2 节的 `silu_and_mul_triton`。`forward_native` 从父类继承（PyTorch 实现），CPU / 非 CUDA 路径走它。为什么留着后备？同一份插件要在没卡的机器上也能 import、也能跑——跑的是 PyTorch 实现，调试和验证都不依赖 GPU。后备是安全网，也是第 1 节讲过的静默失败的来源：它让“替换没生效”看起来一切正常。注册用 `@SiluAndMul.register_oot` 装饰器，按类名注册；装饰器在模块被 import 时执行，执行完替换即完成。注册发生在 import 时，不是调用时——第 1 节的时序在这里落地。类名、注册方式与签名以所装版本为准：不要从记忆或博客里抄一份旧签名，旧签名在新版本上可能悄悄对不上，核对的成本是一分钟，排错的成本是一下午。

打包：插件通过 `vllm.general_plugins` entry point 被 vLLM 启动时加载（加载逻辑见 `vllm/plugins/__init__.py`）。entry point 的具体写法（模块路径的格式）以你所装版本的 `vllm/plugins/__init__.py` 为准，写完先在本地 `pip install -e .` 装上，再确认 `vllm` 启动日志里出现了你的插件。entry point 写错了，插件根本不会被加载——vLLM 是按 entry point 发现插件的，不会去扫描你的代码。打个比方：entry point 是插件在 vLLM 门口的登记表，没登记，门卫不知道你来过，包写得再好也进不了门。

验证分两层做，先便宜后贵。第一层在裸 Python 进程里：import 插件模块，装饰器执行，注册即完成，再检查 `SiluAndMul` 解析到的是不是 `TritonSiluAndMul`——这一步不用 GPU、不用启动 vLLM，注册本身的问题在这里就能抓住。第二层再进 vLLM 进程里验：启动时插件被 entry point 加载，跑一次前向，看 `forward_cuda` 里的打印出现没有。两层查的是同一张表：第一层查注册写进去了没有，第二层查 vLLM 启动时是不是真的加载了这张表。无卡时第一层照样能做：`forward_native` 从父类继承，CPU 路径走 PyTorch 实现，import、注册、解析检查都不依赖 GPU，骨架的正确性先在 CPU 上验完，有卡再验 `forward_cuda`。

验证替换真的发生了（不要只看“没报错”）：在进程内检查 `SiluAndMul` 解析到的是你的 `TritonSiluAndMul`；或者在 `forward_cuda` 里临时加一行打印，跑一次前向看它是否出现。两种办法至少做一种——“替换没生效但模型照样跑”是这个环节最常见的静默失败（模型会悄悄用回默认实现，你的对比表就全是假的）。静默的原因是后备实现在语义上是对的：数值一样，只是慢一点，不报错、不警告，只有分发检查或 profiler 能发现它。打个比方：替补和首发穿同一件球衣，比分牌（输出数值）看不出谁在场上，只有名单（分发检查）或录像（profiler）能看出来。下一节的 profiler 检查，就是同一个问题再问一遍：我的 kernel 真的跑了吗？

避坑：`register_oot` 按类名注册，同一个名字注册两次会断言失败。不要在同一个进程里既 import 插件又手动再注册一次；也不要在 `vllm` 启动后重复加载插件（`load_general_plugins` 有防重复加载的 guard，但你自己的代码里也要避免）。一个错误示范：调试时在 REPL 里 import 两次插件模块、又手动调一次 `register_oot`，断言失败，报错看起来像是插件写错了，其实是注册了两次。`load_general_plugins` 的 guard 防的是 vLLM 自己重复加载，不是防你手滑：你在自己代码里手动再调一次 `register_oot`，guard 拦不住，断言会拦住——报错就是防线起作用的样子，不要把它当成插件写坏了。区分“注册失败”和“注册了两次”：前者是插件没生效，后者是插件生效了两次——报错信息要读全，不要只看最后一行就下结论。

**产出**：可加载的插件包 + 一份“替换已验证”的证据（解析检查或打印日志）。
### 第 4 节 · 端到端验证：同一模型、同一评测集，接入前后的 TTFT / ITL / 吞吐 + profiler 确认 Triton kernel 真的跑了

**目标**：出一份接入前后对比表。

**怎么做**（60 分钟，需 GPU；无卡时标注“待有卡后补测”）：

用 M0 的 `scripts/bench_openai.py` 跑基线：不加载插件，同一模型、同一评测集，记 TTFT / ITL / 吞吐。TTFT（Time to First Token，首 token 延迟）量的是第一个 token 多久出来，ITL（Inter-Token Latency，每 token 间隔）量的是之后每个 token 之间隔多久，吞吐量的是单位时间一共出多少 token。三个数看的是三件事：TTFT 看 prefill 与首 token，ITL 看 decode 的稳定程度，吞吐看整机的产出。为什么用 M0 的脚本？M0 的基线章节已经把它调通过了，同一把尺子量前后，尺子本身不引入误差。

加载插件并用 `--compilation_config.custom_ops '["+SiluAndMul"]'` 启用，再跑同一评测集，记同一组数。两次之间只改这一个变量（模型、评测集、并发、采样参数全都不变），否则对比无效——这是第 5 周“一次只改一个参数”的同一条规矩，搬到了端到端。设问：并发改了会怎样？并发一改，TTFT 和吞吐一起变，分不清是插件的功劳还是并发的功劳，对比表就作废了。采样参数也一样：温度一改，输出长度变了，ITL 和吞吐跟着变，表里的数就没法归因。控制变量不是洁癖，是对比表成立的前提。

用 profiler 确认 Triton kernel 真的跑了（vLLM 的 profiler 或 torch profiler，看 kernel 名单里有没有你的 `_silu_and_mul_kernel`）。没有它，第 3 节的验证就不完整：分发检查说的是“模型调了你的类”，profiler 说的是“GPU 上跑了你的 kernel”，两句话说的不是同一件事，都要拿到。反过来，如果 profiler 名单里没有你的 kernel 名，但 TTFT 变好了，好的是什么？要么是别的变量动了，要么是测量噪声——不管是哪一种，表都不能直接用。

先管理预期：SiluAndMul 是个小算子，端到端提速很可能接近 0——这本身就是结论。把“没变快”写下来并分析原因（算子占比太小？Triton kernel 的启动开销盖过了收益？），不要为了好看去调评测集。道理是 Amdahl 定律：总提速 = 1 / ((1 − p) + p / s)，p 是算子占端到端时间的份额，s 是算子自身的提速倍数。SiluAndMul 占端到端时间的 1%，把它优化到无限快（s 趋向无穷大）最多也只省 1% 的时间，总提速 1 / 0.99 ≈ 1.0101，也就是 1.01%。如果你的 Triton 版只比 PyTorch 版快 2 倍（s = 2），同样 1% 的份额，总提速 1 / (0.99 + 0.005) = 1 / 0.995 ≈ 1.0050，时间只省 0.5%。换个份额算算：占比 5% 时无限优化最多省 5% 的时间（提速约 1.0526）；占比 10% 时最多省 10%（提速约 1.1111）。天花板是份额本身，不是 kernel 的速度。结论：端到端数字量的是算子的占比，不是 kernel 的质量——kernel 的质量第 2 节的微基准已经量过了。这是两个问题、两个数字，别拿一个回答另一个。

启动开销的假设怎么验？看第 2 节的微基准：如果单个 kernel 的时间本身就和启动开销一个量级，端到端省不下来就是意料之中——但这个判断要拿实测数说话，没测出来之前写“待补测”，不要写成结论。同样，评测集、并发、采样参数一旦动过，对比表要重跑，不要在旧表上改数字。一份诚实的对比表，数字可以不好看，但每一行都要能说清是怎么测出来的。

**产出**：一份接入前后对比表（TTFT / ITL / 吞吐 + profiler 证据）。
### 第 5 节 · 小结：一份“自写 kernel 端到端加速报告”（含负面结果分析）

**目标**：写出 M2 的可交付物。

**怎么做**（60 分钟，无 GPU 可做）：

先算一笔总账：6 周 × 5 节 × 60 分钟 = 1800 分钟 = 30 小时。本周 5 节是 300 分钟、5 小时。报告就是这 30 小时的收口。把第 1–6 周的东西收进一份报告：GEMM 优化阶梯总表（第 1 周）、fused attention 的推导与实测（第 3 周）、量化方案与误差账本（第 4 周）、FA 演进与 CUTLASS 改配置记录（第 5 周）、CustomOp 接入前后对比（第 4 节）。五份材料，一份都不能少：少一份，链条上就少一个环节，报告就讲不全“从写 kernel 到接进引擎”这件事。

这份报告是 M2 的可交付物：它证明的是“你会写 kernel、验证它、测出它快在哪、再把它接进引擎”——四件事缺一件，链条就是断的。打个比方：四条腿的桌子，缺一条腿站不住。写 kernel 是第 1–3 周，验证是第 2–3 节讲的方法（数值对齐、分发检查、profiler 三遍问），测速是第 1 周的 Roofline + ncu 和第 2 节的微基准，接进引擎是本周。报告的写法：每份材料都要有数字和证据（推导、实测、profiler 里的 kernel 名单），没有证据的说法写“待补测/未核实”，不要补编。数字和证据对不上的地方，就是下一轮要补做的功课。

报告里必须有一节“负面结果分析”：哪些尝试没变快、为什么（至少写一条真实的：比如 SiluAndMul 接入后端到端无变化的原因）。一份只报喜的报告在推理优化里是不可信的——做过这行的人都知道，大多数优化动不了端到端数字；写得出“为什么没变快”，才说明你真的量过、真的想过。第 4 节的 Amdahl 算账就是现成的分析工具：占比 1% 的算子，无限优化也只省 1%，“没变快”不是失败，是份额太小这个事实的测量结果。负面结果的素材平时就要攒：第 4 周讲过，预测是可证伪的，测出来不对也是结果，藏起来才是浪费——那句话在这里同样适用。

对照本模块检查清单，逐项打勾；没打勾的项写清原因与补做计划（比如“第 4 节 CUTLASS 实测待有 nvcc 的机器后补测”）。打勾的标准是证据，不是印象：打勾意味着这一项有对应的实测数或验证记录，翻报告能找到；没打勾也不要紧，要紧的是写清为什么没做、什么时候补做。补做计划要写具体条件（“待有 nvcc 的机器后”就是条件），不要写“以后再说”。

完成第三部分意味着什么？不是“所有 kernel 都变快了”，是“每个说法都有证据，每个没变快的尝试都有原因”。这份报告就是证据的合集。带着报告走：下一模块（M3，国内芯片移植）要移植的第一个 kernel 就是本周的 SiluAndMul——第 2 节的写法、第 3 节的注册、第 4 节的对比表，到 M3 第 5 周会再用一遍。本周收的口，是下一模块的起点。

**产出**：**M2 的可交付物**——一份“自写 kernel 端到端加速报告”（含负面结果分析）。
## 本模块检查清单

- [ ] 有一张 GEMM 优化阶梯总表：四个版本各自的算术强度、实测耗时、实测算力、ncu 读数与瓶颈判断
- [ ] 能用 ncu 的两个数（SM% / DRAM%）判断一个 kernel 是计算受限、访存受限还是延迟受限
- [ ] 能手推 online softmax 的 rescale 公式，并按 SRAM 容量算出自己 GPU 的分块参数
- [ ] 写出过 fused attention 与 fused dequant matmul，数值都对齐过
- [ ] 改过一个 CUTLASS GEMM 的 tile 配置并测出了变化
- [ ] 能画出 vLLM 里一个 CustomOp 从注册到分发的完整链路
- [ ] 有一份“自写 kernel 接入引擎前后”的端到端对比报告（含负面结果）
