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

指令数算一遍。一个 32×32 的 tile 有 1024 个 float：标量读法 1024 条 load，`float4` 读法 256 条。一个 block 每次迭代搬 A、B 两块，128 次迭代：标量 2 × 128 × 1024 = 262,144 条，`float4` 2 × 128 × 256 = 65,536 条，省 196,608 条。16,384 个 block 合计省下约 32.2 亿条 load 指令。比例是 1/4，这个比例比总数有用。

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

一个 1024 token 的序列，attention 的分数矩阵是 1024×1024。把它算出来、写回显存、再读回来做 softmax，S 和 P 两个中间矩阵各占 2 MiB，一写一读就是 8 MiB 的搬运；而 Q、K、V、O 四个张量加起来只有 512 KiB。A100 的 HBM 带宽是 2 TB/s，8 MiB 对应 4.19 µs，加 512 KiB 的 0.26 µs，共约 4.46 µs——朴素实现的瓶颈不在乘加，在中间矩阵的往返。FlashAttention 的做法是让 S 和 P 永远不落 HBM：Q、K、V 分块读进 SRAM（静态随机存取存储器：GPU 每个 SM 上的一块高速存储，CUDA 里以 shared memory 的形式使用），softmax 与加权求和在片上完成，只把最终输出写回。这要求先回答两个问题：softmax 的整行最大值与分母，分块之后怎么精确算出来；每 SM 只有 164 KB 的 SRAM，块最大能开到多少。这两个问题答完，kernel 才有得写；写完还要用正确性、计时与 Roofline 三项验收。

**本章目标。** 手推 online softmax 的更新式并说明它与整行 softmax 精确相等；按 A100 的 164 KB shared memory 算出 block 上限与寄存器压力；写出单 head、单序列的 Triton fused attention（causal 可选），并用 SDPA 对照、`do_bench` 计时和 Roofline 定位验收。

### 第 1 节 · Online softmax 的数学推导

标准 softmax 作用在一行长为 N 的分数向量 `x` 上：

```text
m = max_j x_j
p_i = exp(x_i − m)
l = Σ_j p_j
softmax(x)_i = p_i / l
```

减最大值是数值稳定措施，不是数学上的需要。FP32 能表示的最大值约 3.4×10³⁸，`exp` 的输入超过 `ln(3.4×10³⁸) ≈ 88.72` 就溢出成 inf；减去 m 之后所有输入 ≤ 0，输出落在 (0, 1]，溢出不会发生。softmax 对整体平移不变，减 m 不改变结果。

问题在于 m 和 l 都是整行级别的量。分块之后，处理第一块时后面的块还没读进来，m 和分母都未知；如果等整行读完再算，S 和 P 就必须先写回显存，融合省下的搬运又还回去了。online softmax（即在线 softmax：分块计算 softmax 时维护运行最大值与归一化因子，使分块结果与整行计算精确相等）就是为这个局面准备的：维护运行量，每来一块更新一次。

设已处理部分的运行最大值是 m，运行和是 `l = Σ exp(x_j − m)`。新来一块，块内最大值 `m_b`、块内和 `l_b = Σ exp(x_j − m_b)`。取

```text
m' = max(m, m_b)
```

把旧状态换到新基准 m' 下。对已处理的每一项：

```text
exp(x_j − m) = exp(x_j − m') · exp(m' − m)
```

两边对已处理的 j 求和：

```text
Σ exp(x_j − m) = exp(m' − m) · Σ exp(x_j − m')
```

所以旧状态在新基准下的和是 `l · exp(m − m')`。新块自己的和是 `l_b · exp(m_b − m')`。两部分相加：

```text
l' = l · exp(m − m') + l_b · exp(m_b − m')
```

这就是 rescale 公式。它不是对旧结果的近似修正，而是把同一个和从基准 m 换到基准 m'；换基准只改变每一项乘的公共因子，不改变各项之间的比例。

attention 的输出多一个 V 的加权：

```text
Õ = Σ_j exp(S_j − m) · V_j
```

同样的换算给出：

```text
Õ' = Õ · exp(m − m') + Õ_b · exp(m_b − m')
```

其中 `Õ_b = Σ_{j∈块} exp(S_j − m_b) · V_j`。最后 `O = Õ / l`。l 和 Õ 必须乘同一个 `α = exp(m − m')`：它们是在同一个基准下算出来的，只换一个就等于把分子分母的比例改了。

精确性可以直接验证。取 `x = [1, 2, 3, 4]`，分两块 `[1, 2]` 和 `[3, 4]`。整行算：m = 4，`exp(x − 4) = [0.049787, 0.135335, 0.367879, 1]`，`l = 1.553002`，softmax = `[0.032059, 0.087144, 0.236883, 0.643914]`。分块算：第一块 m₁ = 2，`l₁ = e⁻¹ + e⁰ = 1.367879`；第二块 m_b = 4，`l_b = 1.367879`；`α = exp(2 − 4) = 0.135335`；`l' = 1.367879 × 0.135335 + 1.367879 × 1 = 0.185122 + 1.367879 = 1.553002`。与整行结果在 FP32 舍入范围内相同。

attention 的例子：`S = [1, 2, 3, 4]`，`V = [10, 20, 30, 40]`，d = 1。整行算 `O = 0.032059×10 + 0.087144×20 + 0.236883×30 + 0.643914×40 = 34.926527`。分块算：第一块 `Õ₁ = e⁻¹×10 + e⁰×20 = 23.678794`；第二块 `Õ_b = e⁻¹×30 + e⁰×40 = 51.036383`；`Õ' = 23.678794 × 0.135335 + 51.036383 = 3.204576 + 51.036383 = 54.240960`；`O = 54.240960 / 1.553002 = 34.926527`。同样的结果。

写成归纳法。每一步的不变式是：

```text
Õ = Σ_{j∈已处理} exp(S_j − m) · V_j
l = Σ_{j∈已处理} exp(S_j − m)
```

处理第一块之前，已处理集合为空，和为 0；m = −inf 只是哨兵。第一块到来时 m' = m_b，`α = exp(−inf − m_b) = 0`，旧状态被清零，新状态正好是第一块的量。之后每一步按上面的代数保持不变式。所有块处理完，m 是整行最大值，l 和 Õ 是整行的量，`Õ / l` 就是标准 attention。分块与整行计算在数学上相等，差别只在浮点舍入和中间结果的存放位置。

有一个边界会直接影响代码：如果某一行的所有位置都被 mask 掉，`m_b = −inf`，`m' = max(m, m_b) = −inf`，`α = exp(−inf − (−inf)) = exp(NaN) = NaN`。这不是精度问题，是定义域问题。kernel 必须保证每一行至少有一个合法位置参与第一次更新；21.3 节会说明标准循环为什么天然满足这一点，以及加了块跳过之后为什么不再满足。

20.3 节的 fused softmax 是这里的单块特例：整行一次读进来，m 一次算到位，α 永远不被用到。分块版把同一套公式用在多块上，公式没有变。

### 第 2 节 · SRAM 容量约束如何决定 block 大小

fused attention（融合注意力：把 QK^T、softmax 与 PV 融进一个 kernel，N×N 中间矩阵不写回显存）的整个卖点是 S 和 P 不落 HBM；代价是 Q、K、V、P 四块 tile 要在同一个 SM 的 SRAM 里同时存在。A100 每个 SM 有 164 KB shared memory（从 192 KB 的 L1/shared 统一容量中最多划出 164 KB），H100 是 228 KB；单个 block 能申请的上限比这少 1 KB（A100 163 KB，H100 227 KB），差的 1 KB 留给驱动。这个数是每 SM 的，不是整张卡的：A100 有 108 个 SM，每个 SM 各管各的 164 KB，block 之间不共享。

一个 block 在 SRAM 里要放的东西：

- Q tile：`Br × d` 个 FP16，`2·d·Br` 字节。它在整个 K/V 循环里只读一次，之后一直复用。
- K tile：`Bc × d` 个 FP16，`2·d·Bc` 字节。每次内层迭代读一块。
- V tile：`Bc × d` 个 FP16，`2·d·Bc` 字节。K 算完分数之后才用它做加权求和。
- P tile：`Br × Bc` 个 FP16，`2·Br·Bc` 字节。它是第二次 `tl.dot` 的左操作数。

K 和 V 各占一份，是因为一次迭代里两件事都要做：先拿 K 算分数，再拿 V 做加权和；流水线还会在当前块计算时预取下一块 K/V。只开一个缓冲时，K/V 各一份；`num_stages=2` 时各多一份，额外 `4·d·Bc` 字节。

`qk`（也就是 S）是 `Br × Bc` 的 FP32，在 Triton 的 lowering 里是寄存器里的张量，不占 shared；如果某个后端把它落到 shared，预算要再加 `4·Br·Bc`。输出累加器 `acc` 是 `Br × d` 的 FP32，`q` 是 `Br × d` 的 FP16，`m_i`/`l_i` 各是长度为 Br 的 FP32 向量，它们都在寄存器里。

按活 tile 算，单缓冲的 shared 预算是：

```text
shared ≈ 2·d·(Br + 2·Bc) + 2·Br·Bc
```

第一项是 Q、K、V，第二项是 P。`Bc` 在括号里出现两次，因为 K 和 V 各有一份。开双缓冲再加 `4·d·Bc`。

`d = 64` 时逐项算出来：

| Br | Bc | Q | K | V | P | 单缓冲合计 | 双缓冲合计 |
|----|----|---|---|---|---|------------|------------|
| 64 | 64 | 8 KB | 8 KB | 8 KB | 8 KB | 32 KB | 48 KB |
| 128 | 32 | 16 KB | 4 KB | 4 KB | 8 KB | 32 KB | 40 KB |
| 64 | 128 | 8 KB | 16 KB | 16 KB | 16 KB | 56 KB | 88 KB |
| 128 | 64 | 16 KB | 8 KB | 8 KB | 16 KB | 48 KB | 64 KB |
| 128 | 128 | 16 KB | 16 KB | 16 KB | 32 KB | 80 KB | 112 KB |
| 256 | 64 | 32 KB | 8 KB | 8 KB | 32 KB | 80 KB | 96 KB |
| 256 | 128 | 32 KB | 16 KB | 16 KB | 64 KB | 128 KB | 160 KB |
| 128 | 256 | 16 KB | 32 KB | 32 KB | 64 KB | 144 KB | 208 KB |

表中每一格都是 `2·d·Br`、`2·d·Bc` 或 `2·Br·Bc` 算出来的，例如 `Br=128, Bc=64`：Q = 2×64×128 = 16 KB，K = V = 2×64×64 = 8 KB，P = 2×128×64 = 16 KB，合计 48 KB；双缓冲多一对 K/V，16 KB，得 64 KB。`128×128` 的 P 是 2×128×128 = 32 KB，合计 80 KB，双缓冲 112 KB；开到 3 级流水线再加 32 KB，144 KB，已经接近 A100 的 164 KB。`128×256` 单缓冲 144 KB，双缓冲 208 KB，超过 A100 单 block 的 163 KB 上限，这个组合在 A100 上开不了双缓冲。

P 是预算里唯一按 `Br × Bc` 增长的部分。`Br` 和 `Bc` 各翻一倍，Q/K/V 只翻一倍，P 翻四倍：`64×64` 的 P 是 8 KB，`128×128` 是 32 KB，正好四倍；同一组变化里 Q 从 8 KB 到 16 KB，只翻一倍。块开大之后，预算的大头会越来越向 P 倾斜。

换 `d` 也会改账。`d = 128, Br = 128, Bc = 64`：Q = 2×128×128 = 32 KB，K = V = 2×128×64 = 16 KB，P = 2×128×64 = 16 KB，合计 80 KB。这和 `d = 64, Br = 128, Bc = 128` 的 80 KB 数值相同，但来源不同：前者是 head 维变宽，后者是每步扫的列变多。`d = 128, Br = 64, Bc = 64` 是 Q 16 KB、K 16 KB、V 16 KB、P 8 KB，合计 56 KB。换模型、换 head 维，上一张表的结论要重算。

shared 只是一半。另一半是寄存器。`qk`、`acc`、`q`、`m_i`/`l_i` 都在寄存器里，按 32 位寄存器数：

```text
regs/thread ≈ (Br·Bc + 1.5·Br·d + 2·Br) / (32·num_warps)
```

`qk` 是 `Br·Bc` 个 FP32；`acc` 是 `Br·d` 个 FP32；`q` 是 `Br·d` 个 FP16，折合 `Br·d/2` 个 32 位寄存器；`m_i` 和 `l_i` 各 `Br` 个 FP32，合计 `2·Br`。除以线程数 `32·num_warps`。这个数只算四个大头，P 的 FP32 临时量、mask、地址与编译器临时量另算，实际用量更高。

| Br | Bc | num_warps | qk | acc | q | regs/thread（下限） |
|----|----|-----------|----|-----|---|---------------------|
| 64 | 64 | 4 | 4,096 | 4,096 | 2,048 | 81 |
| 128 | 64 | 8 | 8,192 | 8,192 | 4,096 | 81 |
| 64 | 128 | 4 | 8,192 | 4,096 | 2,048 | 113 |
| 128 | 128 | 8 | 16,384 | 8,192 | 4,096 | 113 |
| 256 | 64 | 8 | 16,384 | 16,384 | 8,192 | 162 |
| 256 | 128 | 8 | 32,768 | 16,384 | 8,192 | 226 |

取 `d = 64` 算两行：`64×64`、4 个 warp（128 线程）时 `(4096 + 1.5×4096 + 128) / 128 = 81`；`128×128`、8 个 warp（256 线程）时 `(16384 + 1.5×8192 + 256) / 256 = 113`。`256×64` 到 162，`256×128` 到 226；每个线程最多 255 个寄存器，后两组加上编译器的临时量很容易溢出，溢出会把值放到 local memory，延迟和带宽都变差。

寄存器总量也要算。每个 SM 有 64K 个 32 位寄存器；一个 block 的寄存器用量是 `线程数 × regs/thread`。`64×64`、4 warp、81 regs/thread 时是 128×81 = 10,368 个，64K ÷ 10,368 ≈ 6.3，最多 6 个 block；但 shared 那一侧只允许 5 个（164 ÷ 32），所以 shared 先顶到天花板。双缓冲 48 KB 时 shared 只允许 3 个。`128×64`、8 warp、81 regs/thread 时是 256×81 = 20,736 个，64K ÷ 20,736 ≈ 3.2，最多 3 个；双缓冲 64 KB 时 shared 只允许 2 个。`128×128`、8 warp、113 regs/thread 时是 28,928 个，最多 2 个；双缓冲 112 KB 时 shared 只允许 1 个。`256×64` 是 256×162 = 41,472 个，寄存器只允许 1 个，shared 单缓冲却允许 2 个，实际按 1 个算。

把这些数并排看：

| 配置 | 单缓冲 shared | 双缓冲 shared | A100 驻留（单/双） |
|------|--------------|--------------|---------------------|
| 64×64 | 32 KB | 48 KB | 5 / 3 |
| 128×64 | 48 KB | 64 KB | 3 / 2 |
| 64×128 | 56 KB | 88 KB | 2 / 1 |
| 128×128 | 80 KB | 112 KB | 2 / 1 |
| 256×64 | 80 KB | 96 KB | 1 / 1 |
| 256×128 | 128 KB | 160 KB | 1 / 1 |

H100 的 228 KB 宽一些：`64×64` 单/双是 7/4，`128×64` 是 4/3，`128×128` 是 2/2，`256×128` 还是 1/1。驻留数不是越高越好：每个 block 都要搬自己的 K/V，驻留多意味着更多 block 争同一份 L2 与带宽；但驻留太少，延迟藏不住。这个取舍要用实测定。

块大小还决定程序总数。单 head、单序列时，`grid = ceil(N / Br)`：N = 1024 时，Br = 64 只有 16 个程序（CTA，Cooperative Thread Array，线程块），Br = 128 是 8 个，Br = 256 是 4 个。A100 有 108 个 SM，16 个程序最多占 16 个 SM，连六分之一都不到。每个 SM 能驻留 3–5 个 block 这件事，在只有 16 个 block 时没有意义——总共就 16 个。工业实现靠 batch × head 维度把程序数拉上去，长序列还会把 K 维切开（split-K）；本章的 kernel 两件事都不做，所以它的实测时间不能用 0.86 µs 去要求。21.4 节会给出完整对比。

两个容易算错的地方。第一，把 164 KB 当成整张卡的 SRAM：它是每 SM 的 shared memory 上限，一张卡几十上百个 SM 各有各的一份，block 之间不共享。第二，以为 block 越大越好：P 按乘积膨胀，驻留数和程序数一起下降，省下来的 K/V 重复读未必抵得过并行度的损失。公式只负责告诉你哪些组合在预算内，预算内哪个快，要测。

Triton 里不直接管 SRAM，旋钮是 `BLOCK_M`、`BLOCK_N`、`num_warps`、`num_stages`。`BLOCK_M`/`BLOCK_N` 取 2 的幂、16 的倍数：`tl.arange` 要求长度是 2 的幂，`tl.dot` 要求矩阵维度至少 16。超了 shared 配额，编译或运行时会直接报错。手算的预算是估算，真实占用以编译产物里的 shared 用量为准（字段名随 Triton 版本变），两者差得多时，先看是不是流水线多开了缓冲。

这一版从 `64×64` 起步：单缓冲 32 KB、双缓冲 48 KB，4 个 warp，约 81 regs/thread，A100 上双缓冲还能驻留 3 个 block。下一组试 `128×64`：双缓冲 64 KB，8 个 warp，驻留 2 个。`128×128` 是这一版的上限：双缓冲 112 KB，驻留 1 个，113 regs/thread。`256×64` 的 162 regs/thread 已经接近上限，驻留只有 1 个 block，不作为首选。

### 第 3 节 · 动手写 fused attention（单 head、单序列版）

这一版只处理单 head、单序列、最后一维连续的 `[N, D]` FP16 输入，D 取 64 或 128，causal mask（因果掩码：第 m 行 query 只能看到前 m 个 key，生成时看不到未来 token）由 `IS_CAUSAL` 开关。它是示意代码，按 Triton 3.x 的 API 写；`tl.dot`、`tl.trans`、`tl.math` 的参数在不同版本间有差异，跑之前按本地版本核对。下面的代码没有在 GPU 上跑过，也不附带实测数字；21.4 节给出验收方法，数字要自己测。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, Out,
              stride_qm, stride_kn, stride_vn, stride_om,
              N, sm_scale,
              D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              IS_CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        keep = mask_n[None, :]
        if IS_CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(keep, qk, -float("inf"))

        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_i = tl.maximum(l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :],
             acc, mask=mask_m[:, None])


def fused_attention(q, k, v, causal=False):
    # q, k, v: [N, D] fp16, CUDA；单 head、单序列
    n, d = q.shape
    out = torch.empty_like(q)
    if n == 0:
        return out
    block_m, block_n = 64, 64
    grid = (triton.cdiv(n, block_m),)
    _attn_fwd[grid](
        q, k, v, out,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0),
        n, 1.0 / (d ** 0.5),
        D=d, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return out
```

`pid_m` 是程序编号，`offs_m` 是这个程序负责的 Q 行号，`offs_d` 是 0 到 D−1 的 head 维下标。`mask_m` 处理 N 不能被 BLOCK_M 整除的尾巴：`q` 的 load 用 `other=0.0` 把越界行填 0，`tl.store` 再用同一个 mask 拦住它们。越界的行照样参与计算，只是写不出去；浪费的是最后不到一块的行，换来的是程序不越界读内存。

`m_i`、`l_i`、`acc` 是 21.1 节的三个运行量，初值 −inf、0、0 对应归纳法的空集。第一轮 `m_new = m_block`，`alpha = exp(−inf − m_new) = 0`，旧状态被清零。如果把 `m_i` 初值写成 0，第一轮的 `alpha = exp(0 − m_new)` 不是 0；虽然它乘的是全零状态，结果碰巧不变，但归纳法的不变式已经不成立，后面一旦改了初值或加了块跳过就会出错。

内层循环里，`offs_n` 是当前 K/V 块的列号，`mask_n` 是尾巴 mask。K 的 load 用 `other=0.0`，越界列读成零向量，`qk` 在那些列上算出来是 0；随后 `tl.where` 把它们置成 −inf，`exp(−inf) = 0`，对 l 和 acc 没有贡献。K 不能 load 成 −inf：`tl.dot` 里 q 的元素乘上 −inf 会得到 NaN，NaN 会污染整个 qk。先 load 0、dot 完再 mask，是这里必须的顺序。

`tl.dot(q, tl.trans(k))` 的形状是 `[BLOCK_M, D] × [D, BLOCK_N] = [BLOCK_M, BLOCK_N]`。`tl.trans` 是一次布局变换，编译器可能借 shared memory 做转置；它换来的是 K 的 load 仍然按行连续，合并访问。直接按 `[D, BLOCK_N]` 读 K 会把每行的步长变成 `stride_kn`，访存不连续，通常更慢。

`qk = qk * sm_scale` 里的 `sm_scale = 1/sqrt(D)`。q、k 的元素是标准正态分布时，QK^T 的每一项是 D 个独立乘积之和，方差为 D，标准差为 sqrt(D)；D = 64 时是 8。除以 8 之后分数的标准差回到 1。不除的话，两个分数的差很容易到 16，权重比 `e^16 ≈ 8.9×10^6`，softmax 被最大项压成近似 one-hot，数值上不健康。缩放放在 dot 之后、mask 之前，顺序不影响结果。

mask 的条件合成一个：`keep = mask_n[None, :]`，causal 打开时再与 `offs_m[:, None] >= offs_n[None, :]` 相与。`>=` 保留了对角线：第 m 行能看到第 m 个 key 本身，`n = m` 是合法位置；写成 `>` 会把对角线扔掉，整行少一个位置，分母和输出都错。两个条件合成一个 `tl.where`，比写两次少遍历一遍 qk。

`m_block = tl.max(qk, axis=1)` 是块内每行的最大值；`m_new = tl.maximum(m_i, m_block)` 是更新后的运行最大值；`alpha = tl.exp(m_i − m_new)` 是 rescale 系数，指数输入 ≤ 0，所以 `0 < alpha ≤ 1`。`p = tl.exp(qk − m_new[:, None])` 是未归一化的块内概率，`m_new[:, None]` 把 `[BLOCK_M]` 广播成 `[BLOCK_M, 1]`。`l_i = l_i * alpha + tl.sum(p, axis=1)` 对应 21.1 节的 `l' = l·α + rowsum(P_block)`。

`acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)` 是最容易写错的一行。`alpha` 必须同时乘在 `l_i` 和 `acc` 上：两者共享同一个基准，只换一个就把分子分母的比例改了。用 21.1 节的数：第一块 `Õ₁ = 23.678794`，第二块 `α = 0.135335`，`Õ_b = 51.036383`；正确更新是 `23.678794×0.135335 + 51.036383 = 54.240960`，`O = 54.240960/1.553002 = 34.926527`。如果 `acc` 漏乘 `alpha`，`Õ' = 23.678794 + 51.036383 = 74.715177`，`O = 48.110168`，与正确值差 13.18。这个差距远大于 FP16 舍入，验收时一眼就能看出来。

`p.to(v.dtype)` 把 FP32 的 p 转成 FP16，因为 `tl.dot` 要求两个操作数同类型，FP16 输入才能走 tensor core。转换本身有舍入：FP16 的机器精度是 `2^-10 ≈ 9.77×10^-4`，单次舍入的相对误差约 `2^-11 ≈ 4.88×10^-4`。这是输出误差的主要来源之一。输出是 V 的加权平均，V 的元素量级为 O(1)，绝对误差在 10^-3 量级；再叠加参考实现与 kernel 的累加顺序差异，1e-2 的验收容限是合理的。把 p 留在 FP32 再 dot 并不划算：要么走 TF32（TensorFloat-32，19 位格式：8 位指数、10 位尾数；算力减半），要么走 IEEE FP32（慢一个数量级），两条路都比 FP16 的 P 差。

`tl.dot` 的累加精度：FP16 输入默认以 FP32 累加，`out_dtype` 默认是 `tl.float32`；如果显式写 `out_dtype=tl.float16`，累加也降到 FP16，1024 项的累加误差会明显变大。输入是 FP32 时，A100 上的 `tl.dot` 默认走 TF32（10 位尾数），要全精度得传 `input_precision="ieee"`，吞吐会降。这个 kernel 的输入是 FP16，走的是 FP16×FP16→FP32 的路径。

`tl.exp` 在 Triton 里通常落到 libdevice 的 exp，精度高但比 `exp2` 慢。FA2 一类实现把 `sm_scale` 乘上 `log2(e) = 1.442695`，用 `tl.math.exp2` 做指数，把一次 exp 换成一次乘法和一次 exp2；NVIDIA GPU 上 exp2 有硬件近似指令，通常更快。换之前先用同一组输入比较最大误差；本章保留 `tl.exp`，因为读起来和公式一一对应。

循环结束，`l_i = tl.maximum(l_i, 1.0)` 处理整行没有合法 key 的情况。只要一行至少有一个合法 key，`p` 里最大的一项就是 `exp(0) = 1`，所以 `l_i ≥ 1`；把下界钳到 1 不会改变任何合法行的结果，只把空行的 0 变成 1，避免除零。它修不了 `m_new = −inf` 带来的 NaN：如果整行都被 mask，`alpha = exp(−inf − (−inf))` 在更新 l 之前就已经是 NaN。标准循环从 `start_n = 0` 开始，第一块一定含第 0 列，而 causal 条件下列 0 对任何行都合法，所以每一行的 `m_i` 在第一轮就会变成有限值；只有当你跳过整块、或者 N = 0 时，才需要额外保证每行至少处理一个合法块。

`tl.store` 用 `mask_m` 拦住越界的 Q 行。整个 kernel 里 Q 只在循环外读一次，K/V 每块读一次，qk 和 p 只活在片上——这就是 21.1 节换来的搬运账。

边界清单：

- N 不能被 BLOCK_N 整除：K/V load 用 `other=0.0`，qk 在 dot 之后用 `mask_n` 置 −inf；顺序反了会出 NaN。
- N 不能被 BLOCK_M 整除：Q 的 load 和 store 共用 `mask_m`；tail 行计算但不写出。
- 整行被 mask：`m_new = −inf` 会让 `alpha` 变成 NaN；标准循环的首块含列 0，天然避免；加块跳过时要重新保证。
- N = 0：`grid = 0`，kernel 不启动；wrapper 直接返回空张量。
- D 不是 2 的幂：`tl.arange(0, D)` 要求 2 的幂；把 D 补到下一个 2 的幂，load 时 mask 掉多余通道，dot 的贡献为 0。D 至少 16，否则 `tl.dot` 不接受。

用 N = 128、D = 64、BLOCK_M = BLOCK_N = 64 验算 FLOPs：grid = ceil(128/64) = 2 个程序；每个程序走 128/64 = 2 步；每步两次 `tl.dot`，每次 `64×64×64 = 262,144` 次乘加，折合 524,288 FLOPs；每个程序 2×2×524,288 = 2,097,152；两个程序 4,194,304，正好等于 `4N²d = 4×128²×64`。causal 版这一版仍然算满整个方阵，FLOPs 不变；21.5 节讲怎么跳过整块。

参考实现与最小验收：

```python
import math
import torch
import torch.nn.functional as F


def ref_attention(q, k, v, causal=False):
    n, d = q.shape
    s = (q.float() @ k.float().T) * (1.0 / math.sqrt(d))
    if causal:
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return (p @ v.float()).to(q.dtype)


torch.manual_seed(0)
n, d = 1024, 64
q = torch.randn(n, d, dtype=torch.float16, device="cuda")
k = torch.randn(n, d, dtype=torch.float16, device="cuda")
v = torch.randn(n, d, dtype=torch.float16, device="cuda")

out = fused_attention(q, k, v)
ref = ref_attention(q, k, v)
print("max abs err vs FP32 reference:", (out.float() - ref.float()).abs().max().item())

out_sdpa = F.scaled_dot_product_attention(
    q[None, None], k[None, None], v[None, None]
)[0, 0]
print("max abs err vs SDPA:", (out.float() - out_sdpa.float()).abs().max().item())
```

`ref_attention` 用 FP32 算 S、softmax 和 PV，再转回 FP16，是不依赖 kernel 的数值基准；SDPA（Scaled Dot-Product Attention，缩放点积注意力：PyTorch 的 `scaled_dot_product_attention` 接口）是第二个对照，也是 21.4 节的计时基线。N = 1024 时 S 是 1024×1024 的 FP32，占 4 MB，单卡放得下；N 再大就用 SDPA 做基准，别在 PyTorch 里物化更大的方阵。预期的最大绝对误差在 1e-2 量级；如果看到 1e-1 以上或 NaN，先查 mask 和 rescale，不要先调容限。

这一版刻意没做的事：没有 causal 整块跳过，没有 split-K，没有 persistent scheduling，没有 autotune，没有把 K 预转置成 `[D, N]`，也没有 `exp2`。这些是第 23 章读 FA2/FA3 的内容。先把正确性和 SRAM 账做对，再谈这些优化。

### 第 4 节 · 验收：正确性 + 计时 + Roofline 定位

验收的顺序是正确性、计时、Roofline。先对一个没验证过的 kernel 计时，测出来的数字没有意义；先标 Roofline 点再看实现，容易把并行度不足误判成实现质量。

正确性用 21.3 节的 `ref_attention` 做 FP32 基准，再用 SDPA 交叉对照。至少覆盖四组：N = 1024（主测试点）、N = 1000（不能被 64 整除，同时触发 BLOCK_M 与 BLOCK_N 的尾巴）、N = 256（短序列）、causal 开与关。D = 64。判据是最大绝对误差 ≤ 1e-2（预期值，不是实测）；超过 1e-1 或出现 NaN，按逻辑错误处理。几个典型症状：`acc` 漏乘 `alpha` 时，21.1 节的四元素例子会偏 13.18；尾巴没 mask 干净时，多出来的位置给 l 加 1，`l` 从 1.553002 涨到 2.553002，涨 64.4%；causal 写成 `>` 时，对角线被扔掉，第 0 行整行没有合法位置，直接 NaN，其他行的误差也会明显变大；K load 成 −inf 时直接 NaN。

| 检查 | 形状 | 判据 | 实测 |
|------|------|------|------|
| 对照 FP32 参考，非 causal | N=1024, 1000, 256 | 最大绝对误差 ≤ 1e-2 | 待补测 |
| 对照 SDPA，非 causal | N=1024, 1000, 256 | 最大绝对误差 ≤ 1e-2 | 待补测 |
| 对照 FP32 参考，causal | N=1024, 1000, 256 | 最大绝对误差 ≤ 1e-2 | 待补测 |
| 对角线检查，causal | N=128 | 第 0 行输出等于 `v[0]`（误差 ≤ 1e-2） | 待补测 |

计时用 `triton.testing.do_bench`：

```python
from triton.testing import do_bench

t_kernel = do_bench(lambda: fused_attention(q, k, v),
                    warmup=25, rep=100, return_mode="median")
t_sdpa = do_bench(
    lambda: F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None]),
    warmup=25, rep=100, return_mode="median",
)
print(t_kernel, t_sdpa)
```

`do_bench` 的默认返回是均值（`return_mode="mean"`），不是中位数；上面的代码显式传了 `"median"`。均值和中位数都可以用，但两组对比必须用同一种，否则差距里混进了统计口径。以你装的版本的函数签名为准；Triton 3.6.x 的默认是 `mean`，文档字符串里的 median 与实现对不上。

`do_bench` 在每次计时前会用一个大缓冲区把 L2 清掉，所以它测到的是冷 L2 的数；如果你自己写循环且不清缓存，Q/K/V 可能留在 L2 里，数字会偏小。记录时写清口径。另一个细节：清缓存用写操作实现时，缓冲区留下的脏行可能在下一个 kernel 的计时窗口里写回；这个开销随 Triton 版本变化，微秒级 kernel 尤其要留意。

计时对比要同进程、同形状、同 dtype。SDPA 背后可能选 flash、mem-efficient 或 math 后端，记录 `torch.__version__`、CUDA、Triton 版本和 GPU 型号；跨版本比较没有意义。N = 1024 时理论下界是 0.86 µs，比一次 kernel 启动和计时噪声还小；单 head 又只有 16 个 CTA（见 21.2 节），实测时间会明显高于下界。这不是实现失败的证据，先把这两笔账算进去再解读数字。

| 实现 | N | d | causal | 中位耗时 | 实测算力 | 占 312 TFLOPS | 备注 |
|------|---|---|--------|----------|----------|----------------|------|
| Triton 64×64 | 1024 | 64 | 否 | 待补测 | 待补测 | 待补测 | num_warps=4, num_stages=2 |
| Triton 128×64 | 1024 | 64 | 否 | 待补测 | 待补测 | 待补测 | num_warps=8, num_stages=2 |
| SDPA | 1024 | 64 | 否 | 待补测 | 待补测 | 待补测 | 记录 PyTorch 版本 |
| 朴素 FP16（物化 S/P） | 1024 | 64 | 否 | 待补测 | 待补测 | 待补测 | 同一 dtype，口径一致 |

实测算力 = 268,435,456 FLOPs ÷ 耗时；耗时用秒。比如 10 µs 对应 26.8 TFLOPS，是 312 TFLOPS 的 8.6%。这是算术示例，不是实测。

Roofline 逐项算：

| 量 | 算式 | 结果 |
|----|------|------|
| 计算量 | 4·N²·d = 4×1024²×64 | 268,435,456 FLOPs |
| fused 搬运 | 8·N·d = 8×1024×64 | 524,288 B = 512 KiB |
| fused 算术强度 | 268,435,456 ÷ 524,288 | 512 FLOP/字节 |
| 朴素额外搬运 | 8·N² = 8×1024² | 8,388,608 B = 8 MiB |
| 朴素总搬运 | 512 KiB + 8 MiB | 8,912,896 B ≈ 8.5 MiB |
| 朴素算术强度 | 268,435,456 ÷ 8,912,896 | 30.1 FLOP/字节 |
| A100 脊点 | 312 TFLOPS ÷ 2 TB/s | 156 FLOP/字节 |
| fused 计算下界 | 268,435,456 ÷ 312×10¹² | 0.86 µs |
| fused 搬运下界 | 524,288 ÷ 2×10¹² | 0.26 µs |
| 朴素搬运下界 | 8,912,896 ÷ 2×10¹² | 4.46 µs |
| 下界之比 | 4.46 ÷ 0.86 | 5.2 |

每一行的来历：

- 计算量：QK^T 产生 N×N 个分数，每个 d 次乘加，乘加算 2 FLOPs，所以 2N²d；PV 同样是 N×N 个输出、每个 d 次乘加，2N²d；合计 4N²d。
- fused 搬运：Q、K、V、O 各 N×d 个 FP16，每个 2 字节，四个张量合计 8Nd。
- 朴素额外搬运：S 和 P 各 N×N 个 FP16；每个写一次、读一次，2 个矩阵 × 2 字节 × 2 次访问 × N² = 8N²。
- 脊点：312 TFLOPS ÷ 2 TB/s = 156 FLOP/字节。
- 时间下界：字节数 ÷ 2 TB/s，或 FLOPs ÷ 312 TFLOPS。fused 的计算下界 0.86 µs > 搬运下界 0.26 µs，是计算受限；朴素版的搬运下界 4.46 µs > 计算下界 0.86 µs，是带宽受限。

两个比值要分开说。搬运量：朴素版总共 8,912,896 B，是 fused 版 524,288 B 的 17 倍；其中 S、P 的额外 8 MiB 正好是 fused 基础搬运量的 16 倍。时间下界：4.46 µs ÷ 0.86 µs = 5.2 倍，不是 17 倍。原因是两侧的瓶颈不同：朴素版的时间由 8.5 MiB 的搬运决定，fused 版由 0.268 GFLOP 的计算决定；搬运量涨了 17 倍，时间只涨 5.2 倍。

算术强度随 N 变：AI = N/2。N = 256 时 128，低于脊点 156，同一 kernel 在短序列上转成带宽受限；N = 512 时 256，超过脊点；N = 2048 时 1024；N = 4096 时 2048。脊点对应的序列长度是 N = 312。「fused attention 是计算受限」这句话只在 N 大于 312 时成立，N = 1024 的测试点只是其中一个点。

512 KiB 是 HBM 口径，前提是 K/V 在 L2 里被后续 Q block 复用。单 head、N = 1024、Br = 64 时，每个 Q block 都要扫一遍全部 K/V，16 个 block 合计发出约 16×256 KiB = 4 MiB 的 K/V load 请求；K+V 本身只有 256 KiB，A100 的 40 MB L2 放得下，所以 HBM 流量可以接近 512 KiB。如果 L2 放不下、或者被别的数据挤掉，HBM 流量会接近 4.25 MiB（4 MiB K/V + 256 KiB Q/O），AI 掉到约 60，落到脊点以下。这个边界决定了 512 KiB 的账在什么条件下成立。

把点标到 Roofline 图上：横轴 AI = 512，纵轴实测算力。点落在脊点右侧，理论上属于计算受限；如果离 312 TFLOPS 很远，差距来自三处：程序数只有 16 个（21.2 节），`tl.dot` 之外的 `exp`、`max`、求和走不了 tensor core，以及 `tl.dot` 本身的利用率。三处各占多少，要看 profiler，不能凭 Roofline 一张图断定。没有实测时，表里的数字写「待补测」，不要填估算值冒充实测。

如果测的是 causal 版且做了整块跳过，FLOPs 不再是 4N²d，Roofline 的横轴要按实际有用的 FLOPs 重算；否则点会落在错误的位置。21.5 节给出 causal 的块级账。

### 第 5 节 · 小结：在 Roofline 图上标点 + causal mask 练习

Roofline 图上的点是一次验收的收尾：横轴是 21.4 节算出的 AI = 512，纵轴是 `do_bench` 测出的实测算力。点贴着 312 TFLOPS 的水平线，说明实现兑现了理论潜力；点远在下方，说明计算下界和实现之间还有距离。距离的归因要谨慎：单 head 只有 16 个 CTA 这一条，就足以让点远离水平线，不能全记在 `tl.dot` 头上。写结论时把「算出来的」和「测出来的」分开，测不出来的部分写「待补测」。

本章的链条是四步。21.1 节证明分块与整行 softmax 精确相等，给融合开了绿灯；21.2 节算出 SRAM、寄存器和程序数的约束，说明 block 参数不能抄；21.3 节把三条更新式逐行翻成 kernel，每一行都能在公式里找到对应；21.4 节用正确性、计时和 Roofline 验收，把理论账和实测账放在一起。四步里任何一步断了，后面的结论都站不住：rescale 推错，正确性没有根；SRAM 账没算，参数就是猜的；Roofline 没标点，「快不快」只剩感觉。

causal mask 是这一版最直接的扩展。条件写成：

```python
keep = mask_n[None, :]
if IS_CAUSAL:
    keep = keep & (offs_m[:, None] >= offs_n[None, :])
qk = tl.where(keep, qk, -float("inf"))
```

`offs_m >= offs_n` 保留 `n ≤ m` 的位置，正好是第 m 行 query 能看到的前 m+1 个 key。`>=` 不能写成 `>`：第 m 行能看到第 m 个 key 本身，对角线是合法的；写成 `>` 会把对角线 mask 掉，整行少一个位置，分母和输出都错一位。

块级的账可以数出来。N = 128、BLOCK_M = BLOCK_N = 64 时，分数矩阵分成四个 64×64 的 tile：

- 行 0–63、列 0–63：块内三角，一半保留一半 mask；
- 行 0–63、列 64–127：行号最大 63、列号最小 64，全部 `n > m`，整块 mask；
- 行 64–127、列 0–63：行号最小 64、列号最大 63，全部 `n < m`，整块保留；
- 行 64–127、列 64–127：块内三角，一半保留一半 mask。

整块保留和整块 mask 各占四分之一，两个对角块各占四分之一。整个 N×N 矩阵保留 `N(N+1)/2 = 128×129/2 = 8,256` 个位置，mask 掉 16,384 − 8,256 = 8,128 个，占 49.6%。causal 版这一版仍然算满 4N²d = 4,194,304 FLOPs（N = 128），真正有用的只有 8,256/16,384 = 50.4%，即 2,113,536 FLOPs；另一半算完就扔。

整块 mask 的 tile 可以跳过。对第 `pid_m` 个 Q block，行号范围是 `[pid_m·BLOCK_M, (pid_m+1)·BLOCK_M − 1]`；当 `start_n > (pid_m+1)·BLOCK_M − 1` 时，这个 K block 的所有列都大于所有行号，整块 mask。把内层循环的上界从 N 改成 `min((pid_m+1)·BLOCK_M, N)`，就能省掉这些块。跳块之后要保证每行至少处理一个合法块：循环仍然从 `start_n = 0` 开始，第一块含第 0 列，而 causal 条件下列 0 对任何行都合法，所以 `m_i` 在第一轮就会变成有限值，`alpha` 不会遇到 `−inf − (−inf)`。把循环起点也往后挪会破坏这条保证，需要显式守卫。

改完用同一套参考验证：`ref_attention(..., causal=True)` 和 `F.scaled_dot_product_attention(..., is_causal=True)`，最大绝对误差预期仍在 1e-2 量级。误差突然变大时，先查 `>=` 是不是写成了 `>`，再查合并后的 `keep` 有没有把尾巴 mask 挤掉。整块跳过只改变计算量，不改变数学：被跳过的块对 l 和 Õ 的贡献是 0，归纳法的不变式照样成立。

本章的结论有两条：分块不改变结果，改变的是中间结果放在哪；block 大小由 SRAM、寄存器和程序数三者共同决定，不是抄来的默认值。把 causal mask 加上、把整块跳过加上，再往下读第 23 章的 FA2/FA3，看到的就是同一套数学在不同并行策略下的工程化。

**延伸**：FlashAttention 原论文（Dao 等，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)，2022）：IO-aware 的分块 attention 与 IO 复杂度分析。

**延伸**：[Triton 官方教程 `06-fused-attention`](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)：与本版结构相同的参考实现，含 causal 与多头扩展。

**延伸**：[Triton `tl.dot` 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)：输入精度、`out_dtype` 与 `input_precision` 的语义。

**延伸**：[PyTorch SDPA 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：`is_causal` 与后端选择。

**延伸**：[Triton `do_bench` 源码](https://github.com/triton-lang/triton/blob/release/3.6.x/python/triton/testing.py)：`return_mode` 的默认值与每次计时前的 L2 清缓存实现。

## 第 22 章 · 量化 Matmul 深讲：量化方案本身

一个 INT8 权重的 4096×4096 matmul 在 A100 上能快多少？M=32 的 decode 形状下，FP16 要搬 34.08 MB，INT8 只搬 17.17 MB，时间上限几乎跟着减半；M=1024 的 prefill 形状下，算术强度 682.7 已经在脊点右边，搬运减半换不成时间，能指望的只有 INT8 tensor core 的峰值从 312 TFLOPS 抬到 624 TOPS。同一个量化方案，两档形状的收益来源不同；哪一笔能兑现，先看 scale 的粒度把 kernel 定成了什么结构。

第 12 章算过量化的误差口径：格点、scale、离群值与质量表。本章把量化方案当作输入，只追它在 kernel 里的三个后果：scale 的形状与数量、dequant 发生的位置、最终落到 Roofline 图的哪一侧。三条线索对应五节：scale 粒度（22.1）、INT8 `tl.dot` 路径（22.2）、fused dequant（22.3）、autotune 与计时（22.4）、Roofline 收口（22.5）。

**本章目标。** 给定一种 scale 粒度（per-tensor / per-channel / per-token / per-group）和一档目标形状（decode 的 M=32 或 prefill 的 M=1024），能判断 scale 能否提到 K 循环外、dequant 应融进 matmul 还是单独物化，并写出对应的 Triton kernel，用 `triton.autotune`、`do_bench` 和 Roofline 给出可复现的验收口径。

### 第 1 节 · Scale 怎么来：三种粒度与误差从哪来

对称 INT8 量化的 scale 只有一条来源：`s = max|w| / 127`，`w_q = round(w / s)` 钳到 `[-127, 127]`，反量化是 `w' = w_q · s`。舍入让每个元素最多偏离半个格点，所以 `|w − w'| ≤ s / 2`；`s` 由管辖范围内的最大绝对值决定，离群值越大，`s` 越大，小值的绝对误差上界跟着变大。三种粒度的区别，就是“管辖范围”有多大；误差算例与相对误差的读法在第 12 章，本节只看粒度在 kernel 里留下什么。

per-tensor 的 scale 是一个标量；per-channel 给每个输出列一个 scale，权重常用；per-token 给每个输入行（token）一个 scale，激活常用。权重训练完就固定，per-channel static 一次算好；激活每次前向都变，per-token 只能 dynamic（动态）现算，或者用校准集定 static、赌分布不变。误差的来源是同一个：`s` 被该范围内最大的元素顶大，其余元素的误差上界是 `s/2`。粒度越细，离群值能污染的 scale 越少，误差越小；代价是 scale 的数量增加，而且 scale 依赖的下标会改写 kernel 的结构。

还有一档更细的：per-group 把 K 切成每 `g` 个一组，每组一个 scale。以权重 B 为例，scale 的形状是 `[K/g, N]`，分组把误差压得更小，但 `s_b` 从此依赖 k。四种粒度的 kernel 后果：

| 粒度 | scale 的形状 | 数量 | 依赖的下标 | 能否提到 K 循环外 |
|---|---|---|---|---|
| per-tensor | 标量 | 1 | 无 | 能 |
| per-channel | `[N]` | N | 输出列 j | 能 |
| per-token | `[M]` | M | 输出行 i | 能 |
| per-group | `[K/g, N]` | (K/g)·N | 输出列 j 与 k 所在的组 | 不能 |

前三行的 scale 对 k 是常数，可以从 K 维求和里提出来，等整数累加完再乘一次；第四行必须在 K 方向分组处理。至于 dequant 本身发生在循环外还是循环里，还取决于权重是不是以 int8 分块读入：22.2 的 epilogue 在循环外乘 scale，22.3 的 fused kernel 在循环里展开权重，两件事的判据不同。

scale 本身的访存很小：N=4096 时 per-channel 的 `s_b` 是 16 KB，M=1024 时 per-token 的 `s_a` 是 4 KB，相对 16.78 MB 的 int8 权重可以忽略。scale 的成本不在字节数，在它出现的位置：能提到循环外，它就是 epilogue 里的一次乘法；提不出去，它就在 K 循环里每块都要乘一次。

dynamic 的代价则可以单独算。per-token 要在每次前向里现算每行的 amax：以 M=1024、K=4096 为例，读 fp16 激活 8.39 MB、写 int8 激活 4.19 MB，多出 12.58 MB 搬运；同一形状 FP16 matmul 的总搬运是 50.33 MB，量化准备占 25.0%。M=32 时同样是读 0.26 MB、写 0.13 MB，共 0.39 MB，只占总搬运 34.08 MB 的 1.2%。dynamic 的代价随 M 线性增长：decode 形状付得起，prefill 形状要先算这笔账。static 用校准集把 scale 离线定好，省掉这次 pass，代价是分布失配的风险；校准集怎么选、失配误差怎么量，在第 12 章。

本节只讨论对称量化（zero point 固定为 0）；非对称量化多一个零点，scale 的代数不变，kernel 里多一次减法，后面两条路径的结构不变。

### 第 2 节 · INT8 `tl.dot` 路径：int8 输入、int32 累加、epilogue 再乘 scale

A（激活）与 B（权重）都以 int8 存：A 的形状 `[M, K]`，per-token scale `s_a` 形状 `[M]`；B 的形状 `[K, N]`，per-channel scale `s_b` 形状 `[N]`。`tl.dot` 做 int8×int8→int32 的乘加，累加器是整数；scale 等到 K 循环结束再乘。能这样安排，是因为两个 scale 都不依赖 k：

```text
Σ_k (a_q[i,k] · s_a[i]) · (b_q[k,j] · s_b[j])
  = s_a[i] · s_b[j] · Σ_k a_q[i,k] · b_q[k,j]
```

取 `s_a = 0.01`、`s_b = 0.02`，K 方向两项，`a = [3, -5]`、`b = [7, 2]`。先 dequant 再乘：`0.03×0.14 + (-0.05)×0.04 = 0.0022`。先整数点积再乘：`0.01×0.02×(21 − 10) = 0.0002×11 = 0.0022`。两边相等；K 方向有多少项，这个提法就省多少次浮点乘。

如果 scale 沿着 K 变，提法失效。per-group 量化把 K 切成 `g` 个一组，每组一个 `s_b[g, j]`，求和里每一项的 scale 都不同，只能每组单独 dequant 后再累加，或者把每组的部分和分别乘上该组 scale 再加起来。22.3 的 fused kernel 也在循环里乘 scale，但原因是权重以 int8 分块读入、必须在进 `tl.dot` 前展开，不是因为 scale 依赖 k；两种“循环内乘 scale”长得像，理由不同。

量化用 PyTorch 做，这部分是数据准备，不是 kernel 本身：

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

`clamp_min(1e-8)` 是防除零的护栏，不是精度参数：某一行或某一列全为 0 时，`amax` 是 0，`scale` 会是 0，`x / scale` 得 inf 或 NaN；把下限钉在 1e-8 之后，全零行量化后仍是全零，反量化也是 0。除的是 127.0 而不是 128.0，对应上一节的对称约定：最大值落在 127 上。

INT8 matmul kernel（示意，按当前 Triton 版本调整，未在本书环境编译运行）：

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
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
                    other=0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
                    other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

这段 kernel 与第 20 章的 FP16 matmul 结构相同：分组映射、K 循环、`tl.dot` 累加、带 mask 写回。变的只有三处：输入是 int8，累加器是 `tl.int32`，epilogue 多了一行 `acc.to(tl.float32) * sa[:, None] * sb[None, :]`。第 20 章调通的 tile 逻辑与 mask 写法可以复用；这三处就是量化方案落地的全部内容。

指针步进有一处容易写错：`b_ptrs += BLOCK_K * stride_bk`，`stride_bk` 是 B 在 K 方向的行步长（行主序下等于 N），不是 `stride_bn`。写成 `stride_bn` 时，如果 B 是连续的行主序 `[K, N]`，`stride_bn = 1`，指针每步只挪 BLOCK_K 个元素而不是 BLOCK_K 行；形状对得上，读进来的却是错位的数据，结果全错且不会报越界。A 的步进用 `stride_ak`，同理。

累加器必须是 int32。int8 的最大值是 127，两个 127 相乘是 16129；一个 `BLOCK_K=32` 的局部和最坏是 `127×127×32 = 516,128`，是 int16 上限 32,767 的约 15.8 倍——第一个 K 块就会溢出成负数。K=4096 全长累加的最坏值是 `127×127×4096 = 66,064,384`，占 int32 上限 2,147,483,647 的 3.1%，余量约 32 倍。int16 不是精度差一点，是算到一半就错；int32 是唯一装得下的类型。

mask 要同时盖住 M/N 与 K 两个方向。M=32、`BLOCK_M=64` 时，tile 的后 32 行指向 A 之外；只有 K 维的 `k_rem` mask 管不住它。没有 M 维 mask，轻则读进无关数据、结果错得无声，重则越界访问报错。两个 mask 是正交的：K mask 管“这一块 K 读完没有”，M/N mask 管“这一行/列存不存在”。第 20 章的方阵练习里 M、N、K 都是 BLOCK 的整倍数，这个缺口藏得住；decode 的 M=32 是第一个让它露出来的形状。

`tl.dot` 的输入约束按当前 Triton 文档：两个操作数都是二维或三维、内维一致，元素类型支持 int8、float8_e5m2、fp16、bf16、fp32，累加器支持 fp16、fp32、int32；int8 输入的 `out_dtype` 显式写 `tl.int32`。文档没有把 M/N/K 的最小值写成硬性条款，实际能不能落到 tensor core 由后端决定：Ampere 的 INT8 MMA 指令形状是 m16n8k32（PTX 的 `mma.sync.aligned.m16n8k32` 一族），所以 M/N 取 16 的倍数、K 取 32 的倍数最稳，`BLOCK_M=64, BLOCK_N=64, BLOCK_K=32` 就是照这条约束取的。换卡、换 Triton 版本时，先看编译输出有没有落到 tensor core，再看计时；具体形状与对齐要求以你装的版本为准。

正确性用相对误差比。量化误差按比例走，固定绝对容限没有意义；下面的检查把分母下限钉在 1，避免参考值接近 0 时相对误差爆炸：

```python
torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16, device='cuda')
b = torch.randn(256, 128, dtype=torch.float16, device='cuda')
a_q, sa = quant_per_token(a)
b_q, sb = quant_per_channel(b)
c = matmul_int8(a_q, b_q, sa, sb)
ref = a @ b
rel_err = ((c.float() - ref.float()).abs()
           / ref.float().abs().clamp_min(1)).mean()
print('mean relative error:', rel_err.item())
```

预期平均相对误差在 1% 量级（算例，不是实测）；大一个数量级时，先查 scale 的粒度是否弄反（per-token 与 per-channel），再查累加器是不是 int32。输出是 fp16，本身也带一次舍入；要单独看量化误差，把输出留成 fp32 再比。

epilogue 的代价可以逐项算。每个输出元素做 4 次操作：int32→fp32、乘 `sa`、乘 `sb`、fp32→fp16。M=32、N=4096 时输出 131,072 个元素，4 次操作共 524,288 次；总 FLOPs 是 `2×32×4096×4096 = 1,073,741,824`，占比 0.049%。M=1024 时输出 4,194,304 个元素，操作数 16,777,216 次；总 FLOPs 34,359,738,368，占比还是 0.049%。两个形状同占比不是巧合：比例是 `4MN / (2MNK) = 2/K`，K=4096 时就是 0.049%；K 缩到 256，比例升到 0.78%。

这 0.049% 是“操作数占 tensor core FLOPs 的比例”，不是“时间占比”。epilogue 的 4 次操作跑在 CUDA core 上，A100 的 FP32 峰值约 19.5 TFLOPS，tensor core 的 INT8 峰值是 624 TOPS，两者差 32 倍；按时间算，同样的操作数占比会被放大。K=4096 时它仍然很小，K 很小时就不能忽略。把这点写清楚，才不会把“epilogue 只占 0.05% FLOPs”误读成“epilogue 不花时间”。

INT8 相对 FP16 的收益来自两笔独立的账，不要混成 4 倍。A100 的公开规格是稠密 FP16 312 TFLOPS、稠密 INT8 624 TOPS；这两个数不含结构化稀疏，稀疏会把两者各自再翻一倍（624 TFLOPS、1248 TOPS），但 2:4 稀疏要求权重有特定模式，量化方案不保证这一点，本章的账按稠密算。两笔账是：tensor core 峰值翻倍（抬高计算上限），权重每元素从 2 字节降到 1 字节（减少搬运量）。它们落在不同的形状上——decode 形状吃搬运减半，prefill 与方阵吃峰值翻倍；两笔账不会乘成 4 倍：一个形状通常只兑现其中一笔。

### 第 3 节 · Fused dequant+matmul：dequant 融进 matmul，不单独物化 FP16 权重

只量化权重时，激活保持 fp16。权重以 int8 存在 HBM，要用它做 fp16 matmul，有两条路：先跑一个 dequant kernel 把整块权重展开成 fp16 写回 HBM，再跑 matmul 读回来；或者在 matmul 的 K 循环里读 int8 tile，当场乘 scale 转成 fp16，直接喂给 `tl.dot`。前者是两步基线，后者是 fused dequant。图 22-1 把位宽与权重体积放在一起（与第 12 章共用）：16 位到 8 位，权重每元素从 2 字节降到 1 字节；本节只用到这一格。以 4096×4096 的权重为例，两步基线里权重每元素要过手 5 字节——读 int8 1 字节、写 fp16 2 字节、再读 fp16 2 字节；融合版只读 1 字节。

按整块权重算：dequant 读 16.78 MB、写 33.55 MB，matmul 再读 33.55 MB，权重这一项一共 83.89 MB；融合版只读 16.78 MB，省下 67.11 MB，是融合版自身权重搬运的 4 倍。把激活与输出加进来，M=32 的两步基线总搬运约 84.41 MB，融合版 17.30 MB；对照的直接 FP16 matmul（权重已经是 fp16）是 34.08 MB。三个数放在一起，才分得清“融合相对两步基线”和“量化相对 FP16”是两笔不同的账：前者省 79.5%，后者省 49.2%。

两步基线多出来的两次 fp16 读写，读写的还是同一份权重：先写出去，再读回来，中间没有别的 kernel 消费它。融合要消灭的就是这个往返——数据写出去又读回来，纯粹是格式转换的过路费。

Fused kernel（A 是 fp16 激活，B 是 int8 权重 + per-channel scale；示意，按当前 Triton 版本调整）：

```python
@triton.jit
def matmul_fused_dequant_kernel(a_ptr, b_ptr, c_ptr,
                                M, N, K,
                                stride_am, stride_ak, stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                scale_b_ptr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
                    other=0.0)
        b_int8 = tl.load(b_ptrs,
                         mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
                         other=0)
        b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

dequant 发生在循环里：`b_int8 = tl.load(...)` 读 int8 tile，下一行 `b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)` 在寄存器里展开成 fp16，紧接着被 `tl.dot` 消费。`b` 没有 HBM 地址，没有 `tl.store` 写它，也没有下一个 kernel 读它；它的生命周期只有一次循环迭代。22.2 的 epilogue 乘 scale 是同一个手法：中间结果留在寄存器里，不落 HBM。

先转 fp32 再转 fp16，是因为 int8 的整数值乘 fp32 scale 的乘积由 fp32 兜底；直接在 fp16 里乘，乘积先舍入到 fp16 再进 tensor core，等于多丢一轮精度。最后的 `.to(tl.float16)` 是喂给 fp16 tensor core 的输入要求。顺序是 fp32 里算准、fp16 里算快。

`sb` 在循环外读一次：scale 不依赖 k，可以提到循环外；`b_int8` 依赖 k，必须在循环里分步读。N=4096 个 fp32 scale 是 16 KB，相对 16.78 MB 的权重可以忽略；融合省的是权重的 67.11 MB，不是这 16 KB。如果权重是 per-group，`sb` 也要按 k 组取，循环外的那次加载要挪进循环里；kernel 结构不变，多一次索引。

指针步进与 mask 的边界同 22.2：`b_ptrs += BLOCK_K * stride_bk`，M/N 与 K 两个方向的 mask 都要齐。

dequant 的 ALU 代价要另算。每个权重元素做 3 次操作：int8→fp32、乘 scale、fp32→fp16。一个权重 tile 会被 M 方向的每个程序块各展开一次，总操作数是 `3 × ⌈M/BLOCK_M⌉ × K × N`，与 matmul 的 `2·M·N·K` 相比，占比是 `3 × ⌈M/BLOCK_M⌉ / (2M)`。M=32、`BLOCK_M=64` 时，M 方向只有一个程序块，占比 4.7%；M=1024、`BLOCK_M=64` 时，16 个程序块各展开一次，占比 2.3%；`BLOCK_M=128` 时降到 1.2%。M=32 时 `BLOCK_M` 从 64 换到 128，向上取整仍是 1，占比不变——dequant 的摊薄靠 M 变大，不靠 tile 变大。

这些也是操作数比例，不是时间比例：dequant 的 3 次操作跑在 CUDA core 上，matmul 跑在 tensor core 上，按时间算占比会被放大；decode 形状下 dequant 与 HBM 读可以重叠，prefill 形状下它会直接占用 issue 带宽。方向是确定的：M 越大，dequant 的相对代价越小。

把 fused 点放回 Roofline（N=K=4096，A100，FP16 峰值 312 TFLOPS，带宽按 2 TB/s 取整）：M=32 时 FP16 的算术强度约 31.5，fused 约 62.1，仍在脊点 156 左边，访存受限，省一半权重搬运大约能换一半时间；M=1024 时 FP16 约 682.7，fused 约 1024，已在脊点右边，计算受限，省下的 16.78 MB 换算不成时间，dequant 的 ALU 反而多花。fused 的 `tl.dot` 仍是 fp16 输入，峰值还是 312 TFLOPS——它没有 INT8 路径那笔峰值翻倍。

“融合总是更快”不成立。融合是一笔交易：拿 ALU 操作换 HBM 往返。访存受限时，HBM 往返是瓶颈，交易划算；计算受限时，瓶颈是 FLOPs，换回来的字节不值钱，付出去的 ALU 还占地方。22.2 的 INT8 路径与这一节的 fused 路径也不是谁替代谁：前者把激活和权重都量化，省得更多，代价是激活的量化误差与 epilogue 的账；后者只动权重，激活保持 fp16，误差小，省得少。

同样的判断标准在第 20 章的 fused softmax 与第 21 章的 FlashAttention 里已经用过：把中间结果留在寄存器或 SRAM，前提是它没有别的消费者。dequant 出来的 fp16 权重只被本次 `tl.dot` 消费，所以可以融；如果同一份权重后面还有别的 kernel 要用，或者 M 方向有多个 tile、每个 tile 都要重新展开一次，物化一次反而可能更省——22.5 会把重复读的边界写进 Roofline 的假设里。

### 第 4 节 · Autotune + 实测：BLOCK 配置扫一遍，`do_bench` 计时

22.2 与 22.3 的 kernel 都写死了 `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`。这组值在 M=32 和 M=1024 上不可能同时最优：decode 形状程序块少，tile 开大只会让更多 SM 空转；prefill 形状程序块多，tile 开大才能提高复用、摊薄 dequant。把配置交给 `triton.autotune` 扫一遍，按形状各选一组。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
                       'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
                       'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_int8_kernel(...):
    ...
```

`key=['M','N','K']` 的意思是：这三个运行时参数一变，autotune 就重新扫配置；`BLOCK_*`、`GROUP_M`、`num_warps`、`num_stages` 是被扫的变量，不放进 key。M=32 与 M=1024 会被当成两个 key 各扫一遍；如果 kernel 还有别的会改变最优配置的运行时参数（比如 per-group 的 group size），也要放进 key，否则会拿错缓存。调优成本是“配置数 × key 数”次编译加计时：4 组配置 × 2 档形状 = 8 次；配置翻倍，成本翻倍。第一次调用时 autotune 把每组都跑一遍并缓存结果，之后的调用直接取缓存。

启动时 grid 要按当前配置算，BLOCK_* 不再从调用处传：

```python
grid = lambda META: (
    triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
)
matmul_int8_kernel[grid](a_q, b_q, c, M, N, K,
    a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1),
    c.stride(0), c.stride(1), scale_a.float(), scale_b.float())
```

计时用 `triton.testing.do_bench`。它的 `warmup=25, rep=100` 是毫秒预算，不是迭代次数：先估一次单次耗时，再按预算换算 warmup 与重复次数；返回值默认是每次运行的均值，单位毫秒。它在每次计时前清 L2，所以量到的是“每次调用从冷 L2 开始”的时间，调用内部的复用仍然发生。自动调参要在正式计时前完成：do_bench 会先调用一次被测函数估时，autotune 的编译与扫配置就发生在那一次；不要把首次调用算进正式计时。

```python
ms = triton.testing.do_bench(
    lambda: matmul_int8(a_q, b_q, sa, sb),
    warmup=25, rep=100, return_mode="median",
)
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
```

耗时以毫秒计，上面的换算是 `2MNK / (t_ms × 1e9)`；换成微秒，M=32 的换算尺是 `TFLOPS ≈ 1073.7 / t(µs)`，M=1024 是 `34359.7 / t(µs)`。基线要在同一形状、同一张卡、同一测量方法下测；两步基线的 dequant 与 matmul 是两个 kernel，`do_bench` 的 lambda 里要把两个都调用，只测 matmul 会把基线报快、把融合收益报小。

两步基线还有一个 L2 陷阱：dequant 写出的 fp16 权重 33.55 MB，A100 的 L2 是 40 MB，matmul 紧接着读它时可能大量命中 L2，没走 HBM。do_bench 只在两次调用之间清 L2，清不掉同一次调用里的这次复用。所以两步基线的实测可能比“5 字节/元素都走 HBM”的估算快；要判断，用 ncu 看 DRAM 读字节，不要只看时间。

程序块数量决定 SM 覆盖的上限。A100 有 108 个 SM。M=32、`BLOCK_M=64`、`BLOCK_N=64` 时，grid 是 1×64=64 个程序块，覆盖约 59% 的 SM；`BLOCK_M` 与 `BLOCK_N` 都换成 128，grid 是 1×32=32 个，只覆盖约 30%。M=1024 时，64×64 的配置有 16×64=1024 个程序块，128×128 有 8×32=256 个，两组都远超 108 个 SM。预期是：decode 偏爱小 tile（程序块多，SM 覆盖全），prefill 偏爱大 tile（复用高，dequant 摊薄）。M=32 时 M 方向只有一个 tile，`GROUP_M` 退化成 1，分组排序不起作用；它只在 M 方向有多个 tile 时才有意义。

表里的数字全部待补测；正文出现的 TFLOPS 都是算例或上限。同一形状下再与 FP16 matmul 比，比值写在同一行：

| 配置（BLOCK_M, BLOCK_N, BLOCK_K, warps, stages） | INT8 M=32 | fused M=32 | INT8 M=1024 | fused M=1024 |
|---|---|---|---|---|
| 64, 64, 32, 4, 3 | 待补测 | 待补测 | 待补测 | 待补测 |
| 128, 64, 32, 4, 3 | 待补测 | 待补测 | 待补测 | 待补测 |
| 64, 128, 32, 8, 3 | 待补测 | 待补测 | 待补测 | 待补测 |
| 128, 128, 32, 8, 4 | 待补测 | 待补测 | 待补测 | 待补测 |
| FP16 matmul（同形状） | 待补测 | 待补测 | 待补测 | 待补测 |
| 两步基线（dequant + matmul） | — | 待补测 | — | 待补测 |

填完表先看两件事。同一形状下换配置，TFLOPS 动不动：动，说明缺口在配置；怎么换都不动，说明缺口在形状本身（程序块太少或访存结构）。INT8 与 fused 各自相对 FP16 的比值是否落在 Roofline 预测的方向上：M=32 的搬运减半应该换来接近 2 倍的上限，M=1024 的 INT8 峰值翻倍应该换来接近 2 倍，fused 在 M=1024 上不该有明显收益。测出来与预测不符，记下来——第 24 章的负面结果分析要的就是这种记录，藏起来才是浪费。

### 第 5 节 · 小结：量化省的是什么（搬运量），代价是什么（精度），在 Roofline 图上标点

把 22.2 与 22.3 的形状代回 Roofline，八个点一次标全（N=K=4096；A100 稠密峰值 312 TFLOPS / 624 TOPS；带宽按 2 TB/s 取整，脊点约 156 FLOP/字节）：

| 点 | 形状 | 算术强度（FLOP/字节） | 区间 | 峰值 |
|---|---|---|---|---|
| FP16 matmul | M=32 | 约 31.5 | 访存受限 | 312 TFLOPS |
| INT8 matmul | M=32 | 约 62.5 | 访存受限 | 624 TOPS |
| fused dequant | M=32 | 约 62.1 | 访存受限 | 312 TFLOPS |
| FP16 matmul | M=1024 | 约 682.7 | 计算受限 | 312 TFLOPS |
| INT8 matmul | M=1024 | 约 1170.3 | 计算受限 | 624 TOPS |
| fused dequant | M=1024 | 约 1024.0 | 计算受限 | 312 TFLOPS |
| FP16 matmul | 4096³ 方阵 | 约 1365.3 | 计算受限 | 312 TFLOPS |
| INT8 matmul | 4096³ 方阵 | 约 2048.0 | 计算受限 | 624 TOPS |

这些算术强度按“每个矩阵只读一次”的理想缓存模型算；真实 kernel 的权重可能被多个 M tile 重复读，L2 能兜住多少要看 tile 顺序与权重面板大小，实测 DRAM 流量用 ncu 看。M=32 时三个矩阵加起来不到 34.1 MB，L2 是 40 MB，理想模型接近成立；更大的形状要靠分块调度逼近这个假设，偏差用 ncu 的 DRAM 流量核对。

M=32 的三个点（31.5、62.5、62.1）都在脊点左边，访存受限：INT8 与 fused 都把权重搬运减半，时间上限接近 2 倍；两者的搬运只差激活那一项（fused 的激活是 fp16 0.26 MB，INT8 的是 int8 0.13 MB，总搬运 17.30 MB 对 17.17 MB，差 0.8%），所以 Roofline 上几乎重合，差别在精度：fused 的激活没有量化误差，INT8 有。

M=1024 的三个点（682.7、1170.3、1024.0）都在脊点右边，计算受限：INT8 靠峰值翻倍把计算时间减半；fused 的峰值仍是 312 TFLOPS，省下的 16.78 MB 换算不成时间，dequant 的 ALU 反而多花，所以 fused 在这个形状上只能持平或更慢。方阵 4096³ 同理，INT8 靠 624 TOPS，fused 没有收益。

权重占搬运的比例决定量化能省多少：M=32 时 FP16 总搬运 34.08 MB 里权重 33.55 MB，占 98.5%，INT8 把总搬运压到 17.17 MB（0.504 倍）；M=1024 时权重占 66.7%，INT8 总搬运 29.36 MB（0.583 倍）；方阵时权重占 33.3%，INT8 总搬运 67.11 MB（0.667 倍）。权重占比从 98.5% 降到 66.7% 再到 33.3%，量化的省法从“几乎减半”退到“省三分之一”。量化最划算的形状是权重占主导的形状，decode 就是这样的形状——不是因为它小，是因为它的搬运里权重占九成八。

代价是精度。22.2 的相对误差是本章唯一能量化精度的口径：预期在 1% 量级，不是实测；完整的质量表在第 12 章，本章不重复。把误差与提速比并排放：

| 指标 | 来源 | 状态 |
|---|---|---|
| 相对误差（INT8 路径） | 22.2 的检查代码 | 预期 1% 量级，待补测 |
| 相对误差（fused 路径） | 只量化权重，预期小于 INT8 | 待补测 |
| 提速比（M=32 / M=1024） | 22.4 的配置-耗时表 | 待补测 |
| 峰值算力 | A100 公开规格，稠密 | 312 TFLOPS / 624 TOPS |

误差的容忍线由下游任务定，提速比的及格线由部署定；标准要在跑之前写下来，跑完对照标准做决定。快 2 倍、误差 5% 的方案可能不能上线；快 1.2 倍、误差 0.5% 的方案可能能上线——这两个数是举例，不是本章的结论。

更新 Roofline 图：横坐标是上表的算术强度，纵坐标是 22.4 的实测算力。实测之前只能标横坐标，纵坐标待补测；点的位置先回答“该看哪一侧”，实测的高度再回答“离上限有多远”。访存受限的三个点，差在带宽利用率与程序块数量；计算受限的点，差在 tensor core 利用率与 tile 配置。

量化方案本身省的是搬运量：权重每元素 2 字节变 1 字节，算术强度向右移。INT8 路径另外把 tensor core 的峰值从 312 TFLOPS 抬到 624 TOPS，这是硬件路径的账，不改变 matmul 的 FLOPs 数量；fused 路径没有这笔，只省搬运。代价是精度：激活也量化时误差更大，只量化权重时误差小、省得也少。形状决定哪一笔兑现：权重占搬运九成八的 decode 是量化的主场，计算受限的 prefill 与方阵只能吃峰值那一笔。

**延伸**：[Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)（分组排序、autotune 与 `do_bench` 用法）、[`tl.dot` 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)（输入类型与累加器）、[`do_bench` 文档](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)（warmup/rep 与返回值）、[NVIDIA A100 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)（稠密与稀疏规格）、[PTX ISA 的 mma 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)（INT8 m16n8k32）。

## 第 23 章 · FlashAttention 2/3 与 CUTLASS：读工业级实现

第 21 章写出的 fused attention 在数学上是对的：Q 的一块常驻，K/V 分块流式读入，用 online softmax 维护运行最大值、归一化因子和输出累加器，不物化 S、P 两个 N×N 矩阵。同一套数学搬进工业实现，性能却可能差出一大截。FlashAttention-2 的论文给过一个直接读数：FlashAttention 1 在 A100 上只跑到理论峰值 FLOPs 的 25–40%，而调好的 GEMM 能到 80–90%。差的不是推导，是推导之外的决定：一个 block 算多少行、一个 warp 算哪一块、搬运和计算怎么重叠、针对哪一代 GPU 编哪条指令、哪些块可以整块跳过。

这一章读三份工业实现，回答同一个问题：同样是分块 online softmax，工程上还要做哪些决定，才能把利用率从三分之一推到四分之三。读法不是逐行背代码，而是先分清论文改的是算法层、执行层还是硬件层，再在源码里找到对应的位置。

**本章目标。** 能说清 FA1、FA2、FA3 各自把瓶颈归到哪里，flash-attn 前向主循环的骨架与特化分支各管什么，CUTLASS 的三级分工与 3.x 的 CuTe 术语，并能按“改一个 tile 参数 → 重编 → 验算 → 计时”的步骤设计一次自己的对照实验。

### 第 1 节 · FlashAttention 1→2→3 的工程演进

FA1 解决的是访存。标准 attention 把 S = QK^T 和 P = softmax(S) 写回 HBM，再读出来乘 V，两个 N×N 矩阵的往返把时间压在显存带宽上。FA1 用 tiling 把 Q、K、V 切成能塞进 SRAM 的块，用 online softmax 在块之间传递运行统计量，S、P 只在片上出现，显存占用从 O(N²) 降到 O(N)。这是算法层的改动，dense attention 上数学精确，没有近似：不稀疏化，也不做低精度量化。

FA1 论文的加速比要连着基线读：它宣称 GPT-2、序列长度 1K 上训练快 3 倍，Long Range Arena（1K–4K）上快 2.4 倍，基线是当时的常规 PyTorch 实现（版本与配置见论文）；FA2 论文回顾 FA1 时给的概括是“相对优化过的基线 2–4 倍”。3 倍、2.4 倍是端到端训练墙钟时间，2–4 倍是注意力算子层面的区间，两者口径不同。引用这些数字时，把对比对象、模型和序列长度一起写出来，否则数字没有意义。

FA1 把访存账算平了，算力利用率仍然低。FA2 的诊断是工作划分：同一个 head 的 attention 只交给一个 thread block（线程块，一组跑在同一个 SM 上的线程），长序列、小 batch 时 SM 大量闲着；block 内部 warp 之间又用 split-K 分工，K、V 切给不同 warp，每个 warp 算出一片部分和，再写 shared memory（同一 block 内线程共享的片上存储）做跨 warp 归约。归约要同步，同步让 Tensor Core 等。

FA2 的三处改动都对着这个诊断：

- **减少非 matmul FLOPs。** 输出累加器不再每块除以归一化因子，而是保持未归一化，块间只乘 `exp(m_old − m_new)`，循环结束再统一除。softmax、rescale 这类逐元素操作只占 FLOPs 的一小部分，但论文指出非 matmul 吞吐最多只有 matmul 的 1/16，省下它们等于把 Tensor Core 的等待时间省下来。
- **跨 thread block 并行。** 前向把 query 的行块分给不同 block，grid 变成 `(query 行块数, batch, heads)`；一个 head 也能占满多个 SM。长序列、小 batch 这个区间受益最大。
- **warp 间改分工。** FA1 是 split-K：K/V 切给 warp，结果要经 shared memory 相加。FA2 改成 split-Q：Q 的行切给 warp，每个 warp 拿着自己那几行 Q 去乘完整的 K，再乘自己那片 V，输出行天然属于自己，warp 之间不需要归约，中间结果留在寄存器里。

causal mask 还有一个块级优化：完全落在对角线右上方的块，整块输出都是 0，可以直接跳过；对角线上的块才需要逐元素 mask。论文报告，长序列下跳过这些块带来约 1.7–1.8 倍于非 causal 版本的加速（论文宣称，序列长度与实现见原文）。

FA2 的结果按口径分开记：摘要说相对 FA1 约 2 倍，A100 上达到理论峰值的 50–73%；端到端训练 GPT 风格模型时，单张 A100 达到 225 TFLOPs/s（72% 模型 FLOPs 利用率）。这是训练数字，不是推理数字。论文第 4 节的基准更细：跨序列长度相对 FA1 快 1.7–3.0 倍，相对 PyTorch 标准实现快 3–10 倍，相对 Triton 版 FA 快 1.3–2.5 倍。摘要取的是约 2 倍这个概括值，具体区间要看基准配置。

FA3 的起点是硬件换代。FA2 的结构是为 Ampere 写的，搬到 H100 上只有 35% 利用率，而 H100 上调好的 GEMM 能到 80–90%。Hopper 多了两样 FA2 没用上的能力：TMA（Tensor Memory Accelerator，张量内存加速器：负责异步搬运的硬件单元）把 global memory 到 shared memory 的拷贝变成一条异步指令；Tensor Core 的 WGMMA 可以异步执行，计算不必等数据全部到位。FA3 的三项技术都是围绕这两点重排流水线：

- **warp specialization（线程束特化）。** 把 block 内的 warp 分成生产者和消费者：生产者只发 TMA 搬运，把 K/V 块填进多级 circular shared memory buffer；消费者只做计算，等 barrier 通知后取数。搬运和计算由不同的 warp 同时推进，不再由一个 warp 轮流做两件事。
- **matmul 与 softmax 交错。** 一个 consumer warpgroup 内部，把上一块的 softmax 与下一块的 WGMMA 排进两级流水；两个 warpgroup 之间再用 ping-pong 调度，一个 warpgroup 做 softmax 时另一个做 GEMM。目的是让 Tensor Core 在 softmax 阶段也有活干。
- **FP8 块量化与 incoherent processing（非相干处理）。** 每个 Q/K/V 块单独定 scale，避免一个全局 scale 被离群值拖垮；再把 Q、K 乘一个随机正交矩阵（实践中用随机对角矩阵与 Hadamard 矩阵的乘积），把离群值摊到多个分量上，量化误差更均匀。矩阵正交，乘积不改变 attention 的输出。

FA3 的结果同样按口径记：H100 SXM5 上，FP16 前向相对 FA2 快 1.5–2.0 倍、反向快 1.5–1.75 倍，最高 740 TFLOPs/s（75% 利用率）；FP8 接近 1.2 PFLOPs/s；FP8 的 RMSE 是 per-tensor 量化基线的 1/2.6。这里有两个容易读错的地方。第一，1.2 PFLOPs/s 是特定配置下的最高值，论文自己说明长序列下 FP16 可能反超 FP8，head 维度 128、256 且带 causal mask 时 FP8 并不总是更快，“FP8 一定更快”不成立。第二，2.6 倍低误差是相对“per-tensor 量化的 FP8 基线”，不是相对 FP16；FP16 下 FA2 与 FA3 的 RMSE 都是 1.9e-4，FP8 版是 9.1e-3，仍比 FP16 差一个量级以上。

| 代 | 诊断的瓶颈 | 改动 | 论文宣称的结果（含口径） |
|---|---|---|---|
| FA1 | S、P 物化后反复过 HBM | tiling + online softmax，不物化 S/P | 显存 O(N²)→O(N)；GPT-2 序列 1K 训练 3×、LRA 1K–4K 2.4×，基线为当时的 PyTorch/HF 实现；FA2 概括为相对优化基线 2–4× |
| FA2 | 工作划分：A100 上 25–40% 峰值 | 未归一化输出累加器、按 query 行切 block、split-Q 取代 split-K、跳过全遮 causal 块 | 相对 FA1 约 2×（摘要），基准区间 1.7–3.0×；A100 50–73% 峰值；GPT 风格训练 225 TFLOPs/s/A100（72% MFU） |
| FA3 | Hopper 新能力没用上：H100 上 35%，GEMM 可到 80–90% | warp specialization + TMA、GEMM-softmax 交错（2/3 级流水）、FP8 块量化 + incoherent processing | 相对 FA2 前向 1.5–2.0×、反向 1.5–1.75×；FP16 740 TFLOPs/s（75%）；FP8 近 1.2 PFLOPs/s；FP8 RMSE 为 per-tensor 基线 1/2.6 |

表里的数字全部来自论文摘要、正文或结论，没有独立复现，行与行之间不能直接比：FA1 的 3× 是端到端训练对 PyTorch，FA2 的 2× 是注意力算子对 FA1，FA3 的 1.5–2.0× 是 H100 上对 FA2。同一代内部，摘要的概括值、正文的区间、端到端与算子级也各不相同。读论文数字先找三样东西：和谁比、在什么模型和序列长度上、量的是端到端还是单个算子。

三代改动一层比一层靠近硬件：FA1 改访存模式（算法层），FA2 改 block 与 warp 的工作划分（执行层），FA3 改流水线与指令的用法（硬件层）。不变的是三件事：不物化 S/P、分块 online softmax、结果精确。判断一篇新实现改的是哪一层，比背下它的加速比更有用。

### 第 2 节 · 读 flash-attn 源码：主循环长什么样

源码读的是 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。FA2 的 CUDA 实现在 `csrc/flash_attn/src/`：前向主 kernel 在 `flash_fwd_kernel.h`，按 head 维、数据类型和架构选实例的 dispatch 在 `flash_fwd_launch_template.h`，每个实例化是一个很小的 `.cu` 文件，例如 `flash_fwd_hdim64_fp16_sm80.cu`。FA3 的实现另放在 `hopper/` 目录，主循环文件是 `mainloop_fwd_sm90_tma_gmma_ws.hpp`，文件名里的 `tma_gmma_ws` 就是它的三项技术。文件名和目录结构随版本变，读之前以当前仓库为准。

读的顺序与执行顺序相反：先看 dispatch，再看 kernel 本体。dispatch 决定“给定 head 维、精度、架构、causal，实际跑哪个变体”，不先看它，钻进任何一个 `.cu` 都不知道自己读的是哪种输入。`flash_fwd_launch_template.h` 里，`run_mha_fwd_hdim128` 按架构和 causal 选 tile：sm86/sm89 上非 causal 用 128×32、causal 用 64×64，其他架构用 128×64；dropout 打开又换一套。这些选择不是数学分支，是 tile 形状与 shared memory 预算的权衡。

dispatch 之后看 grid。`run_flash_fwd` 里 `dim3 grid(num_m_block, params.b, params.h)`，第一维是 query 行块数——单个 head 的 query 行被拆到多个 block，这就是 FA2 的跨 block 并行。再看 `Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, ...>`：head 维、M tile、N tile、warp 数在编译期定死，`kBlockM`、`kBlockN` 决定 shared memory 大小和每块的计算量。

主循环的骨架可以压成下面这段示意。它保留了 `flash_fwd_kernel.h` 的结构，删掉了模板、边界判断和 dropouts 等分支，不是可直接编译的代码：

```cpp
// 示意：按 flash_fwd_kernel.h 的主循环简化，省去模板参数与边界处理
int n_block_max = ceil_div(seqlen_k, kBlockN);
if (Is_causal) {
    n_block_max = min(n_block_max,
        ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
}
for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    cp_async_wait<0>(); __syncthreads();
    copy(gV[n_block], sV);           // 当前块的 V 搬进 shared memory
    cp_async_fence();

    gemm(acc_s, Q, K);               // S = Q K^T

    cp_async_wait<0>(); __syncthreads();
    if (n_block > n_block_min) {     // 预取下一块的 K
        copy(gK[n_block - 1], sK);
        cp_async_fence();
    }

    mask.apply(acc_s, n_block, m_block);   // causal/边界 mask
    softmax.softmax_rescale_o(acc_s, acc_o, scale_log2);  // m/l/O 更新
    gemm_rs(acc_o, P, V);            // O += P V
}
```

把这段和第 21 章的推导对齐，只需要认三个量。`acc_s` 是当前块的 S，对应推导里的 `Q_block · K_block^T`；`acc_o` 是输出累加器，对应 `Õ`；`softmax_rescale_o` 内部的 `row_max`、`row_sum` 对应 `m`、`l`。`softmax.h` 里的 rescale 逻辑是：先存旧的 `row_max`，取新最大值，算 `scale = exp2((m_old − m_new) · softmax_scale_log2)`，把 `row_sum` 和 `acc_o` 都乘上 `scale`，再对当前块做 `exp2` 并累加。这与第 21 章的更新规则逐项对应，只是实现把乘法顺序和 `exp2` 的缩放常数合并了。

循环从 `n_block_max − 1` 倒着走到 `n_block_min`，不是正着走。causal 下 `n_block_max` 由当前 query 行块决定，倒序走可以先处理对角线附近需要 mask 的块，再进入整块无需 mask 的区域；完全在对角线上方的块根本不进循环。实际代码把循环拆成两段：前 `n_masking_steps` 次带 mask，后面所有块不带 mask，避免在主体循环里反复判断。这是按块跳过 causal 计算在代码里的落点。

流水线体现在两处。Q 在循环外只搬一次，之后常驻 shared memory 或寄存器；K、V 每块轮换，当前块的 V 在 `QK^T` 之前搬入，下一块的 K 在当前块计算期间预取，`cp_async_fence` 与 `cp_async_wait<0>` 负责“搬完再算、算完再覆盖”的次序。`kernel_traits.h` 里 `kSmemKVSize = size(SmemLayoutKV) * 2 * sizeof(Element)`，K、V 各留两份，双缓冲让预取和计算重叠。

warp 的分工看 `TiledMma`：`Layout<Shape<Int<kNWarps>, _1, _1>>` 把 warp 排到 M 维，`Tile<Int<16 * kNWarps>, _16, _16>` 让每个 warp 负责一段 query 行。head 维 128、`kBlockM = 128`、`kNWarps = 4` 时，每个 warp 拿 32 行，行内的 softmax 归约在 warp 内完成，warp 之间不通信。这正是 FA2 用 split-Q 取代 split-K 的代码形态：FA1 的 split-K 需要把每个 warp 的部分和写回 shared memory 再相加，FA2 不需要。

读工业代码的边界：特化分支按 head 维（32/64/96/128/192/256）、架构（sm80/sm90）、causal、local、alibi、softcap、dropout、split-KV、paged KV 展开，组合数很大。读的目标是定位结构，不是读完每个分支。遇到看不懂的模板或 `#ifdef`，先记它管哪个维度，再回到主循环。Triton 版没有这些分支，不等于特化不存在，是编译器把一部分特化藏了起来；第 20 章的 kernel 与这里的差距，主要就在这些显式决定上。

合上代码，能回答四个问题，主循环就算读通了：Q、K、V 各在哪里加载，Q 加载几次、K/V 各几次；`m`、`l`、`acc_o` 在循环的第几步更新，rescale 乘在哪些量上；循环走完一块后，哪些量留到下一块、哪些写回 global；FA2 的三处改动至少一处在代码里的落点。答不出来的位置，就是还没读成结构的地方。

### 第 3 节 · CUTLASS 三级结构：threadblock / warp / thread + MMA pipeline

CUTLASS 的最小例子和 3.x 的抽象不在同一层 API 上，先把两层分开。`examples/00_basic_gemm/basic_gemm.cu` 用 `cutlass::gemm::device::Gemm<float, ColumnMajor, ...>`，默认 threadblock tile 是 128×128×8；`default_gemm_configuration.h` 里对应的默认值是 `ThreadblockShape = 128×128×8`、`WarpShape = 32×64×8`、`InstructionShape = 1×1×1`、`Stages = 2`。`InstructionShape` 是 1×1×1，说明这个 float 实例走的是 SIMT 的标量乘加，不是 Tensor Core；要看 MMA 指令的形状，得换成 half/bf16 的 tensor-op 实例，那是 16×8×8 一类的形状。这一层区分很重要：后面的算术账都成立，但“最终落到 Tensor Core”这句话只对 tensor-op 实例成立。

先把默认 tile 的账算清楚。128×128×8 表示一个 block 算输出矩阵 128 行 × 128 列的一块，K 维每步推进 8。一个 K 步搬入 A 的 128×8 和 B 的 8×128，各 1024 个 float，合计 2048 × 4 = 8192 字节 = 8 KiB；计算量是 2 × 128 × 128 × 8 = 262144 FLOP；算术强度 262144 ÷ 8192 = 32 FLOP/字节。`Stages = 2` 时 shared memory 占 2 × 8 = 16 KiB。4096×4096 的输出按 128×128 切，是 32 × 32 = 1024 个 block；K 维 4096 按 8 切，是 512 步。WarpShape 32×64 把 block 分成 (128/32) × (128/64) = 4 × 2 = 8 个 warp，共 256 个线程。

三级分工与 3.x 的对应关系：

| 层级 | 2.x 模板参数（float 例子） | 3.x 的写法 | 管什么 |
|---|---|---|---|
| threadblock | `ThreadblockShape = 128×128×8` | `TileShape`、`ClusterShape` | 一个 CTA 的输出 tile、shared memory 缓冲、TMA 或 cp.async 搬运 |
| warp | `WarpShape = 32×64×8` | `TiledMMA` 的线程布局，由 MMA atom 推导 | block 内每个 warp 算哪一块、寄存器里放什么 |
| thread | `InstructionShape = 1×1×1` | `MMA_Atom` 与 `TiledCopy` 的分片 | 每条指令、每次访存处理多少元素 |

数据流向是 global memory → shared memory（block 级搬运）→ registers（warp/thread 级分片）→ MMA 或 FFMA → 写回 global。每一级回答“这块数据现在在哪、谁来算”。三级不是 CUTLASS 的发明，是硬件层级的映射：block 对应 SM 与 shared memory，warp 对应 warp 调度器与寄存器，thread 对应最细的指令分片。分级把一次大搬运拆成三级小搬运，每级只需喂饱对应硬件的带宽。

CUTLASS 3.x 换了词汇，读代码先认五个词（名称随版本变化，以当前版本文档为准）：

- **CuTe**：CUTLASS 3.x 的布局代数层，用 `Layout`、`Tensor`、`local_tile`、`partition` 描述“哪块数据在哪个线程的哪个寄存器里”。`cute::gemm(tiled_mma, A, B, C)` 是这一层的通用 GEMM 调用。
- **MMA**：矩阵乘加 atom。3.x 用 `TiledMMA` 把一个 `MMA_Atom` 平铺到 warp/thread；Hopper 上是 WGMMA 系列，atom 名称形如 `SM90_*`。
- **TMA**：Hopper 的异步张量搬运。用 `make_tma_copy` 构造拷贝 atom，运行时由 `SM90_TMA_LOAD` 发起；一个 CTA 可以一次搬一整块 tile，不需要每个线程算地址。
- **cluster**：thread block cluster，多个 CTA 组成一个簇，可以互相访问对方的 shared memory（DSMEM），TMA 还能把同一块数据 multicast 给簇内多个 CTA。
- **pipeline**：生产者/消费者流水线。`cutlass::PipelineTmaAsync<Stages>` 管理多级 shared memory buffer 的 barrier，生产者 `producer_acquire/commit`，消费者 `consumer_wait/release`。

3.x 的入口是一个 collective builder，下面这段是示意，省略了类型别名与布局细节，参数顺序以当前版本的 `CollectiveBuilder` 文档为准：

```cpp
// 示意：Hopper 上的 3.x collective GEMM，不保证按当前版本原样编译
using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

using Collective = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;
```

`TileShape` 的三个数分别是 block 级 M、N、K tile；`ClusterShape` 决定 cluster 里几个 CTA、沿哪个维度排；`StageCountAuto` 让 builder 按 shared memory 预算选流水级数；`KernelScheduleAuto` 选调度策略，在 Hopper 上通常落到 TMA + warp-specialized 的 mainloop。warp 与 instruction 级的形状不再由用户直接给，而是由 `OpClassTensorOp`、元素类型和 atom 选择推导出来。这是 3.x 与 2.x 最大的结构差异：2.x 让人手填三个 Shape，3.x 让人填一个 tile 和一个 cluster，其余交给 builder。

流水线的账可以直接按 tile 算。half 的 128×128×64 tile，一个 stage 的 A 是 128×64×2 = 16 KiB，B 是 64×128×2 = 16 KiB，合计 32 KiB；3 级流水占 96 KiB，4 级占 128 KiB。Hopper 每个 SM 有 228 KiB shared memory，级数越多，预取越深，能同时驻留的 CTA 越少。这条取舍与第 19 章 occupancy 的账是同一个：shared memory 占用上升，驻留的 warp 数下降，延迟藏不住时，深流水反而变慢。

改 tile 的入口在两层 API 上不同。2.x 的 `cutlass::gemm::device::Gemm` 把 `ThreadblockShape`、`WarpShape`、`InstructionShape`、`Stages` 列为模板参数，默认值在 `default_gemm_configuration.h`；3.x 改的是 `TileShape`、`ClusterShape`、`StageCount`，warp/instruction 形状跟着 atom 走。下一节的实验按这两条路径分别设计。

### 第 4 节 · 动手改一个 CUTLASS GEMM：改 tile 配置，重编，测出变化

这一节是一次待补测的实验设计，不是实测报告。本章没有在 GPU 上跑过这些配置，下面的耗时列全部留空；能先确定的是改动步骤、按定义算出的预期方向和失败模式。

先立基线。按仓库 README 用 CMake 编出 `00_basic_gemm`，跑默认 128×128×8，先用 naive 参考 GEMM 验算结果，再计时。当前版本的 `00_basic_gemm` 只打印 `Passed.`，没有自带计时，需要自己在 `gemm_operator(args)` 前后插 CUDA Events，或者用 nsys 抓 kernel 时间。计时口径要固定：预热若干次，正式跑多次取中位数，只计 kernel 时间，不把编译和进程启动算进去。单次运行的第一个数通常偏冷，时钟没拉起来，抖动可能比配置带来的差异还大。

三组 2.x 配置的预期账如下。每 K 步搬运按 A tile + B tile 算，shared memory 按 `Stages = 2` 算，输出仍是 4096×4096：

| ThreadblockShape | 每 K 步搬运 | 算术强度 | shared memory（2 级） | block 数 | 预期方向 | 实测 |
|---|---:|---:|---:|---:|---|---|
| 64×128×8 | 6 KiB | 21.3 FLOP/字节 | 12 KiB | 2048 | 并行度高、单块复用低，适合 SM 填不满的形状 | 待补测 |
| 128×128×8 | 8 KiB | 32 FLOP/字节 | 16 KiB | 1024 | 基线 | 待补测 |
| 256×128×8 | 12 KiB | 42.7 FLOP/字节 | 24 KiB | 512 | 单块复用高、并行度低，可能受寄存器与驻留数限制 | 待补测 |

改法只动一个参数。2.x 的 `Gemm` 可以在模板参数里显式给三个 Shape 与 Stages，下面这段是示意，参数顺序以当前版本 `gemm.h` 为准：

```cpp
// 示意：把默认的 128×128×8 换成 64×128×8，其余保持默认
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape
    cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape
    2>;                                     // Stages
```

改完重编，先验算再计时。结果不对，耗时没有意义；结果对，才把耗时填进表里。64×128 与 256×128 各改一次，每次只改一个 Shape，WarpShape 和 Stages 保持默认，否则时间变了归因不清。

3.x 路径改的是另一组参数，预期方向也不同：

| 改动 | 预期方向 | 实测 |
|---|---|---|
| `TileShape` 的 K 从 64 降到 32 | mainloop 步数翻倍，每级 shared memory 减半，流水开销占比上升 | 待补测 |
| `TileShape` 的 N 从 128 升到 256 | B tile 复用提高，寄存器与 shared memory 占用上升，CTA 数减少 | 待补测 |
| `ClusterShape` 从 1×1×1 改为 2×1×1 | TMA multicast 让两个 CTA 共享一次 B 搬运，HBM 流量下降；网格必须能被 cluster 整除 | 待补测 |
| `StageCount` 从 3 改为 4 | 预取更深，抗延迟更好；shared memory 多一级，驻留 CTA 数可能下降 | 待补测 |

3.x 只改 `TileShape` 或 `ClusterShape` 一行，重编后同样先验算再计时。cluster 大于 1 时，还要确认 grid 的维度能被 cluster 形状整除，否则 launch 会失败。

失败模式按出现顺序记：

- **编译失败。** shared memory 超过每个 block 的上限、TMA 对齐不满足（如 leading dimension 不是 16 字节对齐）、cluster 形状与 grid 不匹配、模板约束不满足，都会在编译期报错。把配置和报错原文一起记下来，这条记录本身就是结论：这条配置在当前硬件上走不通。
- **结果错误。** 编过了但结果不对，先查 tile 与 layout 的对齐、边界处理、cluster 下的数据分区，再查计时。结果错误时的耗时数字一律作废。
- **测速假象。** 把编译时间或首次运行的冷启动算进去、只跑一次、不同配置跑在不同时钟状态上，都会让比较失真。固定预热次数、重复次数和计时口径，再谈差异。

预期与实测对不上时，先查测量，再查分析。tile 变大反而变慢，常见原因是 block 数掉到填不满 SM、shared memory 占用上升导致驻留 CTA 减少、寄存器溢出；tile 变小反而变慢，常见原因是每块复用下降、流水启动开销占比上升。Roofline 和第 19 章的分析给方向，平衡点在实测里。

### 第 5 节 · 小结：Triton 版 vs CUTLASS 版 attention 的性能与可维护性

把第 20、21 章的 Triton 经验与本章的工业实现放在同一张表上，按四个维度比。表里的性能数字来自论文，不是本章实测；可维护性、可移植性、调试成本是工程判断，依据写在后面的段落里。

| 维度 | Triton 版（第 20、21 章） | CUTLASS / flash-attn 版 | 判断依据 |
|---|---|---|---|
| 可维护性 | 一个文件几十到几百行，改 tile 就是改一个 `BLOCK_M`，autotune 扫参数 | 模板层层嵌套，dispatch 按 head 维/架构/causal 展开，改一处要确认影响哪个实例 | Triton 的抽象层级高，代码短；CUTLASS 把控制权交给用户，代码长 |
| 可移植性 | 编译到 PTX，同一份代码可跑 NVIDIA 与部分 AMD 后端；换架构主要靠编译器 | 绑定 CUDA 与具体架构，SM80、SM90 等不同架构世代的 atom 与流水线不同；换芯片要重写 | Triton 的移植成本低，但性能依赖后端成熟度；CUTLASS 的移植是重写 |
| 性能上限 | 能表达分块、cp.async、软件流水；TMA、warp specialization、cluster 的支持随版本推进，能写到什么程度以当前版本文档为准 | 能直接用手写 warp 分工、TMA、cluster、WGMMA 与 FP8 atom | FA3 从 35% 到 75% 的差距，靠的正是 Triton 当前表达不了的那部分控制 |
| 调试成本 | 报错在 Python/Triton 层，可用 `tl.device_print`、解释器模式、autotune 日志 | 编译期模板报错长，编译慢，运行时要用 ncu、SASS 和 barrier 级日志 | Triton 定位快；CUTLASS 定位慢，但一旦跑对，行为更可预测 |

四个维度不能压成一个“谁更快”。Triton 的代码量优势来自抽象：分块、流水线、内存层级由编译器管，代价是编译器不暴露的控制它也写不出来。CUTLASS 把这些控制暴露出来，代码量和调试成本随之上升。性能上限的差距有明确证据：FA2 在 H100 上 35%、FA3 到 75%，中间靠的是 warp specialization、TMA、GEMM-softmax 交错与 FP8 布局这些在写作时仍主要靠 CUDA/CUTLASS 表达的手段；Triton 对这些能力的支持在版本间变化，以当前文档为准。可移植性的方向相反：Triton 一份代码能编译到不同后端，CUTLASS 的每个架构 atom 都要单独适配；第四部分换到 Ascend 时，两条路都要重做一遍，Triton 侧至少有 TritonAscend 这类后端承接，CUTLASS 侧则是重写。

选择按瓶颈所在层级定，不按“库是不是更快”定：

- 优化能说清“哪个 block 读哪块、哪个算子融进去”时，Triton 够用。第 21 章的 fused attention 就属于这一层：省掉 S、P 过 HBM 的往返，几十行能写出来，也能验证。
- 优化必须说清“哪个 warp 搬数据、哪个 warp 算、流水第几级、cluster 怎么分”时，回 CUDA/CUTLASS。FA3 的三项技术都在这一层，Triton 没有对应表达。
- 库已经比自写实现快，且没有特殊形状要适配时，直接用库。自己维护一份 FP8 attention 的成本是数周起步，还要跟硬件代际更新；库已经把这条路径封好。
- 还有时间成本：Triton 改一版很快，适合验证想法；CUTLASS 改一个 tile 就要重编，适合想法已经验证、只剩最后一段性能要榨。第 24 章把自写 kernel 接进引擎，选的是 Triton 路径，原因也在这里：接入、验证、回滚的成本低。

本章没有给出 Triton 版与 CUTLASS 版在同一张卡、同一序列长度上的实测对比。要做这个对比，需要固定模型配置、head 维、序列长度、精度和计时口径，分别跑 Triton fused attention、flash-attn 与一个 CUTLASS GEMM 基线；在那之前，本章的性能结论只到“论文宣称的区间”和“控制手段的有无”，不到“我的机器上快多少”。

**延伸**：[FlashAttention（Dao 等，2022）](https://arxiv.org/abs/2205.14135)、[FlashAttention-2（Dao，2023）](https://arxiv.org/abs/2307.08691)、[FlashAttention-3（Shah 等，2024）](https://arxiv.org/abs/2407.08608)、[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)、[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)。

## 第 24 章 · 收口：把自写 kernel 接入推理引擎

> 边界：vLLM 的 attention 走 attention 后端，不走 CustomOp——后端选择、KV 布局、显存分配都在那一层，一个 forward 函数装不下。CustomOp 的可替换面是激活、归一化这类单 forward 的逐元素算子，所以本章接入引擎的载体是 Triton 版 SiluAndMul；fused attention 留在第 21 章做 standalone 验证。

一个在 standalone 里比 PyTorch 快 2 倍的 kernel，接进 vLLM 之后端到端数字没变，服务也不报错，这是这条路上最常见的结果。难解释的不是“没变快”，而是不知道它到底跑没跑。中间隔着三道门：引擎启动时插件有没有被 import，模型代码拿到的类有没有被换成你的类，这一层在当前编译配置下走的是 CUDA 实现还是 PyTorch 后备。三道门任何一道没开，模型照常出 token，日志照常干净，只有 profiler 的 kernel 名单能证明你写的那段代码真的在 GPU 上执行过。

**本章目标。** 把 Triton 版 SiluAndMul 包成 vLLM 的 CustomOp 插件，用同一模型、同一负载测出接入前后的 TTFT、ITL 与吞吐，并用 profiler 证明 Triton kernel 真的被调用；端到端没变快时，说清是算子份额太小、引擎基线本来就是融合实现，还是替换根本没生效。

### 第 1 节 · vLLM 的 CustomOp 机制：注册、分发、启用三件事，及它的边界

注册、分发、启用是三件不同的事，发生在三个不同的时间点：注册在 import 时，启用在引擎启动读配置时，分发在每次前向时——准确说，分发方法在层对象构造时就被定下来，前向只是调用它。把三件混成一件，后面每一节的坑都在这里埋下。本节的事实都以写书时的 vLLM 主线为准（`vllm/model_executor/custom_op.py` 与 [CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)），**API 名称与签名以当前版本为准**：这套接口改过几轮，抄博客之前先打开自己装的那份源码对一遍。

**注册**说的是名字怎么进表。`vllm/model_executor/custom_op.py` 里有两个文件级字典：`op_registry` 装树内算子（由 `CustomOp.register("名字")` 写入），`op_registry_oot` 装树外替换（由 `register_oot` 写入）。写插件用的是 `@SiluAndMul.register_oot`，不带参数时注册名就是被替换类的类名 `SiluAndMul`。这个名字不是随手取的：`CustomOp.__new__` 拿 `cls.__name__` 去 `op_registry_oot` 里查，命中才把实例换成你的类。所以注册名必须等于要替换的类的类名，写成 `silu_and_mul` 就查不到，替换静默失效。同一个名字在 `op_registry_oot` 里注册两次会断言失败（`Duplicate op name`）——这条防线在自测里很有用，见第 3 节。

**启用**说的是开关怎么打开。编译配置里用 `--compilation_config.custom_ops '["+名字"]'` 打开，`+` 是“在默认集合之外再加这一个”，不是替换整个集合；`-名字` 是关掉。名字取的是层实例的 `name` 属性，不是你想当然的那个字符串：树内的 `SiluAndMul` 注册名是小写 `silu_and_mul`，被 `register_oot` 换掉之后，注册装饰器会把子类的 `name` 改写成注册名（类名），两者的键不一样。**以当前版本为准**：在引擎进程里打印 `SiluAndMul().name`，拿打印出来的值去配 `+名字`，不要凭记忆填。

默认值取决于编译后端：`CustomOp.default_on()` 读 `compilation_config.custom_ops` 里的 `all` / `none` 基底，用 inductor 时默认是 `none`（关闭），此时 custom op 得显式 `+` 才进图。为什么 inductor 默认要关？inductor 要把计算图抓下来融合重写，custom op 对它是黑盒子，看不懂里面是什么。关掉不等于不跑：`dispatch_forward` 在“未启用”分支返回的是 `forward_native`，语义正确、数值一样，只是慢一点——这正是静默失败的根。

**分发**说的是运行时怎么找到你的实现。`CustomOp.__new__` 负责替换，`__init__` 里的 `dispatch_forward` 负责选方法，选完存进 `_forward_method`，之后每次 `forward()` 都调它。选择顺序按平台：ROCM 走 `forward_hip`，CPU 走 `forward_cpu`，TPU 走 `forward_tpu`，XPU 走 `forward_xpu`，**自报 OOT 的平台**走 `forward_oot`，其余（含 NVIDIA CUDA）走 `forward_cuda`。`forward_oot` 这个名字容易误解：它只在平台把自己标记为 out-of-tree 时被选中，在 CUDA 上写了也不会被调用——那一层留给第 27 章的 vllm-ascend。在 CUDA 上替换一个算子，重写的方法是 `forward_cuda`。

| 环节 | 发生时间 | 失效时的表现 | 怎么查 |
|---|---|---|---|
| 注册 | 插件包被 import 时 | 表里没有你的类，什么都不发生 | `op_registry_oot["SiluAndMul"]` 是不是你的类 |
| 启用 | 引擎启动、层对象构造时 | 走 `forward_native`，数值对、速度慢 | 打印 `层实例.name` 与 `_forward_method.__name__` |
| 分发 | 每次前向 | 方法选错（如只写了 `forward_oot`），抛错或走父类 | profiler 的 kernel 名单里有没有你的 kernel |

**边界**划在“单 forward、无状态”上。激活算子输入进来、输出出去，没有跨层状态，没有 KV 布局，没有后端要选；attention 这三样都有，所以它有自己的后端机制和 KV 集成，CustomOp 接不上。选 SiluAndMul 做载体正是因为它在边界内：形状规整，语义只有一行，容易验证到底对不对。

**插件怎么被加载**：vLLM 启动时按 `vllm.general_plugins` 这个 entry point 发现插件包，`load_general_plugins` 在进程内只加载一次（有防重复加载的 guard）。关键时序是：插件包被 import 时装饰器执行，替换进表；没被 import 就什么都不会发生。一个“写好了但没被加载”的插件，和一个不存在的插件，从模型输出上看没有任何区别。

和 monkey patch 比，`register_oot` 走的是引擎自己的表：不改 `vllm.model_executor.layers.activation` 文件里的类属性，升级 vLLM 后不会因为函数签名变了而错得莫名其妙；它还保留了“按开关退回 `forward_native`”这条后备路径。monkey patch 更直接，对所有实例生效，也能改表里没有的算子，代价是绕过了引擎的启用与平台判断，出问题时没有任何一层会替你把关。和 `CustomOp.register` 比，区别是适用场景：`register` 是在树内**新增**一个算子名字，模型代码得调用它才有用；`register_oot` 是**替换**一个已有类名，调用点已经在模型代码里了，你只是换了实现。

两个容易混的判断：“注册了就等于启用了”——错，注册进表、启动时打开开关，缺一个都走后备；“CustomOp 能包一切算子”——错，边界就是单 forward、无状态。本节拿到的是一张地图：三件事各在何时发生，边界在哪，静默失败长什么样。后三节沿着地图走一遍：写 kernel、包成插件、端到端验证。

### 第 2 节 · 动手：写 Triton 版 SiluAndMul，standalone 正确性 + 微基准

先钉死语义。vLLM 树内的 `SiluAndMul`（`vllm/model_executor/layers/activation.py`）在文档字符串里写得很清楚：`x -> silu(x[:d]) * x[d:]`，其中 `d = x.shape[-1] // 2`。落到我要写的 kernel 上，约束有四条：

- **形状**：输入 `(num_tokens, 2d)`，最后一维必须是偶数；输出 `(num_tokens, d)`。vLLM 的 CUDA 实现按 2D 传参，`forward_native` 支持任意前置维；按 2D 写 kernel、进来先 `reshape(-1, 2d)` 最省事。
- **语义**：前一半逐元素过 SiLU 再乘后一半；顺序不能反，`mul_and_silu` 是另一个算子（`x[:d] * silu(x[d:])`），两者不等价。
- **dtype**：vLLM 实际跑的是 fp16 / bf16。SiLU 在 fp32 里算完再 cast 回输入精度，再和另一半相乘——cast 的位置是 load-bearing 的。
- **连续性**：kernel 按行跨步寻址，要求最后一维连续、行跨步等于 `2d`；不满足就先 `contiguous()`，或者在 kernel 里把行跨步当参数传进去。vLLM 树内的 Triton 激活 kernel（`_swiglustep_and_mul_kernel`）就是把行跨步当参数传的。

融合省的是遍数，不是计算量。拿 fp16（每元素 2 字节）算一遍：融合版读输入 `[M, 2D]` 共 `4MD` 字节，写输出 `[M, D]` 共 `2MD` 字节，合计 `6MD`；eager 版是两个 kernel，silu 读 `2MD` 写 `2MD`，乘法读中间结果 `2MD`、读后一半 `2MD`、写 `2MD`，合计 `10MD`。省 `4MD`，占 eager 版访存的 40%。代入 `M = 128`、`D = 128`：输入 64 KiB、输出 32 KiB，融合版 96 KiB，eager 版 160 KiB，省 64 KiB。exp 和 sigmoid 的 FLOPs 一个没少，省掉的是中间结果写回 HBM 再读回来的一次往返，和第 21 章 fused attention 省掉 S、P 落 HBM 是同一笔账。

这笔 40% 只对 eager 成立。vLLM 引擎里跑的 `SiluAndMul.forward_cuda` 调的是 `torch.ops._C.silu_and_mul`——一个已经在 C++ 里融合好的 kernel，它的访存本来就是 `6MD`。所以引擎基线和新 kernel 之间能争的不是遍数，是常数项：向量化宽度、grid 形状、启动开销、边界处理。这个区别决定了第 4 节的预期，也决定了微基准必须测三个基线，而不是两个。

kernel 如下（示意，按当前 Triton 版本调整）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _silu_and_mul_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, d,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    x1 = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x2 = tl.load(x_ptr + row * x_row_stride + d + cols, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    s = (x1_f32 * tl.sigmoid(x1_f32)).to(x1.dtype)   # 先算 fp32，再 cast 回输入精度
    tl.store(y_ptr + row * y_row_stride + cols, s * x2, mask=mask)

def silu_and_mul_triton(x: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    assert x.shape[-1] % 2 == 0 and x.is_contiguous()
    m, n = x.shape
    d = n // 2
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    grid = (m, triton.cdiv(d, BLOCK))
    _silu_and_mul_kernel[grid](x, y, x.stride(0), y.stride(0), d, BLOCK=BLOCK)
    return y
```

逐行看地址算术，这是最容易写错的地方。`x1` 在 `row * 2d + col`，`x2` 在 `row * 2d + d + col`，输出在 `row * d + col`；行跨度是整行 `2d`，写成 `row * d` 读到的就是错行的数。`mask = col < d` 配 `other=0.0`：被 mask 的位置读进来是 0，`silu(0) = 0`，store 时又被同一个 mask 拦住，既不越界读也不越界写。grid 是二维的（行 × 列块），每个程序负责一行里的一块列。

拿正文的算例看 grid 的数：`M = 128`、`D = 128`、`BLOCK = 1024`，列块数 `cdiv(128, 1024) = 1`，grid 是 `(128, 1)`，128 个程序，每个程序 1024 条 lane 里只有前 128 条干活，其余 87.5% 被 mask 掉。这是正确性算例，不是性能算例。真实模型的 `d` 是 11008、13824 这种量级，`BLOCK` 该取多少要跟着 `d` 扫出来，答案在微基准里，不在纸上。

正确性检查拿树内实现当标准，别拿 `F.silu` 手拼一个参考——标准要和应用里真正跑的那份语义一致：

```python
from vllm.model_executor.layers.activation import SiluAndMul

torch.manual_seed(0)
x = torch.randn(128, 256, dtype=torch.float16, device="cuda")
y = silu_and_mul_triton(x)
ref = SiluAndMul.forward_native(x)          # 当前版本里它是 staticmethod，可直接调
print("max abs err:", (y.float() - ref.float()).abs().max().item())
print("allclose:", torch.allclose(y, ref, atol=1e-2, rtol=1e-2))
```

误差不该是 0：求值顺序不同、fp16 舍入路径不同，全等本来就不该出现。要看的是量级：和 fp16 舍入同量级就是对齐了，和输出值本身同量级就是写法错了。三个典型错误各有特征：两半切反，误差立刻变大；行跨度写成 `row * d`，读错行，误差立刻变大；cast 顺序颠倒（拿 fp32 的 SiLU 结果直接乘 fp16 的另一半），误差变大但不一定一眼看得出来——所以逐行核对地址算术这一步不能省。

微基准要测三个基线，缺一个结论就会错：

```python
def eager(t):                      # 引擎的 PyTorch 后备
    return SiluAndMul.forward_native(t)

def cpp_fused(t):                  # 引擎默认实现；签名以当前版本的 csrc 注册为准（这一版是 out 在前）
    out = torch.empty(t.shape[:-1] + (t.shape[-1] // 2,), dtype=t.dtype, device=t.device)
    torch.ops._C.silu_and_mul(out, t)
    return out

x = torch.randn(4096, 2 * 11008, dtype=torch.float16, device="cuda")
for name, fn in (("eager", eager), ("vllm-cpp", cpp_fused), ("triton", silu_and_mul_triton)):
    ms = triton.testing.do_bench(lambda: fn(x))   # 默认返回中位数，以当前 Triton 版本为准
    print(name, f"{ms:.4f} ms")
```

读法：eager 和另外两个之间应该看到接近 40% 的访存差（形状够大时）；vllm-cpp 和 triton 之间的差别，才是这次接入能兑现的部分。三行都包含输出分配的时间，口径一致。如果 triton 比 vllm-cpp 慢，先看 `BLOCK` 与 `d` 的匹配（`d = 11008` 时 `BLOCK = 1024` 要 11 个列块，最后一块只有 768 条有效 lane），再看是不是形状太小、启动开销盖过了收益。这些数字要测出来再写，不要先下结论。

### 第 3 节 · 包成 CustomOp 子类并注册：`register_oot` 按类名替换，进程内验证替换真的发生了

插件包三部分：子类、注册、入口。子类是实现，注册是把类名写进 `op_registry_oot`，入口是让 vLLM 启动时 import 到你的包。缺一个，链条就断。

```python
# my_plugin/silu_and_mul_plugin.py
import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul

from .kernel import silu_and_mul_triton

@SiluAndMul.register_oot          # 不带参数 → 注册名 = 被替换类的类名 "SiluAndMul"
class TritonSiluAndMul(SiluAndMul):
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul_triton(x)
    # forward_native 从父类继承：CPU / 未启用时走它，语义与数值都和原来一致
```

三个细节决定这段代码生不生效。第一，装饰器写在**被替换的类**上（`@SiluAndMul.register_oot`），不是写在 `CustomOp` 上、也不是写在自己的类上；写成 `@CustomOp.register_oot` 会把名字注册成 `CustomOp`，谁也查不到。第二，重写的方法是 `forward_cuda`——CUDA 平台上 `dispatch_forward` 选的就是它；只写 `forward_oot` 在 CUDA 上永远不会被选中，服务照常跑。第三，子类要继承 `SiluAndMul` 而不是 `CustomOp`：父类的 `__init__` 会处理 `compile_native` 参数，模型代码构造它时传的参数得接得住。

打包靠 entry point。`pyproject.toml` 里登记一个 `vllm.general_plugins` 入口，值指向一个可调用对象，vLLM 加载到这个入口时会调用它：

```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_plugin:register"
```

```python
# my_plugin/__init__.py
def register() -> None:
    from . import silu_and_mul_plugin  # noqa: F401  import 时装饰器执行，替换即完成
```

入口点的值是可调用对象还是包路径、函数签名要不要参数，**以当前版本的 `vllm/plugins/__init__.py` 为准**。写完 `pip install -e .` 装上，再看 vLLM 启动日志里有没有出现你的插件；entry point 写错，插件根本不会被加载，vLLM 不会去扫描你的代码。

验证分两层，先便宜后贵。第一层在裸 Python 进程里查表，不需要 GPU、不需要起引擎：

```python
import my_plugin                                   # 触发 import，装饰器执行
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.activation import SiluAndMul

assert op_registry_oot.get("SiluAndMul") is my_plugin.silu_and_mul_plugin.TritonSiluAndMul
print(SiluAndMul().name)                           # 启用开关要用的名字，别凭记忆填
```

第二层在引擎进程里查分发。构造 `SiluAndMul()` 时 `__new__` 会按类名替换，`__init__` 里的 `dispatch_forward` 会定下 `_forward_method`；两个都能直接读：

```python
op = SiluAndMul()
assert type(op) is TritonSiluAndMul, type(op)
print(op._forward_method.__name__)   # 期望 "forward_cuda"；若是 native 或编译包装，说明没启用
```

第二行是本节的关键检查，它把“注册了”和“启用了”分开：类型对但方法名是 `forward_native`，说明开关没打开，替换白做。构造对象会读全局编译配置（`dispatch_forward` 里调 `get_cached_compilation_config`），裸进程里最稳的做法是在引擎启动之后再跑这段，或者按当前版本先初始化好配置——这一条以当前版本为准。

第三层证据在 profiler 里，留到第 4 节。三层问的是同一个问题的三个侧面：表里有没有、分发到没到、GPU 上跑没跑。

避坑：同一个名字注册两次会断言失败。`load_general_plugins` 的 guard 防的是 vLLM 自己重复加载，防不住你在代码里手动再调一次 `register_oot`；REPL 里 import 两次插件包、又手动注册一次，报错看起来像插件写坏了，其实是注册了两遍。读报错要读全：`Duplicate op name` 是注册了两次，`ImportError` 才是插件没进来。

### 第 4 节 · 端到端验证：同一模型、同一评测集，接入前后的 TTFT / ITL / 吞吐 + profiler 确认 Triton kernel 真的跑了

端到端对比只有一条规矩：两次测量之间只改“是否启用这个替换”，其他条件全部钉死。钉不住的条件里，任何一个动了，表里的数就没法归因。

| 控制项 | 怎么钉 | 动了会怎样 |
|---|---|---|
| 模型与权重 | 同一个 checkpoint、同一份量化配置 | 权重变了，算子占比和总时间一起变 |
| 引擎与 CUDA / Triton 版本 | 同一个镜像、同一条启动命令 | 版本升级带来别的优化，功劳算不清 |
| 输入与输出长度 | 第 4 章的 `scripts/bench_openai.py`，固定 prompt、固定 `--max-tokens` | 输出变长，ITL 与吞吐一起动 |
| 并发与每档请求数 | 同一个并发档位、同一请求数 | 并发一改，TTFT 与吞吐同时变，分不清是谁的功劳 |
| 采样参数 | `temperature = 0`、固定 `seed`、固定 `top_p` | 采样一变，输出长度分布变，ITL 跟着变 |
| 预热 | 先跑若干请求把 Triton 的编译与 autotune 热掉，再开始计时 | 首次调用的 JIT 编译会算进 TTFT |

流程按 A/B/A 走：基线（不装插件）→ 接入（装插件 + `--compilation_config.custom_ops '["+名字"]'`）→ 再跑一次基线。第三次的作用是看漂移：如果两次基线的差和 A/B 的差同量级，说明这台机器上的差异小于噪声，不能下结论。每一档并发至少重复三次，先看同一配置内 run-to-run 的离散度，再看 A/B 的差。

profiler 单独跑一遍，不要和计时混在一起：挂上 profiler 会改变时间。用 vLLM 的 profiler 或 torch profiler 抓一次前向，在 kernel 名单里找 `_silu_and_mul_kernel`，同时看它的调用次数对不对得上（每层 MLP 一次 × 层数 × 步数）。名单里有、次数对，才算“Triton kernel 真的跑了”；只有 TTFT 变好、名单里没有你的 kernel，那个变化不是你的。

预期要先摆出来，再测。Amdahl 定律：总提速 `= 1 / ((1 − p) + p / s)`，`p` 是这个算子占端到端时间的份额，`s` 是它自身的提速倍数。`p = 1%` 时，哪怕 `s` 无穷大，总提速也只有 `1 / 0.99 ≈ 1.0101`，省 1%；`s = 2` 时是 `1 / (0.99 + 0.005) ≈ 1.0050`，省 0.5%。`p = 5%` 的上限是 `1 / 0.95 ≈ 1.0526`，`p = 10%` 是 `1 / 0.9 ≈ 1.1111`。天花板是份额，不是 kernel 的质量。

这个算子的份额可以先手算一个量级。以 32 层、intermediate 11008、hidden 4096 的 7B 模型为例，每 token 每层 SiluAndMul 的访存是 `3 × 11008 × 2 B ≈ 64.5 KiB`，32 层约 `2.0 MiB`；decode 时每 token 还要把约 14 GB 权重搬一遍（第 1 章算过），SiluAndMul 的访存占比是 `2.0 MiB / 14 GiB` 量级，约万分之一点四。这是访存占比，不是时间占比，但足以说明 decode 阶段的份额远低于 1%。prefill 时激活随 token 数线性涨，`N = 4096` 时这部分到 8.6 GB 量级，占比明显上去了，可 prefill 的时间由 matmul 的算力决定，它的时间份额仍然小。真要报份额，从 profiler 的 kernel 时间占比里取，别拿字节数顶替。

于是端到端的结果大概落在五种里，每一种对应不同动作：

| 结果 | 归因 | 下一步 |
|---|---|---|
| 没变快（差异在噪声内） | 份额太小，Amdahl 上限本来就这么低 | 保留数字，写清份额与上限，不调评测集 |
| 变慢了 | Triton 版常数项不如树内 C++ 实现；或 JIT 开销没热掉 | 看微基准与预热，再决定换 BLOCK 还是放弃 |
| 变快了 | 要先排除别的变量：版本、并发、输出长度 | 用 A/B/A 与 profiler 双证据确认 |
| 完全没变化 | 替换没生效，走的是 `forward_native` 或树内实现 | 回第 3 节查表、查 `name`、查 `_forward_method` |
| 测不出来 | 机器噪声大于效应 | 提高重复次数、固定时钟、换更大 batch 的负载 |

“完全没变化”和“没变快”是两件事：前者是没接上，后者是接上了但份额太小。分不清这两个，报告里的每个数字都不可信。

### 第 5 节 · 小结：一份“自写 kernel 端到端加速报告”（含负面结果分析）

报告要能让人不看你的机器、只看纸面，就判断这些数字是怎么来的、能推广到哪。按下面六节写，每节都要有证据，没有证据的写“待补测”。

1. **环境**：GPU 型号与显存、驱动、CUDA、Triton、vLLM 版本与 commit、模型 checkpoint、量化配置、启动命令原文。
2. **算子**：语义（`silu(x[:d]) * x[d:]`，`d` 的定义）、形状与 dtype 约束、代码位置、与树内实现的差别。
3. **standalone 证据**：正确性（max abs err、allclose 的阈值与理由）、微基准（eager / 树内 C++ / Triton 三行，形状、`BLOCK`、`do_bench` 口径）。
4. **接入证据**：注册表检查、`name` 与 `_forward_method` 的打印、entry point、启用的命令行参数。
5. **端到端**：控制变量表（第 4 节那张）、A/B/A 三组数、每档并发的 TTFT / ITL / 吞吐（至少 p50 与 p95）、profiler 里 kernel 的名字与调用次数。数字全部标“预期/待补测”，测过才改成实测。
6. **负面结果与归因**：哪些尝试没变快、哪些更慢、分别归到哪一类原因。

报告的第 6 点（负面结果与归因）是信息量最大、也最容易写空的部分，也是最容易写空的一节。合格的负面结果长这样：条件写全 → 现象 → 归因 → 证伪或补测办法。三类常见归因各配一个例子：

- **份额太小**：decode 阶段 SiluAndMul 的访存占比约万分之一点四，`p < 1%`，Amdahl 上限 `1.01×`；实测差异落在噪声内。补测办法是换 prefill 为主、batch 更大的负载再量一次，看份额上去之后有没有变化。
- **基线本来就是融合实现**：树内的 `torch.ops._C.silu_and_mul` 访存已经是 `6MD`，Triton 版省不出遍数，只能争常数项；微基准显示 Triton 慢 10%，端到端因此更慢。结论是这条替换在当前版本上不划算，不是“Triton 不行”。
- **没生效**：profiler 名单里没有 `_silu_and_mul_kernel`，`_forward_method` 是 `forward_native`，原因是启用的名字写成了树内注册名而不是替换后的 `name`。补测办法是改名字后重跑，并把两次的打印都留在报告里。

写不出的归因不要编。“不知道”加一条补测计划，比一个编出来的解释有用；报告的价值在于每个说法都有对应的证据位置，而不在于结论好看。第三部分到这里结束：五章的东西合起来证明的是一件事——你会写 kernel、验证它、测出它快在哪、再把它接进引擎。第 29 章会把这个 SiluAndMul 原样搬到 Ascend 上，第 2 节的写法、第 3 节的注册、第 4 节的控制变量，到那边会再用一遍。

**延伸**：[vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)、[`vllm/model_executor/custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)（注册表与分发逻辑）、[`vllm/model_executor/layers/activation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py)（`SiluAndMul` 与树内 Triton 激活 kernel）、[Triton `do_bench` 文档](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)、[Amdahl 定律原文](https://dl.acm.org/doi/10.1145/1465482.1465560)（Amdahl，1967）。

## 本模块检查清单

- [ ] 有一张 GEMM 优化阶梯总表：四个版本各自的算术强度、实测耗时、实测算力、ncu 读数与瓶颈判断
- [ ] 能用 ncu 的两个数（SM% / DRAM%）判断一个 kernel 是计算受限、访存受限还是延迟受限
- [ ] 能手推 online softmax 的 rescale 公式，并按 SRAM 容量算出自己 GPU 的分块参数
- [ ] 写出过 fused attention 与 fused dequant matmul，数值都对齐过
- [ ] 改过一个 CUTLASS GEMM 的 tile 配置并测出了变化
- [ ] 能画出 vLLM 里一个 CustomOp 从注册到分发的完整链路
- [ ] 有一份“自写 kernel 接入引擎前后”的端到端对比报告（含负面结果）
