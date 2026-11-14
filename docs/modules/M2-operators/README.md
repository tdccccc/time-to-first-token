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

一个 1024 token 的序列，attention 的分数矩阵是 1024×1024。把它算出来、写回显存、再读回来做 softmax，S 和 P 两个中间矩阵各占 2 MiB，一写一读就是 8 MiB 的搬运；而 Q、K、V、O 四个张量加起来只有 512 KiB。A100 的 HBM 带宽是 2 TB/s，8 MiB 对应 4.46 µs，512 KiB 只对应 0.26 µs——朴素实现的瓶颈不在乘加，在中间矩阵的往返。FlashAttention 的做法是让 S 和 P 永远不落 HBM：Q、K、V 分块读进 SRAM（静态随机存取存储器：GPU 每个 SM 上的一块高速存储，CUDA 里以 shared memory 的形式使用），softmax 与加权求和在片上完成，只把最终输出写回。这要求先回答两个问题：softmax 的整行最大值与分母，分块之后怎么精确算出来；每 SM 只有 164 KB 的 SRAM，块最大能开到多少。这两个问题答完，kernel 才有得写；写完还要用正确性、计时与 Roofline 三项验收。

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

寄存器总量也要算。每个 SM 有 64K 个 32 位寄存器；一个 block 的寄存器用量是 `线程数 × regs/thread`。`64×64`、4 warp、81 regs/thread 时是 128×81 = 10,368 个，64K ÷ 10,368 ≈ 6.3，最多 6 个 block；但 shared 那一侧只允许 5 个（164 ÷ 32），所以 shared 先顶到天花板。双缓冲 48 KB 时 shared 只允许 3 个。`128×64`、8 warp、81 regs/thread 时是 256×81 = 20,736 个，64K ÷ 20,736 ≈ 3.2，最多 3 个；双缓冲 64 KB 时 shared 只允许 2 个。`128×128`、8 warp、113 regs/thread 时是 28,928 个，最多 2 个；双缓冲 112 KB 时 shared 只允许 1 个。

把这些数并排看：

| 配置 | 单缓冲 shared | 双缓冲 shared | A100 驻留（单/双） |
|------|--------------|--------------|---------------------|
| 64×64 | 32 KB | 48 KB | 5 / 3 |
| 128×64 | 48 KB | 64 KB | 3 / 2 |
| 64×128 | 56 KB | 88 KB | 2 / 1 |
| 128×128 | 80 KB | 112 KB | 2 / 1 |
| 256×64 | 80 KB | 96 KB | 2 / 1 |
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
