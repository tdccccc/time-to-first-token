# M3 · 国产芯片适配

> 前置：第一部分（度量、profiling、压测脚本）与第二部分（vLLM/SGLang 已跑通）。本部分的终点：同一模型、同一评测集，在 NVIDIA 与至少一种国产芯片上都跑通、压测，并写出一份可复现的对比报告。
>
> 规则：国产芯片适配进展极快，任何写死的支持矩阵都会很快过时。本部分不写静态结论，只给方法论和可复现的验证脚本；涉及具体芯片、具体版本的实测结果，按「某年某月某版本实测」记录，由读者自行验证后填写。

## 为什么单开一个模块

调研结论：国产芯片适配**没有任何动手的学习材料**，只有框架插件、一张支持状态追踪表和零散厂商文档。M3 要补的就是这一块：从“知道有这些芯片”到“真的在上面跑起模型、量出数字”。

## 第 25 章 · 芯片与软件栈地图：先画地图，再谈适配

第 27 章会把同一个模型真的放到 Ascend 上跑起来，第 26 章会拆开 CANN 的算子链路。在动手之前，有一个更基础的问题：一块没见过的国产芯片摆在面前，怎么判断它能不能跑推理、缺的是哪一层、这个判断能管多久？

难的不是芯片，是缺少坐标系。厂商文档各写各的，支持状态散在追踪表、插件仓库和发行说明里；同一句“支持”，可能指引擎能启动，也可能指某个算子在某个版本上跑通过。没有统一的坐标，适配就退化成每遇到一块新卡都重新打听一遍，打听来的结论还没有地方放。

本章先画这张地图：9 块芯片的清单、4 面墙的判断、5 层软件栈、插件与统一算子层各自的位置，最后合成一张每个格子都带核实日期的表。地图不回答哪块芯片更强，只回答这块芯片走到了哪一步、下一步该查什么。

**本章目标。** 给定一块国产芯片和一个目标模型，能画出它从驱动到推理引擎的五层软件栈，用生态、显存、通信、精度四面墙判断它能不能跑推理，并写出一张每格都带核实日期、过期即作废的软件栈地图。

### 第 1 节 · 国产芯片全景：先列清单，再问“能不能跑推理”

先列名字。[awesome-china-gpu](https://github.com/bytetora/awesome-china-gpu/blob/HEAD/README.md) 这张追踪表的主线芯片列了 9 块：昇腾、寒武纪、摩尔线程、海光、昆仑芯、天数智芯、壁仞、燧原、沐曦。9 是追踪表当时的条目数，不是国产芯片厂商的全集：没有进表的芯片照样存在，进了表也不等于已经支持推理。名单的用处是给地图定横轴，横轴上每多一个点，就多一套要核实的软件栈；9 块芯片，每块记软件栈与接入方式两项，一共 18 个待填格。

判断能不能跑，用四面墙。Four Walls（四面墙）来自 awesome-china-gpu 这张社区维护的追踪表，是它组织支持状态时用的分析框架，不是任何机构发布的官方标准。四面墙是四个问题：生态（ecosystem）、显存（memory）、通信（communication）、精度（precision）。生态墙问有没有框架插件、容器镜像或原生框架把芯片接进主流推理引擎；显存墙问权重、KV 缓存与激活放不放得下；通信墙问一块卡不够用时，多卡并行能不能扩展；精度墙问量化之后输出还能不能用。四面墙缺一面，结论就不完整：生态再好，显存装不下也跑不起来；显存再大，量化后输出乱码也交不了差。

显存墙先算一笔账，因为它是四面墙里最像算术的一面。7B 模型（70 亿参数）在 FP16 下，每个参数 2 字节，70 亿 × 2 = 140 亿字节，即 14 GB。GB 按十进制算，1 GB = 10<sup>9</sup> 字节，与第 1 章 7B 权重 14 GB 的算法一致。同样算法换两个规模：0.5B 模型是 5 亿 × 2 = 10 亿字节，1 GB；32B 模型是 320 亿 × 2 = 640 亿字节，64 GB。同一个 7B 模型压到 INT8，每参数 1 字节，70 亿字节即 7 GB，正好是 FP16 的一半。这些是算例，不是实测：只算权重，不含 KV 缓存与激活，真实占用只会更大。模型规模翻倍，权重显存跟着翻倍；量化把这一格减半，代价记在精度墙上。

生态墙落到清单上，就是“接入方式”那一栏，大体三种：vLLM 插件（引擎原生支持，插件把平台注册进来）、容器镜像（厂商把驱动、SDK、引擎与模型打包，拉下来就能跑）、原生框架（芯片自己的框架，不走 vLLM 这条路）。三种方式没有高下，只有代价不同：插件复用引擎生态，版本要与引擎对齐；镜像开箱即用，版本钉死在镜像里；原生框架自由度大，接引擎的活要自己干。这一栏比峰值算力稳：峰值数字一个季度就可能变，一块芯片走插件还是走镜像，变得慢得多。

通信墙与精度墙算不出一个整齐的数，却一样能挡住推理。通信墙问多卡并行顺不顺：一块卡装不下或单卡跑不快，就要把模型切开分到多张卡上，切开之后卡与卡之间要不断交换中间结果，交换走卡间互联，带宽与延迟由硬件给定。单卡能跑与多卡能扩展是两个结论，清单里分开记。精度墙问量化之后输出还能不能用：权重从 FP16 压到 INT8，每个权重的表示变粗，舍入误差逐层传到输出；误差会不会被模型本身消化掉，要看 scale（缩放系数）的取法与舍入方式，最终要看评测集上的实测。“支持量化”是功能有没有，“量化后可用”是质量够不够。

清单的时效比内容重要。本章引用的支持状态来自一份核实日期为 2026 年 6 月的引擎对比表；凡是写“支持”的地方，都是截至那个日期的核实结果，不是永久结论。几个月后再读，那份表可能已经过期，其中的状态只能当线索，不能当现况。抄表时状态、版本、核实日期三项一起抄；没有日期的“支持”按未知处理，空着比猜一个强。

一个常见的误解：参数表好看，等于能跑推理。峰值算力与显存容量是硬件能力的上限，能不能跑起来还要看另外三面墙。某块芯片跑不起来，先问是哪面墙挡住了，再下结论；笼统地说“这块芯片不行”，等于什么都没说。本节留下的是 9 行清单与四个定义，后面的章节在这张清单上继续填：第 26 章填昇腾那一行的软件栈细节，第 28 章拿四面墙给覆盖矩阵的结论归因，第 30 章与第 31 章的报告里，每条结论都要带核实日期。

### 第 2 节 · 软件栈分层：驱动 → SDK → 算子库 → 框架插件 → 推理引擎

软件栈分 5 层，从下往上是驱动/固件、SDK/运行时、算子库、框架插件、推理引擎。层数有限，分层才用得上：出了问题，候选位置只有这 5 个，逐个排除总能定位。没有分层时，问题只有一个笼统的名字，叫“国产芯片跑不起来”；有了分层，同一个问题会变成“驱动认不出卡”或“插件没把平台注册上”，后者直接指向下一步要查什么。

每一层干什么、坏了长什么样：

| 层 | 负责 | 缺失或版本不对时 |
|---|---|---|
| 驱动 / 固件 | 认卡、分配显存、提供设备接口 | 设备列表里根本没有这块卡，后面所有层都无从谈起 |
| SDK / 运行时 | 厂商软件栈：编译器、运行时、集合通信，CANN、DTK、MUSA 在这一层 | 编译不过，或运行时找不到符号，报错里出现 SDK 自己的组件名 |
| 算子库 | 芯片侧算子实现与融合算子，AscendC、TBE 写的算子在这一层 | 某个算子没有实现，引擎不报错，回退到 CPU 或低效实现，服务能跑但慢 |
| 框架插件 | 把芯片接进引擎：平台类、算子分发、注意力后端、通信配置 | 引擎启动不了，或启动时认不出这个平台 |
| 推理引擎 | vLLM、SGLang 本身：模型结构、调度、KV 缓存、采样 | 长得最像应用问题：某个请求出错、某个配置不生效，根子可能在下面任何一层 |

症状出现的位置与问题所在的位置经常不在同一层。算子库缺一个实现，用户看到的是推理变慢，慢的症状在引擎层，缺的东西在算子库层；驱动没装好，用户看到的是引擎启动失败，失败的症状在引擎层，缺的东西在驱动层。症状在上，根子在下，所以排查方向是自下而上的：先确认卡认不认得出，再确认 SDK 能不能用，再确认算子在不在，最后才看引擎层的配置。底层问题不解决，上层检查全是白做。

拿一个具体症状走一遍。启动时报找不到设备：先看设备列表里有没有这块卡，没有就是驱动层；有卡但 SDK 报找不到符号，是 SDK 层；SDK 正常但某个算子回退到 CPU，是算子库层；算子都在但引擎认不出平台，是插件层。每一层的检查都是同一句：这一层的东西在不在、对不对。顺序不能倒：从引擎层开始查的人，会先怀疑模型代码与启动命令，绕一大圈才发现是驱动没装。

四面墙与五层叠在一起，症状能从“大概是显存问题”收敛到一个格子：

| 墙 | 主要落点 | 问什么 |
|---|---|---|
| 生态 | 框架插件、推理引擎 | 有没有插件或镜像把这块芯片接进引擎 |
| 显存 | 驱动 / 固件、SDK / 运行时 | 显存分不分得出来，KV 缓存放不放得下 |
| 通信 | SDK / 运行时、框架插件 | 多卡互联带宽够不够，集合通信原语齐不齐 |
| 精度 | 算子库、框架插件 | 量化算子写得准不准，量化后输出能不能用 |

显存墙落在两层上不是笔误：分不出来是驱动层，管不好是 SDK 层。同一条墙落在两层，定位时就要多问一句“是分不出来，还是管不好”。定位记录的写法可以现在就定下来：现象写报错原文或症状，定位写层，解法写命令或改动，验证写怎么确认修好了，四列缺一列，记录就不完整。

插件层与引擎层之间还有一条边界，排查时经常用到：问一句“这个问题换一块芯片还会不会出现”。还会出现，多半在引擎层；换了芯片就消失，多半在插件层或更下面。这条边界把“与芯片有关”和“与芯片无关”切开，排查范围立刻小一半。第 27 章会把它变成三层验证：先确认 NPU 可见，再确认 PyTorch 能认 Ascend，最后确认 vLLM 能识别到 NPU 平台，三层缺一不可。

层与层之间的接口是下一个故障点。接口是上一层调用下一层时用的约定：引擎层调插件层的平台接口，插件层调算子库层的算子名，算子库层调 SDK 层的编译与运行时，SDK 层调驱动层的设备与显存接口。接口稳定，各层就能独立演进；版本升级改了接口，上下层对不上，症状会同时在两层出现。第 27 章的版本矩阵——vLLM、vllm-ascend、CANN、torch_npu 四者的版本对齐——查的就是这种错位，具体版本对应关系以各仓库当前文档为准。

### 第 3 节 · 插件机制：vLLM 是怎么做到一块代码跑多家芯片的

“插件”有两个层次，替换的东西完全不同。第 24 章讲的 CustomOp 是算子级替换：一个 SiluAndMul 换成自己写的 Triton 版，换的是一个算子，模型代码与引擎的其他部分都不动。[vllm-ascend](https://github.com/vllm-project/vllm-ascend) 是平台级插件：它给 vLLM 提供一整套芯片适配，包括平台类 `NPUPlatform`、算子实现、注意力后端、通信、编译配置，模型代码不用改，换的是底下的整套实现。一个算子与一整套实现，规模差一个数量级以上，混为一谈是这一节最要防的误解。

从引擎的角度看，平台级插件做三件事：被发现并加载、注册平台类、在平台被选中之后提供该平台的算子与后端。注册链的顺序是：加载 → 注册平台 → 启动时按设备选平台 → 按平台分发算子。先选平台，再分发算子；算子分发发生在平台选定之后，同一个算子名在不同平台下会分发到不同的实现。没有先选平台，分发就不知道往哪分。

插件的目录地图按职责分，下面是 vllm-ascend 的顶层结构（目录名与增删随版本变化，以仓库当前结构为准）：

| 目录 / 文件 | 职责 |
|---|---|
| `platform.py` | `NPUPlatform`：设备识别与平台选择 |
| `ops/` | 芯片侧的算子实现 |
| `attention/` | 注意力后端的选择与实现 |
| `quantization/` | 量化方案 |
| `distributed/` | 多卡并行与通信 |
| `compilation/` | 图编译一类的配置 |
| `_cann_ops_custom/` | 自定义算子 |

目录名本身就是路标：找一个算子的实现，先想它属于哪一类，再去对应的目录里找，比在整个仓库里翻快得多。顶层结构会随版本新增、改名或合并，拿旧地图找新目录会扑空；核对时记下仓库版本与日期，变了的条目随手更新。

拿 RMSNorm 走一遍分发链，顺序就从一条规则变成一条看得见的路。模型前向里调一次 RMSNorm，调用点在模型代码里，名字是统一的；vLLM 启动时已经按设备选定了 NPU 平台，分发时按平台找实现，落到 `ops/` 或 `_cann_ops_custom/` 里的 Ascend 实现，而不是 CUDA 实现或 PyTorch 参考实现。调用点不变，变的只是名字背后挂的是哪一份实现。定位一个算子，就是沿这条路反方向走：从调用点出发，经过分发点，找到插件里的实现文件，记下路径与函数名。

为什么 vLLM 要做成插件机制？给每块芯片各写一个引擎，成本太高。模型代码、调度逻辑、KV 缓存管理、采样逻辑与芯片无关，只需要写一份；平台识别、算子实现、注意力后端、通信原语与芯片有关，收进插件，每块芯片一份。新增一块芯片的成本，因此从“重写一个引擎”降到“写一个插件”。省掉的是引擎公共部分的重复劳动，插件里的算子仍要一个一个写出来。

对照第 8 章在 NVIDIA 上的部署，两条路的模型代码是同一份，差的是底下的栈：NVIDIA 的算子实现与注意力后端在引擎自带的代码里，装好引擎就能认 CUDA；国产芯片的这些实现由插件提供，引擎本身不认这块芯片，要先装插件、注册平台。这个对照也划出了后面几章的位置：第 26 章与第 27 章拆插件下面的 CANN 与部署链，第 28 章量算子覆盖，模型写法始终不变。

一个常见的误解：插件是打补丁，哪里不行贴哪里。补丁的印象是小、临时、绕过问题，而平台级插件是一套完整的、与引擎平行的芯片适配实现，有自己的平台类、算子库、注意力后端与通信配置。把它当成补丁的人，会低估它的规模，也会在读代码时迷路。另一个相关的误解，是以为模型代码要为每块芯片改一版：平台级插件存在的意义，就是让模型代码不用改，改的是底下的栈。

注意力后端单独占一个目录，说明后端选择是平台级的事情：平台选定之后，注意力走哪一份实现也跟着定下来，模型代码里不用为每块芯片写分支。编译配置同理：同一份模型代码在不同平台下编译的配置可能不同，配置错了，编译出来的图可能跑不起来，也可能悄悄走了低效路径。平台级插件接管所有与芯片有关的选择，留给模型代码的是与芯片无关的部分。

### 第 4 节 · 统一算子层：FlagGems / FlagCX 的思路

统一层是两个库加一个插件。按 [vllm-plugin-FL 文档](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)可以确认这个结构：FlagGems 管算子、FlagCX 管多卡通信，插件把两者接进 vLLM 的分发链。三者缺一个，统一层就不完整：没有 FlagGems，算子还是要每家各写一份；没有 FlagCX，多卡通信还是要每家各写一份；没有插件，两个库再统一也进不了引擎。

分发是一条三级优先级链：先找 FlagGems 的统一实现，没有再找厂商实现，再没有就回退到 PyTorch 参考实现（fallback）。启动时先识别当前芯片、选出后端，分发链的每一级都按这个后端去找。三级链的含义是：统一实现覆盖常见算子，厂商实现补上统一层没覆盖的部分，PyTorch 参考实现兜底，哪怕前两级都没有，结果也算得出来，只是慢。服务能跑起来的概率很高，跑得快不快，取决于落在哪一级：落在第一级是统一实现的速度，第二级是厂商实现的速度，第三级通常是三者中最慢的。

按同一份文档，支持的平台列了 12 个：NVIDIA、Ascend、T-Head、Iluvatar、MetaX、摩尔线程、Tsingmicro、海光、燧原、昆仑芯、阿里巴巴 PPU、CPU ARM64。名单与数量以当前文档为准，平台增删比芯片型号变化得快。12 个平台共用一套分发链，一个算子写一份就能在这些平台上跑；没有统一层时，同一个算子要写 12 份，改一个 bug 也要改 12 处。

回退的代价可以用一个算例讲清，它不是实测。假设某个算子占总时间的 20%，它的 PyTorch 参考实现比芯片侧实现慢 5 倍：总时间变成原来的 0.8 + 0.2 × 5 = 1.8 倍，也就是慢了 80%。同一个算例换一组数：这个算子只占总时间的 1%，同样慢 5 倍，总时间是 0.99 + 0.01 × 5 = 1.04 倍，只慢 4%。回退本身不可怕，回退落在关键路径上才可怕。判断一个回退要不要紧，要把支持状态与耗时占比一起看：只看状态不看占比，会把小问题当成大问题，也会把大问题当成小问题。

统一层解决的是“一份 kernel 代码跑多家芯片”，没解决的是“每家芯片的峰值性能”。统一实现为了可移植，会牺牲一部分特化优化：一个 kernel 要在 12 个平台上都跑对，就不能用某块芯片独有的指令与访存技巧，写法要取各家都能接受的交集；交集的写法能跑，但跑不到某块芯片的上限。要榨干某块芯片的性能，最终还是要写它的原生算子：第 26 章的 AscendC、第 29 章的移植案例走的就是这条路。这与第 23 章 Triton 与 CUTLASS 的取舍是同一个逻辑，换了一块芯片：一份代码跑多处、省人力，特化为特定硬件、榨性能。

[FlagTree](https://github.com/flagos-ai/FlagTree) 是 FlagGems 背后的算子开发框架：算子用一套抽象写出来，再针对不同芯片生成实现。FlagGems 是生成出来的统一算子库，FlagTree 是写算子与生成实现的框架本身。第 29 章会回答三个问题：一个算子的“统一描述”长什么样，针对不同芯片的生成发生在什么时候，某块芯片上统一实现跑不起来时回退链是怎样的。

FlagCX 管的是多卡之间的集合通信：张量并行把模型切开之后，卡与卡要交换中间结果，交换用的原语由 FlagCX 统一提供，再由各家芯片的后端承接。算子统一解决单卡上的计算，通信统一解决多卡之间的协同，缺一个，多卡推理就接不起来。25.1 的通信墙问“多卡能不能扩展”，落到软件栈上就是这一层。单卡能跑的时候 FlagCX 的存在感很低，一上多卡，它就在关键路径上。

什么时候用统一层、什么时候写原生算子，可以定一条判断规则：先上统一层把服务跑起来，再看 profiling（性能分析）的结果，哪个算子占的时间最多、又落在第三级分发上，就优先写原生实现补上去。统一层管从 0 到 1 的覆盖，原生算子管从 1 到快的峰值；顺序倒过来，先给每个算子写原生实现，人力会陷在覆盖面上，服务迟迟跑不起来。三级分发链还有一个边界情况要标出来：统一实现与厂商实现都没有，落到 PyTorch 参考实现，服务还能跑；如果连参考实现都没有，这一级是往下一级走还是直接报错，文档没有写，这里不猜，标“待核实”，读分发逻辑的源码确认。

### 第 5 节 · 小结：画一张带日期的软件栈地图

把前四节合成一张表：横轴是芯片，纵轴是软件栈分层，每个格子写“支持状态 + 版本 + 核实日期 + 来源”。横轴的 9 块芯片来自 25.1 的清单，纵轴的 5 层来自 25.2 的分层，格子里的内容来自 25.3 的插件地图与 25.4 的统一层笔记。一个完整的格子长这样：`vllm-ascend vX.Y.Z，2026-06，社区对比表，未在本机复核`。版本、日期、来源三项齐全才算记录；没有日期的格子一律视为未知，不知道它说的是哪个版本，也不知道它过期没有。

下面是单格的记录格式，照这个填，信息才落得进格子：

| 字段 | 写什么 | 示例 |
|---|---|---|
| 芯片 | 厂商 + 型号 | 昇腾 910B |
| 层 | 5 层之一 | 框架插件 |
| 状态 | 支持 / 部分支持 / 不支持 / 未核实 | 部分支持 |
| 版本 | 插件或 SDK 的版本号 | vllm-ascend x.y.z |
| 核实日期 | 年-月，来自文档、仓库或本机验证 | 2026-06 |
| 来源 | 链接，或写“本机实测” | 追踪表 / 本机实测 |

再把骨架画出来。下表是 9 块芯片 × 5 层的空表，空格代表未核实，不是“不支持”；填的时候在格子里写状态与版本，行尾写这一行最后一次核实的日期：

| 芯片 | 驱动 / 固件 | SDK / 运行时 | 算子库 | 框架插件 | 推理引擎 | 最后核实 |
|---|---|---|---|---|---|---|
| 昇腾 |  |  |  |  |  |  |
| 寒武纪 |  |  |  |  |  |  |
| 摩尔线程 |  |  |  |  |  |  |
| 海光 |  |  |  |  |  |  |
| 昆仑芯 |  |  |  |  |  |  |
| 天数智芯 |  |  |  |  |  |  |
| 壁仞 |  |  |  |  |  |  |
| 燧原 |  |  |  |  |  |  |
| 沐曦 |  |  |  |  |  |  |

定一条制度：本部分所有关于芯片与软件栈的事实陈述都必须带核实日期，超过 3 个月未核实即视为作废，需要重新核实。3 个月这个期限可以拿一个具体日期走一遍：假设某条状态是 2026 年 11 月 12 日核实的，3 个月后是 2027 年 2 月 12 日，到那天还没有重新核实，这条状态就作废，格子退回“未核实”。期限存在的理由是版本变化快：版本一变，支持状态、覆盖情况、性能数字都可能跟着变；写死一个没有日期的结论，等于把过期信息当成现成事实。

检查清单并进总清单，加两条：Four Walls 四面墙能复述吗；地图上的每个状态都有核实日期吗。第一条防的是尺子没掌握，后面归因没有尺可用；第二条防的是日期没写，地图看起来完整，实际上每一格都无法信任。本章没有硬件、没有实测，所以地图上现在全是空格，这是它的初始状态，不是结论。

地图的用处不在它永远正确，而在它给了每一件新事实一个位置：新核实到的状态有地方写，过期的结论有地方标，下一步该查哪一层一目了然。第 26 章从昇腾这一行往下拆 CANN 与 AscendC；第 27 章把插件这一层落到部署与版本矩阵；第 28 章用算子覆盖矩阵替换掉“算子库”那一列的空格；第 30 章与第 31 章的报告里，每个数字都要能追到地图上的一格和它的日期。

**延伸**：[awesome-china-gpu 追踪表](https://github.com/bytetora/awesome-china-gpu/blob/HEAD/README.md)、[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[vllm-plugin-FL 文档](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)、[FlagTree](https://github.com/flagos-ai/FlagTree)、[FlagGems](https://github.com/FlagOpen/FlagGems)。

## 第 26 章 · CANN 与 AscendC 深讲：算子从定义到编译到调用

在 Ascend 上跑 vLLM，有一类问题不报错：服务起来了，输出也对，但 profiler 里那个本该走芯片实现的算子，落到了一个慢得多的实现；或者启动日志里出现一句 `aclnn... not found`，模型照样能跑，只是某个融合路径悄悄关了。要查这类问题，得先知道一个算子从源码到被调到，中间在几处登记过。

一个算子要过四道手。**定义**：写出接口（shape、dtype、format）与计算逻辑，再写 host 侧的切分参数。**编译**：把源码编成 NPU 上能跑的 kernel，连同原型信息与 host 接口一起打成算子包。**注册**：把算子包放进 CANN 的搜索路径，再把算子接到 PyTorch 与 vLLM 的名字系统里。**调用**：模型前向按名字找到实现，经 aclnn 下发到 Runtime，最后落到 kernel。四道手分属四段代码，任何一道没接上，症状都不在出错的那一道：kernel 编好了但没生效，问题常在注册；算子名解析到了但运行时报找不到符号，问题常在算子包路径。

**本章目标。** 能画出 CANN 的分层并指出算子开发落在哪层；能读懂一个 AscendC 最小算子的流水线并算出它的访存账；能沿 RMSNorm 走完定义、编译、注册、调用四环，指出每环的文件与接口；能判断一个算子该不该自己写。

版本锚点见第 27 章；本章只讲结构与链路，不重复具体版本号。下文出现的目录与接口名按当前公开源码整理，版本升级后可能改名或合并，以当前 CANN 与 vllm-ascend 版本为准。

### 第 1 节 · CANN 软件栈：先分清每一层是干什么的

CANN 不是一个大库，而是一套有明确分层的栈。官方逻辑架构自顶向下分五层，再加上栈底的驱动与固件：

| 层 | 主要组件 | 这一层坏了，症状长什么样 | 先查什么 |
|---|---|---|---|
| 计算语言 / 接口 | AscendCL（`acl`、`aclrt`、`aclnn` 接口） | 调用处报参数、流、显存错误；`aclnnXxx` 找不到算子 | 接口名与 CANN 版本是否对得上 |
| 计算服务层 | 算子库 AOL（NN、BLAS、融合算子）、调优引擎 AOE、框架适配 Framework Adaptor | 算子能调到但慢或精度不对；框架侧找不到算子 | 算子覆盖与实现版本 |
| 计算编译层 | 图引擎 GE、图编译器、算子编译器（Ascend C 编译器、TBE 编译器） | 编译不过、tiling 越界、UB 超限 | 源码、编译日志、tiling 参数 |
| 计算执行层 | Runtime、Graph Executor、HCCL（集合通信库）、DVPP / AIPP | kernel 起不来、流与事件错、多卡通信卡住 | runtime 日志、集合通信配置 |
| 计算基础层 | 共享虚拟内存、设备虚拟化、主机-设备通信 | 设备不可见、内存映射失败 | 驱动、固件、容器设备映射 |
| 驱动与固件（五层之下） | driver、firmware | `npu-smi info` 看不到卡 | 驱动安装、设备节点、容器权限 |

这张表与第 25 章的五层不是同一张表。第 25 章按“能力由谁提供”切：驱动 → SDK → 算子库 → 框架插件 → 推理引擎；CANN 五层按“CANN 自己的组件”切。两张表要对得上：驱动与固件对应第 25 章的驱动层；执行层、编译层、基础层，加上服务层里的框架适配，大致对应那里的 SDK（PyTorch 侧由 `torch_npu` 承接）；服务层的 AOL 对应算子库层；vllm-ascend 对应框架插件层；vLLM 对应推理引擎层。对不上的地方要记住粒度不同：CANN 的服务层同时装着算子库、调优与框架适配，图引擎与编译器在编译层；第 25 章把算子库与插件拆成两层，不展开 CANN 内部的编译层。各组件在官方架构图里的归属可能随版本微调，以当前 CANN 文档为准。

算子开发落在哪一层，答案不是单独一层。源码在**计算编译层**被编译器处理：Ascend C 源码或 TBE 的 DSL 在这里变成 NPU 二进制。产物进**计算服务层**的算子库：算子包被安装到 OPP（算子包目录）里，和厂商预置算子放在同一个检索空间里。调用走**计算语言层**：框架最终调的是 `aclnn` 开头的 C 接口。要把这三段接起来，还差**框架适配**：`torch_npu`（PyTorch 的 Ascend 适配层）把 PyTorch 的算子调用翻译成 AscendCL 调用。所以“写一个算子”横跨编译层与接口层，“让 vLLM 调到它”还要跨框架适配与插件层；只写了 kernel 而没做后两步，算子等于不存在。

图引擎 GE 与单算子直调是两条路。GE 管整图：图优化、算子融合、内存复用、下沉执行；ATC 可以把离线模型编译成 OM（离线模型），推理时整图加载。vLLM 这类引擎在 eager 模式下走单算子直调：每个算子经 `torch_npu` 或 vllm-ascend 的扩展调到 `aclnn`，Runtime 逐个下发。图模式是另一条路，可能用 ACLGraph 捕获重放，也可能经图编译下沉；当前 vllm-ascend 在什么条件下走哪条，以它当前版本的配置项为准。两条路的排查入口不同：单算子看 `aclnn` 与算子包，图模式还要看图编译日志与融合结果。

拿一个具体报错走一遍。启动日志里出现 `aclnnAddRmsNormBias` 相关的 not found，先别改模型：这句话的意思是“算子名查到了，但 OPP 里没有这份实现”。检查顺序是：算子包有没有装、`ASCEND_CUSTOM_OPP_PATH` 有没有在当前 shell 里生效、算子包是不是为当前 SoC 编的、CANN 版本与算子包要求的版本是否一致。这四步都在服务层与接口层，和 vLLM 的模型代码无关。另一类报错是 `import torch_npu` 成功、vLLM 却认不出 NPU 平台，那在框架适配与插件层，属于第 27 章的范围。

分层的边界也要说清：CANN 五层不是严格的调用栈。AscendCL 是一组对外接口，框架与算子都通过它往下调，但它本身不规定层与层的调用顺序；框架适配虽然画在服务层，实际位置在 CANN 与 PyTorch 之间；GE 可以被绕过，也可以和单算子调用混用。把五层当成“每一层只能调下一层”的 OSI 模型，会在排查时把方向找错。它的用法是给症状一个候选位置，不是给调用关系画一条唯一路径。

### 第 2 节 · TBE 与 AscendC：两代算子抽象长什么样

两代抽象共用同一条链路：定义、编译、调用。差别不在链路，而在每一环里你写什么、编译器替你做什么。先把链路钉住：定义包含接口（输入输出的 shape、dtype、format）与计算逻辑，外加 host 侧的切分参数（tiling，把一次调用按 UB 容量与核数切成若干块）；编译把源码变成 NPU 二进制、算子原型信息与 host 接口库；调用是框架按算子名找到实现、传张量、下发 kernel。三个环节的产物一环扣一环：定义里写错 dtype，编译能过、调用时才对不上；编译出的算子包没注册，调用时按名字找不到。

| 环节 | TBE（上一代） | AscendC（当前主力） |
|---|---|---|
| 定义 | 用 DSL（domain-specific language，领域专用语言）写计算表达式与调度：算什么、怎么切 | 用 C++ 写算子类：`Init` 备地址与 buffer，`Process` 写搬运、计算、搬回 |
| 编译 | TBE 编译器读 DSL，生成 IR，做双缓冲、流水同步、内存分配等 Pass，再生成 C 代码并编译 | Ascend C 编译器把 C++ 编成 NPU 二进制；host 侧另编 tiling 与原型 |
| 调用 | 算子包装进 OPP，经 aclnn 或框架算子名调用 | 同样进 OPP，经 aclnn 或 torch 算子名调用 |

TBE 的核心是“描述 + 生成”。开发者用 DSL 写计算逻辑，用调度原语描述切分；TBE 编译器负责中间表示、编译优化与代码生成，双缓冲、流水线同步、片上内存分配这些事由编译器的 Pass 完成。TBE 还有一条更底层的 TIK 路径，直接用 intrinsic 写，控制更细、写起来更重。TBE 的抽象把双缓冲、同步、内存分配这三件事交给编译器，开发者写计算与切分，代码因此更短、显式同步更少。

AscendC 把这三件事交还回来。算子是 C++ 类，`Init` 里设全局内存地址、给队列分配 buffer，`Process` 里显式调 CopyIn、Compute、CopyOut；双缓冲靠队列深度，同步靠 `EnQue` / `DeQue`。编译器仍然在：Ascend C 编译器把 C++ 变成 NPU 指令，不支持的 API、对齐与 UB 容量问题会在编译或算子校验时拦下；tiling 的边界错误则常在运行或单算子测试时暴露。但编译器不再替你决定队列多深、buffer 多大、哪一步和哪一步重叠，这些成了源码的一部分。两代的差别不是“新的更快”，而是控制权与信息量的交换：TBE 写起来短，能做的优化受 DSL 表达范围限制；AscendC 写得长，能把搬运粒度、队列深度、核间切分捏到手里，写错也会直接变慢或算错。没有实测，不给“哪代更快”的结论。

TBE 仍然要读，原因有两个。存量算子库里还有 TBE 时代的实现，读厂商算子包或旧代码时会遇到，认不出 DSL 就接不了手。更实际的原因是，TBE 恰好把“算子抽象”抽象掉的三样东西摆在明面上：调度、内存、同步；AscendC 让你手写的，正是这三样。对照着看，才知道自己在写什么。新算子开发以 AscendC 为主，TBE 的定位与支持状态以当前 CANN 文档为准。

编译的产物要说准：它不是一个孤零零的二进制，而是一个算子包。包里通常有四样：按 SoC 编的 kernel 二进制、算子原型与信息库（shape 推导、dtype 支持、format 约束）、tiling 实现、host 侧的 aclnn 接口库（头文件与动态库）。注册信息不是一张单独的映射表，而是算子包里的原型与信息库，安装后被 OPP 检索到。这也解释了为什么“编译过了”不等于“能调到”：编译只保证包做出来了，注册与检索是安装和路径的事。

失败要分三类，第一现场不同。编译错在源码与编译器之间：C++ 语法、不支持的 API、UB 超限、静态可查的 tiling 越界。加载错在算子包与路径之间：`aclnn` 找不到、符号未定义、SoC 不匹配。运行错在数据与同步之间：对齐、尾巴、队列同步、tiling 与 kernel 假设不一致。三类的排查入口分别是编译日志、`ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`、以及 kernel 内的搬运与同步代码。

定义这一环的接口约定最容易被低估。RMSNorm 的接口不只是“输入一个张量、输出一个张量”：归一化沿最后一维、weight 的长度等于 hidden size、eps 以属性还是张量传入、输出除了 y 还有没有 rstd，这些都要在原型里写清楚。接口写错时，编译往往能过，报错发生在调用现场（shape 或 dtype 不匹配），根子却在定义。修定义比在调用处加转换更对症。

### 第 3 节 · AscendC 流水线模型：读懂一个向量加法算子

向量加法是最小的完整流水线：三个全局内存地址进，一次搬运、一次加法、一次搬回，没有归约、没有矩阵乘。读它不是为了学加法，是为了看清 AscendC 把一个 kernel 拆成了哪些部件、这些部件怎么接力。

kernel 入口长这样（示意，不保证编译通过；完整可编译版本以 CANN 文档的 `add_custom` 样例为准）：

```cpp
// 示意：AscendC 向量加法 kernel 入口，不保证编译通过
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 只用向量核
    GET_TILING_DATA(tilingData, tiling);             // 取 host 侧算好的切分参数
    // workspace 判空、系统 workspace 处理、tiling key 分支在这里省略
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}
```

五个参数各管一件事。`x`、`y`、`z` 是三个张量在全局内存（GM，Global Memory）里的地址；`workspace` 是一段临时显存，算子需要跨核交换或暂存时用；`tiling` 是 host 侧算好的一段参数，里面有总长度、每块长度、块数、尾巴长度。kernel 不自己算这些，它从 `tiling` 里读。`GET_TILING_DATA` 把这段字节还原成结构体。`extern "C"` 让入口不被 C++ 改名，`__global__ __aicore__` 标明这是跑在 AI Core 上的 kernel。`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明只用向量核（AIV，AI Vector Core），矩阵乘那类算子会声明用矩阵核（AIC，AI Cube Core）或混合模式；核类型在编译期固化，改它要重编。并非所有产品都支持 `KERNEL_TYPE_AIV_ONLY`，取值约束以当前 CANN 文档为准。

算子类把初始化和执行分开，执行里再拆成三个阶段。下面是与入口配套的类骨架（同样只是示意）：

```cpp
// 示意，不保证编译通过
constexpr int32_t BUFFER_NUM = 2;      // 2 表示双缓冲
constexpr int32_t TILE_LENGTH = 1024;  // 每块元素数，实际由 tiling 决定

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ float*)z, totalLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        // 这里是单核示意；多核时每个核只处理自己那一段
        tileCount = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);  // GM -> UB
        DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t i) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);             // 在 UB 上算
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t i) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);  // UB -> GM
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    int32_t tileCount;
};
```

部件各自管什么：

| 对象 | 管什么 |
|---|---|
| `GlobalTensor<T>` | GM 上的张量视图；`SetGlobalBuffer` 把裸地址与长度包成它 |
| `LocalTensor<T>` | 片上统一缓冲区（UB，Unified Buffer）上的张量视图，由队列分配 |
| `TPipe` | 算子内部的资源管理器：UB 与同步事件都经它申请 |
| `TQue<..., N>` | 队列：承载 `LocalTensor`，负责交接与同步，模板参数 N 是 buffer 深度 |
| `TBuf<...>` | 临时 UB 缓冲，放中间结果；本算子没用到 |

三个阶段与三条硬件流水对应：

- CopyIn 走 MTE2（GM → UB 的搬运单元），`DataCopy` 把一块输入搬进 UB，`EnQue` 把 buffer 交给下游。
- Compute 走 VEC（向量计算单元），`DeQue` 等到数据就绪，`Add` 算完，结果放进输出队列，`EnQue`。
- CopyOut 走 MTE3（UB → GM 的搬运单元），`DeQue` 取结果，`DataCopy` 搬回全局内存，`FreeTensor` 释放 buffer。

三条流水是独立的硬件单元，所以上一块在算的时候，下一块可以已经在搬。这是“流水线”三个字的物理基础；`EnQue` / `DeQue` 则是把数据依赖写成同步，保证消费者不会早于生产者取数。

双缓冲就是队列深度 2：`BUFFER_NUM = 2` 加 `InitBuffer(queue, 2, bytes)`。深度 1 时，CopyIn 搬完才能 Compute，Compute 算完才能 CopyIn 下一块；深度 2 时，队列里能同时挂两块，一块在算、一块在搬。代价是 UB 占用翻倍：三个队列各两份，UB 占用是每个队列只开一份时的两倍。UB 不够时，要么减小块，要么少开队列。双缓冲重叠的是“等待”，不是“搬运量”：它不能让 12 MiB 变少，只能让搬运尽量贴着带宽上限走。

把账算出来。N = 2^20 = 1,048,576 个 fp32 元素：x 与 y 各 1,048,576 × 4 B = 4 MiB，z 同样 4 MiB，一次加法读 8 MiB、写 4 MiB，共 12 MiB = 12,582,912 B；加法做 1,048,576 次。平均每次加法搬 12 B，或写成算术强度：1,048,576 ÷ 12,582,912 ≈ 0.083 次加法/字节（一次加法算 1 FLOP，即 0.083 FLOP/B）。这是典型的访存主导算子：计算单元几乎不花时间，时间花在搬运上。第 1 章的 Roofline 判断在这里直接复用：算术强度远低于脊点，优化目标首先是让搬运不停。

分块改变的是搬运次数，不是总量。每块 1024 个元素：1,048,576 ÷ 1024 = 1024 块，每块读 1024 × 4 × 2 = 8,192 B、写 4,096 B，共 12,288 B，1024 块合起来仍是 12 MiB。每块 4096 个元素：256 块，每块 49,152 B，总量不变。块大，每次搬得多、同步次数少，但 UB 占用大；块小，接力次数多、同步开销占比高。块大小没有脱离硬件的“最优值”，要拿 UB 容量、搬运带宽与同步开销去试。

流水的收益可以算。假设 CopyIn、Compute、CopyOut 各花 1 个单位时间，有 n 块：串行是 3n；排成流水线，第一块走完三步要 3，之后每块多 1，总共 3 + (n − 1) = n + 2。n = 4 时，12 变成 6，2 倍；n = 1024 时，3072 变成 1026，接近 3 倍——3 是三个阶段给的上限。这组数是算例，不是实测：它假设三步等长、交接无开销、buffer 足够。向量加法的 Compute 远小于搬运，真实上限比 3 低，时间最终仍由 12 MiB 除以带宽决定。流水线能做的是把“等搬运”和“等计算”叠起来，不能突破带宽。

尾巴要单独处理。向量长 10,000、每块 1024：块数是向上取整的 10，前 9 块各 1024 个元素，最后一块只剩 10,000 − 9 × 1024 = 784 个。最后一块若按 1024 搬，会多读多写 240 个元素，越界或污染结果。UB 的搬运按 32 字节对齐访问，长度不是 32 字节整数倍时，常见两种处理：用 `DataCopyPad` 按有效长度搬非对齐尾巴，或把尾巴交给标量循环。具体写法以文档为准，但账要记住：分块算术里永远有一块不满，tiling 必须把尾巴长度传给 kernel。

核类型是最后一块。AIV 跑向量计算，AIC 跑矩阵乘；`KERNEL_TYPE_AIV_ONLY` 表示只用向量核。有的算子会声明混合模式，把向量部分与矩阵部分分给两种核。核类型在编译期确定，决定了这个算子的性能天花板落在哪：逐元素算子落在 AIV，矩阵乘落在 AIC。第 28 章做覆盖分析时，状态一列除了“有实现/没实现”，还该记核类型，因为它决定了这个算子还有多少优化空间。

### 第 4 节 · 一个算子的完整链路：从定义到被 vLLM 调用

前面三节拆开讲层、讲抽象、讲流水线；这一节选一个算子走完整条链。选 RMSNorm 的理由是它在每一层 transformer 里都出现，且同时有两条路径：一条走 CANN 内置算子，一条走 vllm-ascend 的自定义融合算子。两条路径正好把“看得到源码”和“只看得到接口”两种情形都覆盖了。

下表按当前公开源码整理，未在本机逐行核对；目录与接口名随版本变化，以当前 CANN 与 vllm-ascend 版本为准。

| 环节 | 产物 | 在哪找 | 怎么确认 |
|---|---|---|---|
| 定义 | 算子原型、tiling 参数、kernel 源码 | `op_host/<op>.cpp`（原型注册、shape/dtype 推导、`TilingFunc`）、`op_host/<op>_tiling.h`、`op_kernel/<op>.cpp`（`Init` / `Process`） | 能指出三份文件各自的职责 |
| 编译 | 算子包：kernel 二进制、原型与信息库、tiling 实现、aclnn 接口库 | 算子工程根目录的 `build.sh`，产物在 `build_out/` 或 `vendors/<vendor>/` | 编译日志、产物目录里有 `op_api/include` 与 `op_api/lib` |
| 注册 | 三处登记：CANN 的 OPP 路径、PyTorch 的算子库、vLLM 的 op 类替换 | `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`；`csrc/torch_binding.cpp` 的 `TORCH_LIBRARY_EXPAND` 与 `EXEC_NPU_CMD`；`vllm_ascend/utils.py::register_ascend_customop` 里的 `CustomOp.register_oot` | `printenv`、`torch.ops._C_ascend` 里有没有这个名字、模型里的类有没有被替换 |
| 调用 | 前向里的一次调用，最终落到 aclnn 与 kernel | `vllm/model_executor/layers/layernorm.py::RMSNorm` → `vllm/model_executor/custom_op.py::CustomOp` → `vllm_ascend/ops/layernorm.py::AscendRMSNorm.forward_oot` → `torch_npu.npu_rms_norm` 或 `torch.ops._C_ascend.npu_add_rms_norm_bias` | profiler 里能看到对应 kernel，调用栈能对上 |

定义这一环，自定义算子与内置算子的可见度不同。自定义算子由 `msopgen`（算子工程生成工具）从一份原型 JSON 生成工程（`-c ai_core-<soc>` 指定目标 SoC）：JSON 写算子名、输入输出的 name/format/dtype、属性；`op_host` 里的 `TilingFunc` 按输入 shape 与 UB 容量算每块长度、块数、尾巴；`op_kernel` 里是上一节那种 `Init` / `Process` 结构。RMSNorm 的普通路径走的是 CANN 内置算子：`torch_npu.npu_rms_norm(x, weight, eps)` 返回 `(y, rstd)`，定义与编译都在厂商的算子包里，能看到的是接口与行为，不是源码。融合路径 `npu_add_rms_norm_bias` 才是 vllm-ascend 自己接的自定义算子，能在 `csrc/` 的 C++ 包装与算子包里看到落点。这也解释了为什么“读一个算子的定义”有时只能读到接口文档：内置算子不开放源码，能核对的是 dtype、shape、eps 这些接口约定。

编译这一环把源码变成算子包。`build.sh` 调 cmake 与 Ascend C 编译器，按工程里配置的 SoC 编出 kernel，再打包成 `custom_opp_*.run` 或直接生成 `vendors/<vendor>/` 目录。包里的四样东西前面说过：kernel 二进制、原型与信息库、tiling、aclnn 接口库。编译失败常见三种：源码用了当前编译器不支持的 API；UB 超限或 tiling 越界；SoC 型号与实机不一致。这三种都在编译或算子校验阶段暴露，第一现场是编译日志，不是运行日志。

注册是整条链里最容易被低估的一环，因为它不产生任何计算，却决定调用时跑的是哪份实现。这里有三个不同层次的注册，混在一起谈就会定位错：

1. **CANN 算子包注册**：算子包安装到 OPP 搜索路径，或通过 `ASCEND_CUSTOM_OPP_PATH` 指过去，`op_api/lib` 加进 `LD_LIBRARY_PATH`。没有这一步，`aclnnXxx` 找不到实现。
2. **PyTorch 算子注册**：`torch_npu` 的内置算子由它自己的 op_plugin 注册；vllm-ascend 的自定义算子由 `csrc/torch_binding.cpp` 里的 `TORCH_LIBRARY_EXPAND` 与 `EXEC_NPU_CMD` 暴露成 `torch.ops._C_ascend.<name>`，`vllm_ascend/platform.py` 里导入编译扩展时触发注册。注册失败时，插件会打印警告并关掉 custom ops；有兜底的算子退回 `torch_npu` 内置实现或 `forward_native`，没有兜底的会直接报错。
3. **vLLM 层注册**：`vllm_ascend/utils.py::register_ascend_customop` 把 `"RMSNorm"` 映射到 `AscendRMSNorm`，再调 `CustomOp.register_oot` 替换模型里的类。没有这一步，torch 算子存在，模型却仍然走原来的 `forward_native`。

三处注册对应三个“名字”：CANN 侧的 aclnn 算子名、PyTorch 侧的 `torch.ops` 名、vLLM 侧的 Python 类名。定位时先说清是哪一处没对上，再查对应的表。内置算子省掉第一处的自定义算子包，但后两处仍要把它接进模型；少一处，模型就还是走原实现。

调用这一环从模型代码开始。`RMSNorm` 在 vLLM 里是一个 `CustomOp`，构造时按平台选好 `_forward_method`；NPU 是 out-of-tree（引擎外挂）平台，选中的是 `forward_oot`。`register_ascend_customop` 已经把 `RMSNorm` 换成 `AscendRMSNorm`，所以实际执行的是 `vllm_ascend/ops/layernorm.py::AscendRMSNorm.forward_oot`。它按有没有残差分两条路：有残差时，custom ops 开着走融合的 `torch.ops._C_ascend.npu_add_rms_norm_bias`，关着走 `torch_npu.npu_add_rms_norm`；没有残差时走 `torch_npu.npu_rms_norm`。这几种走法最后都落到 aclnn 接口，由 Runtime 下发到 kernel。

于是“算子名在哪一层被解析”有了准确答案：解析发生三次。Python 层把 `RMSNorm` 解析成 `AscendRMSNorm`；PyTorch dispatcher 按 `PrivateUse1`（PyTorch 给自定义设备保留的 dispatch key）把 `torch.ops._C_ascend.npu_add_rms_norm_bias` 解析到 C++ 实现；CANN 的 OPP 再把 `aclnnAddRmsNormBias` 解析到具体 kernel。三次里任何一次断了，症状都不在同一个地方：第一次断了，模型走原实现；第二次断了，抛“算子不存在”；第三次断了，运行时报 `aclnn` 找不到。

怎么确认链路通没通，命令比猜测可靠：

```bash
npu-smi info                                    # 驱动/固件是否认出卡
python -c "import torch, torch_npu; print(torch_npu.__version__)"
python - <<'PY'                                 # 导入扩展才会注册 _C_ascend
import torch, torch_npu, vllm_ascend.vllm_ascend_C
print(hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias"))
PY
printenv ASCEND_CUSTOM_OPP_PATH                 # 没输出说明当前 shell 没设
```

这几条只回答“有没有”，不回答“快不快”。要确认某个调用真的落到了哪个 kernel，用 `msprof` 或 vLLM 的 profiler 看 kernel 名与调用栈；`msprof` 的用法与字段以当前 CANN 文档为准。本章没有在实机上跑这条链，上面的命令是核对路径用的，不构成“已跑通”的证据。

三个问题沿链路回答，答案里要分清“结构推断”与“已核实”。第一，算子名在哪层解析：三次，位置如上；具体到某个版本的 `forward_oot` 走了哪条分支，要读当前版本源码。第二，芯片侧没有实现时会发生什么：取决于缺的是哪一层。vLLM 层没替换，走 `forward_native`；PyTorch 层没有对应算子，报 not implemented；CANN 层没有 kernel，`aclnn` 找不到。三种后果不同，判断靠日志与 profiler，不要假设一定会回退。第三，换一个 CANN 版本，哪一环最可能变：编译产物与算子包的 ABI、aclnn 接口、OPP 路径最可能变；定义里的接口约定与模型侧的调用点相对稳定。前两条是结构推断，第三条是版本问题的通用方向，具体差异要等第 27 章装好环境后对照。

回退的代价可以算。设某个算子占一次前向总时间的比例是 p，回退后它慢 s 倍，其余部分不变，则总时间变成 (1 − p) + p × s，慢的倍数就是这个值。p = 20%、s = 10 时，总时间是 0.8 + 2 = 2.8，慢 2.8 倍；p = 20%、s = 2 时是 0.8 + 0.4 = 1.2，慢 1.2 倍；p = 5%、s = 10 时是 0.95 + 0.5 = 1.45，慢 1.45 倍。这组数是算例，s 是假设值：真实慢倍数要等覆盖分析时逐个算子量。结论是判断回退要不要紧，看的是“耗时占比 × 慢的倍数”，不是“有没有实现”这一列。

### 第 5 节 · 小结：CANN 学到什么程度算够

“够”不是背完 API，是三个能当场做的动作：读懂一个最小算子、把一条报错定位到四环中的一环、判断一个算子该不该自己写。

**能读懂最小算子。** 合上文档，对着 `add_custom` 的代码能说出六件事：入口的五个参数各是什么，tiling 从哪里来；`Init` 里做了哪三件事（设全局地址、分配队列 buffer、按 tiling 切分）；`Process` 里三个阶段怎么接力，`EnQue` / `DeQue` 各保证什么；`BUFFER_NUM = 2` 为什么能重叠、代价是什么；N = 2^20 个 fp32 的搬运量是 12 MiB、算术强度约 0.083 次加法/字节，因此是带宽受限；尾巴块怎么处理、为什么 `DataCopy` 要求对齐。六件事答得出，最小闭环就算读懂了。

**能定位错误。** 拿到现象，先归到四环，再查对应位置：

| 现象 | 落在哪一环 | 先查什么 |
|---|---|---|
| `aclnnXxx not found` | 注册（CANN 侧） | 算子包、`ASCEND_CUSTOM_OPP_PATH`、SoC 是否匹配 |
| `undefined symbol` / 库加载失败 | 注册（CANN 侧）或版本 | `LD_LIBRARY_PATH`、CANN 与 torch_npu 版本 |
| 编译报 UB 超限、tiling 越界 | 定义 / 编译 | tiling 参数、`InitBuffer` 总量、块大小 |
| 能跑但结果错 | 定义 / 调用 | 对齐、尾巴、`EnQue` / `DeQue`、tiling 边界 |
| 模型能跑但慢，profiler 看到低效 kernel | 注册（vLLM / PyTorch 侧）或实现本身 | 类有没有被 `register_oot` 替换、custom ops 是否被禁用、kernel 是不是本来就慢 |

这张表不是故障大全，是归因的起点：先确定哪一环，再往那一环的日志与文件里找。归错环的代价是把时间花在重装环境或改模型上。

**能判断该不该自己写。** 先用内置算子与图优化，再拿 profiler 找热点；一条经验规则是，热点算子占前向时间超过 10%，且现有实现明显低效或回退，才进入“自己写”的候选。阈值不是定律，要按你的模型与硬件重算。写之前先问三个问题：语义能不能用 AscendC 的 API 表达（归约、转置、非对齐访存是不是都有原语）；边界清不清楚（dtype、shape、对齐、尾巴、workspace 需求）；有没有可对比的基线（同输入的 `torch_npu` 内置实现或统一算子层实现）。三个问题有一个答不上来，先别写。

收益上界可以用 Amdahl 定律算：若算子占前向时间 p，自定义实现把这一段加速 s 倍，端到端加速比是 1 / ((1 − p) + p / s)。下表是几个组合：

| p \ s | 2× | 5× | 10× |
|---:|---:|---:|---:|
| 5% | 1.026 | 1.042 | 1.047 |
| 20% | 1.111 | 1.190 | 1.220 |
| 50% | 1.333 | 1.667 | 1.818 |

这张表说明两件事：占比小的算子，写得再好也改不动端到端；占比大的算子，s 从 5 提到 10 的边际收益远小于从 2 提到 5。所以顺序是先用统一算子层或内置实现把覆盖补齐，再对真正的热点写原生算子，不是一上来就手写。第 24 章在 GPU 上做过同样的事，第 29 章会把一个 Triton kernel 沿这条链搬到 Ascend。

不必会的东西也要列清楚：不必背 AscendC 的 API 清单，不必从零写一遍完整的 tiling 框架，不必读驱动、固件与图编译器的内部实现。这些查文档与源码就够。需要会的是读代码、算账、定位与判断，这四样不随版本过期。

诚实标注：本章没有在实机上编译或运行任何 AscendC 算子，`add_custom` 骨架与链路表都是按公开文档与源码整理的；“编译通过”“kernel 真的跑了”“慢多少倍”这三件事都要等第 27 章装好环境后补测。待补测清单：编译一次最小算子并记录报错；用 `msprof` 看一次 kernel 的实际耗时；对比内置 RMSNorm 与自定义融合实现的端到端差异；核对当前版本的 `_cann_ops_custom/` 布局与本章描述是否一致。每项后面写清下一步去哪里补，比只写“待补测”有用。

本章的结论收成三条。CANN 的分层用来把报错翻译成位置，算子开发横跨编译层与接口层，不是单独一层。AscendC 的代价是把 TBE 编译器替你做的调度、同步、内存分配接过来，读最小算子就是读这三件事。一个自定义算子要在 vLLM 里生效，得在 CANN/OPP、PyTorch、vLLM 三处都登记；少一处，症状会出现在别处。够用的标准是：拿到一个算子或一条报错，能说出它在四环的哪一环、下一步查哪个文件或命令，并算得出写它的收益上界。

**延伸**：[CANN 官方文档](https://www.hiascend.com/document)（软件架构、Ascend C 算子开发指南、TBE & AI CPU 算子开发指南的入口）、[Ascend C 静态 Tensor 编程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)（TPipe、TQue、double buffer 的官方说明）、[vllm-ascend](https://github.com/vllm-project/vllm-ascend)（插件目录、自定义算子与版本矩阵）、[vLLM `custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)（`register_oot` 与平台分发）、[CANN 总体架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/quickstart/quickstart/quickstart_18_0003.html)（五层划分的原文）。

## 第 27 章 · 同一模型在 Ascend 上跑通 + vllm-ascend 插件架构

**本章目标。** 装出一条能跑 vLLM 的 Ascend 环境并留下带版本号的安装记录；用同一模型、同一 prompt、同一输出长度完成非流式与流式两次冒烟；读清 vllm-ascend 从入口点到算子实现的分发链；把 Ascend 与 NVIDIA 的部署差异整理成一张“先查什么”的对照表。

同一份模型权重从 NVIDIA 机器搬到 Ascend 机器上，`vllm serve` 这条命令几乎不用改；变的是启动前装的那条链、启动后日志里出现的平台名，以及出问题时该看哪一层。第一次部署最先卡住的通常不是模型，而是版本：CANN、TorchNPU（PyTorch 的昇腾适配层）、PyTorch 与 vllm-ascend 必须从同一张兼容表里选，任何一项单独换版本，都可能让插件加载失败或算子走错实现。版本对齐之后，第二个卡点是定位顺序：卡认不出、框架认不出、引擎认不出，三种失败都写在启动日志里，根因却分属三层。

### 环境：CANN + torch_npu + vllm-ascend 的安装链

写作时钉下的版本锚点是：Python 3.12、CANN 9.2.1、PyTorch 2.10.0、TorchNPU 2.10.0.post7、vllm-ascend v0.30.0（2026 年 10 月发布）。这是 vllm-ascend v0.30.0 安装文档标为“一起验证”的一套组合，不是永久有效的清单。动手那天先打开 vllm-ascend 仓库 README 与对应版本的 installation 页，核对当天写明的版本表，必要时再对照 CANN、PyTorch、TorchNPU 各自的发布说明，然后按那张表装；本章后面的安装命令、环境变量与启动参数都对应这一版，版本一变，写法以当前版本文档为准。安装记录留 6 项：上面 5 个版本，加驱动与固件（同一张兼容表里写作 Ascend HDK 26.1.1），每项后面写来源与日期。

`npu-smi` 的输出列、`torch.npu` 的设备接口与 `vllm serve` 的参数都随版本变化；下面出现的命令按 v0.30.0 书写，执行前以当前版本文档为准。

叫“链”而不是“清单”，因为每一环都要调用下一环的头文件、库或 Python 包。CANN 是底座；TorchNPU 站在 CANN 上，把 PyTorch 的张量与算子接到 NPU；vllm-ascend 站在 TorchNPU 上，向 vLLM 注册一个 NPU 平台；vLLM 在最上面，模型代码、调度与 KV 缓存管理不因芯片改变。顺序装反不会得到一句“顺序错了”：CANN 没装好，TorchNPU 的包装上了也 import 不进来；TorchNPU 与 PyTorch 版本不配对，vllm-ascend 启动时可能连 `torch.npu` 都找不到；插件版本不对，vLLM 可能能启动，但日志里没有平台插件行。症状停在哪一环，就从哪一环往上查。

安装路径有两条，选一条走完，不要混着抄。第一条是官方镜像：镜像里已经装好 CANN、NNAL（昇腾的神经网络加速库，提供 ATB 运行时等组件）、PyTorch/TorchNPU、vLLM 与 vllm-ascend，宿主机只需要可用的驱动、固件与 Docker。A2 单卡的拉取与启动如下：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0
docker pull "$IMAGE"

export DEVICE=/dev/davinci0
export MODEL_CACHE="${HOME}/.cache"
mkdir -p "$MODEL_CACHE"
docker run --rm --name vllm-ascend --shm-size=1g \
  --device "$DEVICE" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$MODEL_CACHE:/root/.cache" \
  -p 8000:8000 \
  -it "$IMAGE" bash
```

A3 是双 DIE（双裸片）设计，要挂两个 `/dev/davinci*` 节点；300I Duo、200I Pro 与 950DT 的设备节点、驱动路径和挂载项各不相同。这些差异照 installation 页对应的标签页抄，不要拿 A2 的命令改一个标签就当通用命令。第二条路径是在已有 CANN 环境里安装：按 installation 页的“CANN environment”一栏执行，PyTorch 与 TorchNPU 由该路径一并安装，vLLM 与 vllm-ascend 的版本号必须相同。两条路径共用一张版本表；镜像标签、pip 包与 CANN 版本对不上时，先回到 README 的兼容表，不要用“装最新版”来解决。

装完先做三层验证，顺序从下往上。第一层是驱动与固件：`npu-smi info` 能列出预期的 NPU，Health 为 OK，没有驱动初始化错误。`npu-smi` 的输出列与 `torch.npu` 的设备名随驱动和 TorchNPU 版本变化，下面的命令按 v0.30.0 的环境书写，其他版本以对应文档为准。设备名、芯片数与列名随硬件和驱动版本变化，缺设备或 Health 不是 OK 时，先停下查驱动与固件，不要继续往上装。第二层是框架：`torch_npu` 能 import，`torch.npu.is_available()` 为真，能在 `npu:0` 上建张量并做一次运算。第三层是插件：vLLM 启动后，日志里出现平台插件被加载与激活的行。

```bash
npu-smi info
python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_ascend

assert torch.npu.is_available(), "No available Ascend NPU detected"
x = torch.randn(2, 2, device="npu:0")
print(x.device, torch.npu.device_count())
PY
```

第三层没有单独的 import 检查：`import vllm_ascend` 成功只说明包装上了，平台有没有被 vLLM 选中要看启动日志。v0.30.0 的 quick start 给出的日志样例里有 `Available plugins for group vllm.platform_plugins`、`- ascend -> vllm_ascend:register` 与 `Platform plugin ascend is activated`；行号与措辞随版本变化，认三件事：插件组被列出、ascend 条目被加载、平台被激活。没有 ascend 条目，先查 `pip show vllm-ascend` 与 vLLM 版本是否相同；有条目但没有激活，查 `VLLM_PLUGINS` 是不是把插件过滤掉了——这个环境变量用来限制加载哪几个插件，机器上同时装了多家平台插件时尤其要查。

三层验证每层只有过与不过，组合起来是 2³ = 8 种状态，只有一种全过。8 种状态的用处是定位，不是打分：`npu-smi` 过、`torch.npu` 不过，问题在驱动与框架之间，常见原因是 CANN 的环境脚本没进当前 shell，或 TorchNPU 与 PyTorch 的版本不配对；`torch.npu` 过、插件层不过，问题在插件，常见原因是 vllm-ascend 与 vLLM 版本不一致或插件被 `VLLM_PLUGINS` 过滤。安装记录里把三层的“过/不过”和“失败指向哪一层”各写一行，排查就有了起点。

动手那天，README 要求的版本与上面的锚点不一致是正常的。做法是跟 README 走，同时记两组数：锚点是什么、实际装了什么、为什么跟 README 走。锚点是写作时的快照，README 是安装时的当下；第 30 章写报告时引用实际装的版本，并把锚点作为“出发时的计划”一并保留。版本漂移本身不构成错误，记录里只剩一组数、看不出哪组是计划哪组是实际，才会让复现的人卡住。

```bash
pip show vllm vllm-ascend torch torch_npu | grep -E '^(Name|Version):'
```

两个常见误解。第一，“TorchNPU 装好了就等于 vLLM 能跑”：TorchNPU 只让 PyTorch 认 NPU，vLLM 认不认平台由 vllm-ascend 决定，上面 8 种状态里就有前两层全过、第三层不过的一种。第二，“版本差不多就行”：PyTorch 与 TorchNPU、TorchNPU 与 CANN、vllm-ascend 与 vLLM 都是成对约束，单换一个组件得到的是“看起来更新”的组合，不是兼容的组合。

没有实机时，HiDevLab 在线开发平台是官方给的 Ascend 试用入口（README 的 Accessing Ascend NPU 一节）；能用到哪个型号、额度与排队情况以平台当天说明为准，不要按印象写。无卡时能做完的是：读 README 与 installation 页、拉镜像、核对设备节点与挂载项、把版本记录表建好。启动推理与三层验证的第三层，等有卡再补。

### 部署与冒烟：同一模型在 Ascend 上跑起来

先算模型体量的账，再决定拿哪个模型冒烟。Qwen3-0.6B 是 6 亿参数，FP16 权重占 6×10⁸ × 2 字节 = 1.2×10⁹ 字节，约 1.12 GiB；7B 模型是 7×10⁹ × 2 = 1.4×10¹⁰ 字节，约 13.0 GiB；32B 模型是 32×10⁹ × 2 = 6.4×10¹⁰ 字节，约 59.6 GiB。三笔都是算例，只算权重，不含 KV 缓存、激活与框架开销，真实占用只会更大。0.6B 的量级把“显存装不下”这个变量先排除，适合第一次冒烟；跨芯片对比要换回第二部分在 NVIDIA 上跑过的同一个模型，权重按同一公式重算。模型换了，输出不同时分不清是芯片差异还是模型差异；prompt 或输出长度换了，耗时也没有共同的分母。

启动方式按 vllm-ascend 对应版本的 quick start。`VLLM_WORKER_MULTIPROC_METHOD=spawn` 是官方示例给的多进程启动方式，它在当前版本上是否必须、为什么，以当前版本文档为准；模型名与其余参数按你的环境改。

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen3-0.6B --dtype float16 --max-num-seqs 8
```

`--dtype`、`--max-num-seqs` 是 vLLM 的通用参数，NPU 路径上是否全部生效、默认值是多少，同样以当前版本文档为准。启动后先记四类日志行：平台插件行、模型加载完成行、图捕获或编译行（开了图模式才有）、显存统计行。日志格式每个版本不同，记原文，不记转述。

冒烟分两步：先非流式确认能出结果，再用同一条件发流式请求，看 token 是不是一个个出来。同一个 prompt、同一个 `max_tokens`、`temperature` 固定为 0，两步只改 `stream`。

```bash
curl -sS http://localhost:8000/v1/models

curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Beijing is a","max_tokens":32,"temperature":0}' \
  -w '\ntotal=%{time_total}s\n'

curl -sS -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Beijing is a","max_tokens":32,"temperature":0,"stream":true}'
```

非流式请求的 `time_total` 是一次请求的总耗时，不是首 token 延迟。TTFT（Time to First Token，首 token 延迟：从发请求到收到第一个输出 token 的时间）与 ITL（Inter-Token Latency，token 间延迟：相邻输出 token 的平均间隔）要从流式响应的每个数据块到达时间算；`curl -N` 关掉客户端缓冲，数据块才会一个个到，不加 `-N` 时 curl 可能攒够一块才输出，看起来像非流式。流式响应最后的 `[DONE]` 是结束标记，不是 token，不要把它算进 ITL。

要数字，用第 4 章的 `scripts/bench_openai.py`：它按流式 chat completions 计时，逐请求记 TTFT、总耗时、输出 token 数与 ITL。字段含义见第 5 章；命令里的参数按你的服务与模型改，只把 `--base-url` 指向 Ascend 服务。

```bash
python3 scripts/bench_openai.py --base-url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B --max-tokens 32 --concurrency 1 --requests 1
```

一次请求的样本量是 1，能证明链路通，不能下性能结论。TTFT、ITL、吞吐要等第 30 章按三级并发、每级多次测量来跑；脚本读不到 usage 时会按字符估 token 数，记录里要注明 `tokens_estimated` 是 true 还是 false。

显存记两处。服务启动前后各跑一次 `npu-smi info`，抄 HBM-Usage；如果能在服务进程内取数，再用 `torch.npu.max_memory_allocated()` 取框架侧峰值。两个数口径不同：`npu-smi` 是设备侧占用，包含同一张卡上所有进程；框架侧只算 PyTorch 分配器。记录里写清用的是哪一个，不要把两个数混成一个“显存占用”。

结果对不对，先与 NVIDIA 上同一模型、同一 prompt 的输出对照。`temperature=0` 下同一模型两次运行的输出也未必逐字相同，判据是“同一个模型在正常工作”：答案语义一致、没有大段乱码、没有提前截断、没有答非所问。逐字不同不算失败；大段乱码、反复重复同一个词、输出与 prompt 完全无关，先停下来查，不要继续压测。正式的精度回归在第 30 章，这一步只负责拦住最粗的错误。

冒烟记录一张表：日期、硬件型号、HDK/CANN/PyTorch/TorchNPU/vLLM/vllm-ascend 版本、模型、prompt、`max_tokens`、实际输出 token 数、是否流式、TTFT、ITL、吞吐、总耗时、HBM-Usage。`max_tokens` 是上限，模型可能因 EOS（End of Sequence，序列结束符）提前结束；实际输出长度从响应的 usage 字段读，比较两边的数时按实际长度对齐。没有卡就把表头建好，数值写“待补测”，旁边写触发条件：拿到什么型号的卡、装好哪一组版本之后，用同一张表补测。只写“待补测”不写触发条件，等于把补测留给了没有上下文的人。

两个常见误解。第一，“冒烟通过就等于性能达标”：冒烟只回答能不能出结果，性能要到第 30 章的三级并发压测才回答；一次调用的耗时连“快不快”的边都沾不上，它只是“通没通”的证据。第二，“输出看起来通顺就等于数值对了”：肉眼对照拦不住细粒度偏差，某个算子回退到低精度实现，输出照样通顺；那要等第 30 章的精度回归拿评测集去量。

### vllm-ascend 插件架构：插件怎么 hook 进 vLLM、算子分发发生在哪

把第 25 章的五层软件栈图翻出来，vllm-ascend 坐在“框架插件”那一层：向上要实现 vLLM 的平台接口，向下要调用 CANN 与算子实现。图 27-1 与第 25 章共用，画的就是这个位置。一个独立安装的 Python 包，要让 vLLM 在启动时认出它、把工作进程、模型执行器与注意力后端都换成自己的实现，靠两段机制：Python 入口点（entry point，打包元数据里声明的一类扩展点）负责被发现，平台类负责把其余 hook（挂接点，引擎在固定位置调用插件实现的接口）接上。

入口点在 `setup.py` 里声明（v0.30.0）：`vllm.platform_plugins` 组里有一条 `ascend = vllm_ascend:register`，vLLM 启动时遍历这个组，调用 `vllm_ascend.register()`，这个函数返回字符串 `vllm_ascend.platform.NPUPlatform`，vLLM 据此 import 平台类。同一份 `setup.py` 还声明了 `vllm.general_plugins` 组的四条（KV connector、model loader、service profiling、model 注册），这些插件在 vLLM 的各个进程里加载，负责模型、连接器与服务 profiling 一类的扩展。平台插件决定“用哪个平台”，通用插件补充“平台之外还要注册什么”，两组不要混。排障时这两组有直接用处：日志里的 `Available plugins for group vllm.platform_plugins` 列出被发现的平台插件；`VLLM_PLUGINS` 按名字限制加载哪些插件，机器上同时装了多家平台插件时，用它指到 ascend 可以排除别的插件干扰。以上名称取自 v0.30.0 的 `setup.py` 与 `vllm_ascend/__init__.py`，换版本先按这两处核对，以当前源码为准。

`NPUPlatform` 继承 vLLM 的 `Platform`，类属性已经交代了它是什么：`_enum = PlatformEnum.OOT`（树外平台）、`device_type = "npu"`、`device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"`（设备可见性由这个环境变量控制）、`dispatch_key = "PrivateUse1"`（PyTorch 私有后端的分发键）。vLLM 核心通过这些属性判断当前平台、决定张量落在哪个设备、读哪个环境变量。平台类的方法把其余组件逐个接上；下表是 v0.30.0 的落点，路径与类名以当前源码为准。

| hook | vLLM 侧接口 | vllm-ascend 落点（v0.30.0） | 作用 |
|---|---|---|---|
| 平台 | `vllm.platform_plugins` 入口点、`Platform` 基类 | `platform.py` 的 `NPUPlatform` | 设备识别、平台选择、设备可见性 |
| 工作进程 | `WorkerBase` | `worker/worker.py` 的 `NPUWorker` | 初始化设备、加载模型、执行前向 |
| 模型执行器 | vLLM 的 model runner（v1 为 `GPUModelRunner`） | `worker/model_runner_v1.py` 的 `NPUModelRunner` | 组 batch、准备输入、管理 KV 缓存、图捕获 |
| 注意力后端 | `AttentionBackend` | `attention/attention_v1.py` 的 `AscendAttentionBackend`；MLA 走 `mla_v1.AscendMLABackend`，FA3 走 `fa3_v1.AscendFABackend` | attention 的计算与 KV 布局 |
| 通信 | `CommunicatorBase` | `distributed/device_communicators/npu_communicator.py` 的 `NPUCommunicator` | 多卡集合通信 |
| 编译与图 | `get_compile_backend`、`get_static_graph_wrapper_cls` | `compilation/compiler_interface.AscendCompiler`、`compilation/acl_graph.ACLGraphWrapper` | 图编译与图捕获 |
| 算子 | `CustomOp.register_oot` | `utils.register_ascend_customop()` 按类名登记；例如 `ops/layernorm.py` 的 `AscendRMSNorm` | 把 vLLM 的算子类换成 Ascend 实现 |

平台类不自己实现所有东西，它回答的是 vLLM 的“你要用哪个类”：`get_attn_backend_cls()` 按 MLA、稀疏、压缩等条件返回不同的后端类路径，`get_device_communicator_cls()` 返回通信类，`get_compile_backend()` 返回编译后端；worker 类由 `device/hardware_profile.py` 里的 `default_worker_cls` 写进 `parallel_config.worker_cls`。平台选定之后，这些选择跟着定下来，模型代码里没有一个 `if is_ascend()` 分支。

`patch/` 不是 hook，是 hook 不到的地方的补丁。v0.30.0 的 `patch/__init__.py` 把补丁分两组：platform 组在 worker 启动前、由 `NPUPlatform.pre_register_and_update()` 调用 `adapt_patch(is_global_patch=True)` 打上；worker 组在每个 worker 的 `__init__` 里调用 `adapt_patch(is_global_patch=False)` 打上。读插件代码时看到某个 vLLM 类的行为被改了、却找不到注册点，先到 `patch/` 里按类名搜，多半在那里。

算子级分发是另一条链，落点不在平台类的方法里，而在 vLLM 的 `CustomOp`（vLLM 的算子替换基类，按类名支持树外实现接管）。模型代码构造 `RMSNorm(...)`；`RMSNorm` 是 `CustomOp` 的子类，`CustomOp.__new__` 先按类名 `"RMSNorm"` 查 `op_registry_oot`，查到就不再实例化 `RMSNorm` 本身，而是实例化登记的子类。vllm-ascend 在 `register_ascend_customop()` 里把 `"RMSNorm": AscendRMSNorm`、`"SiluAndMul": AscendSiluAndMul` 等一组映射逐条交给 `CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)`，于是 `op_registry_oot["RMSNorm"] = AscendRMSNorm`。`CustomOp.__init__` 再调用 `dispatch_forward()`；当前平台是 `PlatformEnum.OOT`，`is_out_of_tree()` 为真，`forward` 绑到 `forward_oot`；`AscendRMSNorm.forward_oot` 调用 `torch_npu.npu_rms_norm(...)` 或 `torch.ops._C_ascend.*` 的自定义算子，这才落到芯片实现。

这条链解释了“算子名不变、实现换一份”是怎么做到的：模型代码里的类名是键，`op_registry_oot` 是表，`forward_oot` 是值的入口。三步在 vLLM 的 `vllm/model_executor/custom_op.py` 与 vllm-ascend 的 `utils.py`、`ops/layernorm.py` 里能逐行对上；换版本时以这几处源码为准。它也解释了回退：某个算子没有登记到 `op_registry_oot`，`CustomOp` 基类的 `forward_oot` 默认调用 `forward_native`，它就在 NPU 上跑 PyTorch 原生实现——能出结果，但一个算子一个算子地跑，慢；`forward_native` 也没有实现的算子会直接报错。第 28 章的覆盖矩阵查的就是这张登记表与 `forward_native` 的覆盖，而不是只查 CANN 有没有这个算子。

注意力后端是平台级分发，和算子级分发分开看。`get_attn_backend_cls()` 根据选择器里的 `use_mla`、`use_sparse`、`use_compress`、`use_pcp` 返回不同类路径：dense 在非 FA3 条件下走 `AscendAttentionBackend`，MLA 走 `AscendMLABackend`，稀疏走 `AscendSFABackend`，满足 FA3 条件时走 `AscendFABackend`。这些类在 `attention/` 目录下，负责 attention 的计算与 KV 布局；`ops/` 里的算子则是更细粒度的逐算子实现。读代码时先分清目录分工：找“attention 怎么算”去 `attention/`，找“某个逐元素算子在哪”去 `ops/`。

动手定位一个算子，选 RMSNorm，因为它在 24.3 节作为 CustomOp 的可替换面出现过，这次看它的芯片侧版本。记录四站：模型代码里构造 `RMSNorm` 的位置（Qwen3 的 decoder layer 里 `input_layernorm` 与 `post_attention_layernorm` 都是它）、`CustomOp.__new__` 的查表、`utils.register_ascend_customop()` 里的 `"RMSNorm"` 映射、`ops/layernorm.py` 的 `AscendRMSNorm.forward_oot`。四站里任何一站只写“大概在附近”，定位就不算完成；文件路径、类名、方法名都要抄下来，并写上读的是哪个 tag。没有卡的读者做到这一步就够了：这条链不依赖运行，只依赖源码。

把插件想成薄翻译层会读错两件事。第一，它不是只转接算子调用：工作进程、模型执行器、注意力后端、通信、编译后端都有一份 Ascend 实现，任何一处没接上，服务都起不来或跑不对。第二，`attention/` 不是 attention 算子的全部：它装的是后端选择与后端实现，逐算子的实现还在 `ops/` 与 `_cann_ops_custom/`。目录名是入口，不是结论。

### 踩坑记录：把每个问题和解法写下来

每条按四列记：现象、定位、处理、是否可复现。再加两项固定信息：版本组合与日期。现象写原文或可观察到的行为，不写“报错了”；定位写它落在哪一层（驱动与固件、SDK、框架适配、算子库、插件、引擎）；处理写命令或改动；是否可复现写“是”“条件复现”或“否”，条件复现要写清条件（换版本、换型号、开图模式）。没有实机时，下表是待对账的常见坑位，不是已经踩过的记录；实测遇到哪条，把现象换成原文，把“待实测”改掉。显存不足（OOM，Out Of Memory）单独占一行，因为它的处理动作与别的坑不同。这里只写症状类型，不抄具体报错码；同一个症状在不同版本上的原文可能不同，以当前版本为准。

| 现象 | 定位 | 处理 | 是否可复现 |
|---|---|---|---|
| `npu-smi info` 列不出设备，或 Health 不是 OK；容器里看不到卡 | 驱动与固件层 | 核对 HDK 版本；容器启动时把 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 与驱动库挂进去 | 条件复现：换机器、换 HDK 或容器漏挂载时出现 |
| `source set_env.sh` 之后仍加载不到 CANN 的库或头文件，编译或启动报 SDK 组件缺失 | SDK 层 | 确认 CANN Toolkit、Ops 与 NNAL 都装了且版本一致；确认 source 的是当前安装路径下的脚本 | 是：版本或路径不对就复现 |
| `import torch_npu` 失败，或 `torch.npu.is_available()` 为假 | 框架适配层 | 按整组换 PyTorch 2.10.0 与 TorchNPU 2.10.0.post7；不要只升级其中一个；用 `pip show` 核对 | 是：两个包版本不配对就复现 |
| vLLM 能启动，但日志里没有 ascend 平台插件的条目，或平台未被激活 | 插件层 | 用 `pip show vllm vllm-ascend` 对齐版本；检查 `VLLM_PLUGINS` 是否过滤掉了插件；确认入口点在 `setup.py` 里 | 是：插件版本与 vLLM 不一致或插件被过滤时出现 |
| 开图模式后启动卡在捕获阶段，或输出异常、精度下降 | 编译与图执行层 | 先关图跑 eager，确认模型与算子正确；再按 graph mode 文档逐项打开 `cudagraph_mode`、`enable_npugraph_ex` 一类的开关 | 条件复现：取决于模型、输入形状与版本 |
| 服务能跑但某段明显慢，profile 里出现 PyTorch 原生算子 | 算子库与插件层 | 查第 28 章的覆盖清单；确认该算子是否在 `register_ascend_customop()` 的映射里；没有映射就换实现，或接受回退并记下它的耗时占比 | 是：同一模型、同一算子重复出现 |
| 启动加载权重时 HBM 不足，或并发上来后 KV 缓存分配失败 | 显存墙、驱动与插件 | 先降 `--max-model-len` 与 `--max-num-seqs`，再评估量化或张量并行；`--gpu-memory-utilization` 在 NPU 路径上的行为以当前版本文档为准 | 条件复现：取决于模型、输入长度与并发 |

四列的用途各不相同：没有现象原文，后来的人对不上自己的报错；没有定位，排查没有起点；没有处理，清单只是问题集；没有是否可复现，读的人不知道这条能不能照抄。版本与日期同样不能省：同一个现象在 CANN 9.2.1 与旧版本上可能根因不同，记录不写版本，四列就失去前提。

排查顺序沿用 27.1 的三层验证往回退：插件层失败，退到框架层；框架层失败，退到驱动层；驱动层过了、框架层不过，问题在两层之间（CANN 环境变量、设备权限、TorchNPU 与 PyTorch 的配对）。版本类问题先查 `pip show` 的四个包，再对照兼容表；不要一上来重装整条链，重装会同时改掉多个变量，修好了也不知道是哪一步修好的。

没有实机时，把上表当“预期坑位”清单：每条先写推断的现象与定位，标“待实测”。有卡之后逐条对账：遇到了，补原文与处理；没遇到，写“本次未遇到”并保留——没遇到也是信息，说明这条路在当前版本组合上是通的。不要把推断写成遇到过，也不要把没遇到悄悄删掉。

### 小结：Ascend 与 NVIDIA 的部署差异对照表

把第二部分在 NVIDIA 上的部署流程与本章逐项并排，差异集中在模型下面那几层。下表五列：维度、NVIDIA、Ascend、差异在哪、先查什么。最后一列是这张表的用法：不是判谁更好，是出问题时按顺序查。表里的 ACLGraph 是 vllm-ascend 的图捕获封装，NPUGraph Ex 是它的图执行优化开关，两者的能力与默认值以当前版本的 graph mode 文档为准。

| 维度 | NVIDIA（第二部分的流程） | Ascend（本章的流程） | 差异在哪 | 先查什么 |
|---|---|---|---|---|
| 安装链 | 驱动 + CUDA/cuDNN → PyTorch → vLLM（原生支持） | HDK + CANN/NNAL → PyTorch + TorchNPU → vLLM + vllm-ascend | 多两层，版本组合从四个组件变成七个，升级单位是整组 | 安装记录里的 6 项版本与 vllm-ascend README 的兼容表 |
| 启动命令 | `vllm serve <model>`，`CUDA_VISIBLE_DEVICES` 控设备 | 同一条命令，另加 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，`ASCEND_RT_VISIBLE_DEVICES` 控设备，平台插件自动加载 | 命令相近，设备可见性变量与多进程启动方式不同 | 启动日志里的平台插件行与 `npu-smi info` |
| 显存 | `nvidia-smi` 的已用显存；`torch.cuda.max_memory_allocated()` | `npu-smi info` 的 HBM-Usage；`torch.npu.max_memory_allocated()` | 工具与统计口径不同，权重与 KV 的算术相同 | 权重 + KV 的账，再看 HBM-Usage 与框架侧峰值 |
| 图模式 | CUDA Graph，按 `--compilation-config` 的 `cudagraph_mode` 等开关配置 | ACLGraph 与 NPUGraph Ex，按 `--compilation-config`、`--additional-config` 的对应开关配置 | 捕获与重放的实现不同，可用开关与默认值不同 | 当前版本的 graph mode 文档；先 eager 跑通，再开图 |
| 算子覆盖 | CUDA/Triton 实现 + CustomOp 替换 | CANN/AscendC 实现 + `register_ascend_customop()` 替换 | 未覆盖时，NVIDIA 侧通常还有 CUDA 或 Triton 实现可走；Ascend 侧走 `forward_native` 或直接报错 | 第 28 章的覆盖矩阵，再用 profiler 找热点算子 |
| 调试工具 | `nvidia-smi`、`ncu`、`nsys`、PyTorch Profiler | `npu-smi`、msprof/msprobe、service profiling、PyTorch Profiler | 工具名、采集方式与指标名不同，能采到的项也不同 | 当前版本的 performance 与 debug 文档，确认支持哪个工具 |
| 版本兼容 | CUDA、cuDNN、PyTorch、vLLM 之间的对应 | HDK、CANN、NNAL、PyTorch、TorchNPU、vLLM、vllm-ascend 之间的对应 | 组件更多，必须按兼容表整组换 | vllm-ascend README 的兼容表与 release notes |

结论：差异不在模型，在模型下面的栈。同一份模型代码、同一个 prompt、同一个输出长度，两边都能跑；换掉的是安装链、平台插件、图模式开关、算子实现与调试工具。把差异读成“谁更好”没有意义：两边的约束数量与工具链不同，NVIDIA 侧要对的版本少，Ascend 侧多两层，但把与芯片有关的选择都收进了插件，模型代码不用改。判断一次部署卡在哪，按这个顺序查：版本组合 → 平台识别 → 图模式 → 算子回退 → 性能。前两步不过，后面的数字都不用看。

这张表里依赖实测的格子——显存占用、TTFT/ITL、图模式收益、算子回退的耗时占比——没有卡就写“待补测”，并在备注里写触发条件。已经能确定的是流程与检查点：版本记录 6 项、三层验证 3 条结果、冒烟记录 1 张表、接口落点表 7 行、差异表 7 行。第 28 章接着这张表往下走：拿 profiler 导出的真实算子清单，逐条查它们在 Ascend 上由谁承接，把“算子覆盖”那一行从定性变成清单。

**延伸**：[vllm-ascend 仓库](https://github.com/vllm-project/vllm-ascend)（README 的版本与硬件要求、`setup.py` 的入口点）、[vLLM Ascend v0.30.0 安装文档](https://docs.vllm.ai/projects/ascend/en/v0.30.0/installation.html)、[vLLM Ascend v0.30.0 Quick Start](https://docs.vllm.ai/projects/ascend/en/v0.30.0/quick_start.html)、[Introducing vLLM Hardware Plugin](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)（Platform、Worker、ModelRunner、AttentionBackend、Communicator 的插件化背景）。

## 第 28 章 · 算子覆盖分析

> 硬件：第 1 节需要能跑 profiling 的环境（NVIDIA 卡即可）；第 2–5 节的对照工作无卡可做；芯片侧的核实需要 Ascend 环境或官方文档，核实不了的项标“待核实”。
>
> 本周主线：先拿一张真实的算子清单，再逐条问每个算子在 Ascend 上由谁承接——清单是事实，对照是判断，瓶颈归因是清单与对照相乘之后才配下的结论。

### 第 1 节 · 导出算子清单：用 PyTorch Profiler 导出一次推理的全部算子

**目标**：拿到一份真实的算子清单——一次推理到底调了哪些算子、各占多少时间。

**怎么做**（60 分钟，需 NVIDIA 卡；无卡时先写 profiler 脚本，实测标注“待补测”）：

先把这一节的定位说死：只导出，不分析。导出是一次人口普查（census），分析是普查之后才做的事。普查的价值在于完整——一个算子再小，也要出现在清单上；分析的诱惑在于提前下判断——看到某个算子占了 40% 的时间，手就痒，想立刻说它是瓶颈。这一节把手按住。清单不完整，后面所有的对照都会漏项；清单一旦漏了长尾里某个算子，第 2 节的矩阵就永远少一行，而少的那一行恰好是回退项的概率，并不比任何一行低。先把清单拿全，再谈哪一行重要。

方法复用 M0 第 3 节的 profiling 手法，对 M1 里跑通过的模型做一次前向。复用是有原因的：M0 第 3 节已经解决过“怎么采、采什么、怎么读表”这些基本问题，本周不需要发明第二套 profiling 方法，需要的是把同一套方法对准一个新的问题——一次推理的算子构成。模型沿用 M1 里跑通过的那个，同样有原因：清单的可信度取决于模型本身能不能跑通，一个还没跑通的模型采出来的清单，里面混着报错重试与异常路径，读它等于读一份被污染的样本。先跑通，再采样，顺序不能反。

采之前要先定三个口径，口径不定，清单上的数字就没法互相比较。第一个口径是采哪一次前向：热身（warmup）跑完之后、进入稳定状态的那一次。第一次前向里混着缓存预热与惰性初始化的开销，把它采进来，清单反映的就是“第一次跑”而不是“平时跑”。热身几遍再采，采到的才是稳定态的前向。第二个口径是形状：这一次前向的 batch 与序列长度是多少。Prefill 形状与 decode 形状的算子构成并不相同——序列很长时矩阵乘占大头，单 token 解码时矩阵乘的占比下来了，逐元素算子的相对占比反而上去。清单必须把形状写在表头上，一份没有形状的清单是没有语境的数字，下周换个形状再采，Top 10 很可能换一批名字。第三个口径是按什么时间排序：按 CUDA 核函数（kernel）时间排，还是按 CPU 端的算子时间排。推理延迟由 GPU 端的执行时间决定，排序的依据就取 CUDA 时间；CPU 时间留着排查 launch 开销时再看。两个时间口径排出来的 Top 10 可能不一样，选哪一个要先说清楚，而不是采完了再挑好看的那个。

Profiler 导出的一行里至少有四列：算子名、调用次数、累计耗时、占比。算子名是 PyTorch 层面的名字，例如 `aten::mm`、`aten::bmm`；调用次数是这次前向里它被调了多少次；累计耗时是所有调用加起来的时间；占比是累计耗时除以总时间。还有一列容易被忽略：自耗时（self time）与总耗时（total time）的区别。一个算子如果内部又调了别的算子，总耗时会把子算子的时间也算进来，自耗时只算它自己那一层。排序的时候用自耗时还是总耗时，结果会不一样——一个总耗时很大的算子，可能只是个空壳，时间全花在它调的子算子身上。这一节的清单按自耗时口径取数，理由只有一个：后面第 2 节要逐条对照“这个算子由谁承接”，对照的对象是算子本身，不是它背后的调用树。口径写在清单的表头上，免得下周有人拿总耗时重排一遍，得出另一份 Top 10，还以为清单错了。

拿一组算例把这四列的关系算清楚。假设某次前向的总时间是 50.0 毫秒，耗时前十的算子分别是 21.0、9.0、4.5、3.5、2.5、2.0、1.5、1.0、0.6、0.4 毫秒。先加总：21.0 加 9.0 是 30.0，再加 4.5 是 34.5，再加 3.5 是 38.0，再加 2.5 是 40.5，再加 2.0 是 42.5，再加 1.5 是 44.0，再加 1.0 是 45.0，再加 0.6 是 45.6，再加 0.4 是 46.0 毫秒。Top 10 合计 46.0 毫秒，占总时间 50.0 毫秒的 92.0%。剩下的长尾一共 4.0 毫秒，占 8.0%。占比那一列逐项算出来是 42.0%、18.0%、9.0%、7.0%、5.0%、4.0%、3.0%、2.0%、1.2%、0.8%，十个数加起来正好是 92.0%，与合计对得上。Top 3 是 21.0 加 9.0 加 4.5 等于 34.5 毫秒，占 69.0%——前三名就占了近七成，这组数里清单的头重脚轻非常明显。这组数是算例，不是实测，作用是让你在拿到真实清单的第一时间就能验算：Top 10 加起来对不对得上它声称的占比，长尾还剩多少，有没有哪一行的时间与占比互相矛盾。验算不过的清单，先修清单，再往下走。

调用次数那一列单独看会骗人。同一组算例里，排第一的矩阵乘被调了 48 次，累计 21.0 毫秒，平均每次 0.4375 毫秒；排第六的某个逐元素算子被调了 1000 次，累计 2.0 毫秒，平均每次 0.002 毫秒。单次耗时一比，0.4375 除以 0.002 等于 218.75 倍——矩阵乘调一次的时间，够逐元素算子调两百多次。调用次数排第一的算子，耗时可能只排第六；耗时排第一的算子，调用次数可能只有几十次。按调用次数排序的清单，排出来的是“谁最忙”，按耗时排序的清单，排出来的是“谁最贵”。第 2 节要对照的是“谁最贵”，因为回退的代价是按耗时占比算的，不是按调用次数算的。一个被调 1000 次的算子回退了，每次多花的时间再乘 1000 次，账也要回到累计耗时上算——调用次数本身不决定瓶颈，调用次数乘以单次耗时才决定。

算子名那一列也有坑：PyTorch 层面的名字与芯片侧的承接之间隔了一层映射。Profiler 导出的是 `aten::mm`、`aten::bmm` 这样的名字，而第 2 节要问的是“这个 PyTorch 算子在 Ascend 上由哪个实现承接”。`aten::mm` 与 `aten::bmm` 在名字上只差一个字母，承接却可能完全不同：一个是二维矩阵乘，一个是批量矩阵乘，走的 kernel 路径与分块策略都不一样，清单上必须把它们当成两个算子，合并成“矩阵乘”一类就等于把对照的粒度弄粗了。反过来也有合并的坑：`aten::silu` 与 `aten::mul` 在 profiler 里是两个算子，实际执行时可能被融合成一个 SiluAndMul kernel——清单上是两行，硬件上是一次执行。按算子名逐条数，会把一次融合执行数成两次；按 kernel 数，又会丢掉 PyTorch 层面的调用结构。这一节的清单保留 PyTorch 算子名原样，不提前合并、不提前折算，融合那笔账留到第 2 节对照时再算。清单的任务是如实记录，不是提前聪明。

预期会看到的构成，源文件里已经点过名：矩阵乘（GEMM 类）、attention、归一化（RMSNorm）、激活（SiluAndMul）、各类逐元素算子。预期之所以值得先写下来，是为了对照——真实清单出来之后，先问一句“预期里的名字是不是都在”，再问“有没有预期之外的名字”。预期之外的名字最有信息量：一个没想到的算子进了 Top 10，往往意味着模型里有一段之前没注意到的计算，例如某个 reshape 或类型转换被反复调用。预期是用来发现意外的，不是用来确认自己没错的。清单采完之后，预期之外的行要单独圈出来，第 2 节对照时优先查它们——熟悉的算子对照起来有路可循，陌生的算子才最容易在“待核实”里躺平。

最后是无卡时的退路：先写 profiler 脚本，实测标注“待补测”。脚本本身是可以现在写完的——模型、形状、导出列、排序口径、输出格式，全都不依赖显卡。写完脚本，清单的框架就立住了：表头、列、Top 10 的位置都在，只等有卡跑一遍把数字填进去。待补测不是没做，是把“能做的部分”与“必须等硬件的部分”分开，前者现在交货，后者不硬编数字顶替。清单上任何一行没有实测数字，就写待补测，不拿别的模型、别的形状的数来凑——凑出来的清单比空着的清单更有害，空着至少诚实。

**产出**：一份算子清单（算子名 × 调用次数 × 耗时 × 占比，Top 10 标出），表头注明模型、形状与排序口径（CUDA 自耗时）；无卡时脚本先行，数字标注“待补测”。

### 第 2 节 · 逐条对照：哪些算子在芯片侧有对应实现，哪些会回退

**目标**：出一份算子覆盖矩阵——每个算子在芯片侧的状态是什么。

**怎么做**（60 分钟，无硬件可做；对照依据是官方文档与插件源码，核实不了的标“待核实”）：

覆盖矩阵要回答的问题只有一个：清单上的每个算子，在 Ascend 上由谁承接。这个问题听起来像查表，实际是个判断题——“由谁承接”有四种答案，每一种答案背后的含义、证据要求与风险都不一样。先把四档的定义钉牢，再逐条往里填，顺序不能反。定义不清的矩阵，填得越满越危险，因为每一行都像是有结论，其实只是把不确定藏进了分类里。

第一档是原生实现：CANN 或者 AscendC 里有对应实现，性能有保障。原生这两个字的分量在于“官方实现”——它跟着芯片的软件栈一起发布、一起测试，接口稳定，出了问题有明确的责任方。查它的证据是 CANN 文档与 AscendC 的算子清单，写进矩阵的时候要写清文档链接与版本。性能有保障这四个字也要小心：它保障的是“这个实现是为这块芯片写的”，并不保证它在你的形状下是最快的。原生实现在某个 batch 下可能走通用路径，换个形状才走优化路径——状态是原生实现，性能仍然要实测确认。把“有原生实现”读成“性能没问题”，是这一档最常见的误读。

第二档是插件实现：vllm-ascend 的 `ops/` 或者 `_cann_ops_custom/` 里有实现。插件与原生的区别不在“能不能跑”，而在“谁维护、跟谁走”。插件实现是推理框架一侧为了补齐覆盖而写的定制算子（CustomOp），它跟着 vllm-ascend 的版本走，CANN 升级的时候它可能要跟着改。第 3 周第 3 节定位过这类位置，方法就是那一节的方法：先在 `ops/` 里找，再到 `_cann_ops_custom/` 里找，两个地方都没有，再去文档里找。插件实现的风险不在正确性，而在跟随性——框架升级、芯片驱动升级，插件都可能要重新适配。矩阵里标插件实现的行，要把文件路径写清楚，路径就是它的证据，也是它下次失效时最先要复查的地方。

第三档是回退：没有芯片侧实现，走 CPU 或者 PyTorch 参考实现。回退这两个字容易被读成“跑不起来”，其实正好相反：回退的意思是“能跑，但是慢”。CPU 回退是把计算搬到 CPU 上做，PyTorch 参考实现是用 PyTorch 的通用算子顶上去，两个都是“先保证能跑”的退路。回退的代价不在功能上，在时间上——而时间的代价要到下一段的算式里才算得出来。回退还有一个更隐蔽的形态：静默回退。模型跑起来了，没有报错，速度却比预期慢得多，查了半天才发现某个算子一直在走 CPU。矩阵的意义之一就是把静默的东西显性化：每一行回退都要写出来，写出来的回退才会被当成瓶颈来治，没写出来的回退只会被当成“这块芯片就这么慢”。

第四档是待核实：文档与源码里都没找到明确说法。待核实不是第五种状态，是“现在下不了结论”的诚实标记。查不到与不存在是两回事：文档里没写，可能是这个算子确实没有实现，也可能是实现藏在另一个名字下面，或者文档滞后于代码。把查不到直接写成回退，等于用猜测填了矩阵；把查不到写成待核实，等于把问题留给了下一轮核实。矩阵里待核实的行数本身就是信息：待核实占 Top 10 的一半，说明对照工作只做了一半，结论的置信度也要跟着打对折。核实日期那一列就是为此存在的——每个状态都要有个日期，日期一过，状态就过期。芯片软件栈的版本在变，今天的原生实现，明天可能换了路径；今天查不到的算子，明天可能补了插件。没有日期的矩阵是过期了也不知道的矩阵。

填矩阵的时候，对照的顺序也有讲究：从 Top 10 往下填，先填耗时占比大的行。理由在下一段的算式里：占比大的行，一档之差就是端到端的几成；占比小的行，档位错了也伤不到大局。时间有限的时候，先把贵的算子查实，便宜的算子可以先标待核实——矩阵的填写顺序本身就是一次资源分配。

关键问题是回退的算子在不在关键路径上，判断的方法是把“状态 × 耗时占比”一起看，不要只看状态。算式先摆出来：把一次前向的总时间归一化成 100 份，一个占 20 份的算子回退后慢 10 倍，它的时间从 20 份变成 200 份，总时间从 100 份变成 80 加 200 等于 280 份，端到端慢 2.80 倍。一个只占 1 份的算子同样回退后慢 10 倍，总时间从 100 份变成 99 加 10 等于 109 份，端到端只慢 1.09 倍。同样的回退、同样的 10 倍，账算出来一个是 2.80 倍，一个是 1.09 倍——差的不是回退本身，是回退的算子原来占多少。换一组数再验一次：占 20 份的算子回退后只慢 3 倍，总时间是 80 加 60 等于 140 份，端到端慢 1.40 倍。回退倍数从 10 倍降到 3 倍，端到端从 2.80 倍降到 1.40 倍——瓶颈的定价同时取决于占比与回退倍数，两个数缺一个，结论都下不了。这几组数都是算例，不是实测，它们的任务是把“状态 × 占比”这个乘法刻进判断里：以后看到矩阵里任何一行回退，先问它占多少，再问它慢几倍，两个数都有了，才轮到说“瓶颈”两个字。

这组算式还有一个推论值得注意：占比最大的那个算子，哪怕只慢一点点，账也很大。占 42 份的算子回退后只慢 2 倍，总时间是 58 加 84 等于 142 份，端到端慢 1.42 倍——慢 2 倍听起来不严重，乘上 42% 的占比，端到端还是慢了四成多。反过来，占 1 份的算子就算慢 10 倍，端到端也只慢 9%。优化的优先级就按这个乘法排：先治“占比大 × 回退多”的行，再治“占比大 × 回退少”的行，“占比小”的行无论回退多少倍都可以往后放。矩阵的阅读顺序就是这个乘法的排序，不是按算子名字母排，也不是按档位排。

填覆盖矩阵：

| 算子（PyTorch 名） | 耗时占比 | 芯片侧状态 | 实现位置（文件/文档链接） | 核实日期 |
|---|---|---|---|---|
| `aten::mm` | … | 原生实现 | … | 某年某月 |
| `aten::silu` / `aten::mul` | … | 插件实现 | … | 某年某月 |
| … | … | 待核实 | — | 待核实 |

矩阵的每一列都有它的任务。算子名那一列用的是 PyTorch 名，与第 1 节清单上的名字一一对应——对照的第一步是名字对齐，清单上的 `aten::mm` 与矩阵里的 `aten::mm` 必须是同一个算子，名字一错行，后面的状态就贴错了行。耗时占比那一列从第 1 节清单上抄过来，不重新测，抄的时候连形状一起抄，占比离开形状就没有意义。芯片侧状态那一列只填四档里的一个，不写模棱两可的话——“可能有实现”不是状态，“大概能跑”也不是状态，写不进四档的话就写待核实。实现位置那一列是证据列：原生实现写文档链接，插件实现写文件路径，回退写回退到的路径（CPU 还是 PyTorch 参考实现），待核实写一个破折号——破折号的意思是“这里本该有证据，现在没有”，比空着更诚实。核实日期那一列是有效期列：状态是哪天查的、对照的是哪个版本的文档与源码，日期与版本一起写，过期复查的时候才知道从哪查起。

矩阵填完之后要做一次完整性检查，检查的方法是算术。Top 10 的耗时占比加起来是多少？沿用第 1 节的算例，Top 10 合计占 92.0%，剩下的长尾一共 8.0%。长尾里有多少个算子？假设清单上一共有 97 个算子，Top 10 之外还有 87 个，8.0% 除以 87，每个长尾算子平均占 0.092%。平均数这么小，意味着长尾里单个算子的档位几乎不影响端到端——就算长尾里有一半算子回退，账也大不到哪去。但平均数也会骗人：长尾的分布可能极不均匀，87 个算子里可能藏着一个占 3% 的算子与 86 个占 0.06% 的算子。完整性检查的第二步就是把长尾按占比再排一次，确认 Top 10 的切口没有正好切在两个占比相近的算子中间——第 10 名占 0.8%，第 11 名占 0.79%，切口切在哪，矩阵的覆盖就差多少。Top 10 是工作量的切口，不是物理上的分界线，切口附近的算子值得多看一眼。

还有两个误解值得在填表之前先拆掉。一个误解是“有实现就等于快”。原生实现与插件实现都只回答“有没有人接这个算子”，不回答“接得快不快”。一个插件实现可能只覆盖了某几个形状，换个形状就悄悄回退；一个原生实现可能在小 batch 下走通用路径，性能与回退差不了多少。状态是覆盖问题，速度是性能问题，矩阵只解决前者，后者要等有卡实测。另一个误解反过来：“回退就等于不能用”。回退的模型能跑、结果正确，只是慢。原型验证阶段，回退完全可以接受——先把流程跑通，再逐个消灭回退。把回退当成“不能跑”，会把一个能跑的方案误判成死路；把回退当成“没事”，会把一个慢三倍的方案误判成上线状态。矩阵的价值正在于把这两种误读都挡在外面：回退写在那里，占比写在那里，端到端的账也写在那里，结论自己会说话。

最后是这一节的证据纪律。对照依据只有两类：官方文档与插件源码。论坛帖子、聊天记录里的说法、二手转述，都不能作为状态的证据——它们可以当线索，顺着线索去文档与源码里找，找到才写进矩阵，找不到就写待核实。线索与证据的区别要分清：线索告诉你“去哪找”，证据告诉你“找到了什么”。矩阵里每一行的实现位置列，写的都应该是证据本身，而不是线索的出处。

**产出**：一份算子覆盖矩阵（Top 10 全覆盖，带核实日期），外加一句阅读方法：矩阵按“状态 × 耗时占比”的乘法读，占比大的回退行优先处理。

### 第 3 节 · 迁移 benchmark：读懂 TritonAscendBench 的设计

**目标**：理解迁移评测的设计——它测什么、怎么判分、没卡时能做什么。

**怎么做**（60 分钟，无硬件可做）：

一个 benchmark 的设计，读法是先问它测什么，再问它怎么判分，最后问它在什么条件下会撒谎。三个问题按这个顺序问，读出来的才是设计本身；只看分数不看设计，读出来的只是一张排行榜。TritonAscendBench 测的是生产级 Triton kernel 从 GPU 迁移到 Ascend 的可用性——“可用性”三个字是题眼：它测的是“一个在 GPU 上写好的 Triton kernel，搬到 Ascend 上之后还能不能用”。速度是可用性的一部分，不是全部；一个跑得快但结果不对的 kernel，可用性是零分。

先看规模：101 个任务，分 L1（36 个）、L2（35 个）、L3（30 个）三档难度。三个数先加起来验算：36 加 35 等于 71，再加 30 等于 101，与总数对得上。占比逐项算：L1 是 36 除以 101，约 35.6%；L2 是 35 除以 101，约 34.7%；L3 是 30 除以 101，约 29.7%。三档的占比加起来是 35.6 加 34.7 加 29.7 等于 100.0%，账是平的。L1 与 L2 加起来是 71 个任务，占 70.3%——七成的任务集中在中低两档，最高档只占不到三成。这个分布本身就是设计意图：迁移能力的底盘在中低档，L1 与 L2 测的是“常规迁移能不能成”，L3 测的是“难啃的迁移能不能成”。一个只在 L3 上丢分的 kernel，与一个在 L1 上就丢分的 kernel，问题完全不在一个层次上：前者是攻坚能力，后者是基本功。读分数的时候先按档拆开看，拆开之前不要谈总分。

再看它的工作流。入口命令是 `tab init-solution`，作用是生成解法骨架——一个空的、结构已经摆好的解法，写 kernel 的人往里面填。骨架的意义在于格式统一：所有人的解法长一个样子，评测才能批量跑，分数才能互相比较。骨架生成之后是 `tab run`，跑单次评测；`tab run-suite` 是批量跑，把一整套任务一次跑完；最后是 `tab package` 与 `tab validate-submission`，打包与校验提交——打包是把解法收成可提交的形态，校验是检查提交的格式与完整性，格式不对的提交在判分之前就被拦下来。五个命令连起来是一条流水线：生成骨架、单跑验证、批量验证、打包、校验提交。每一步都有明确的产物，下一步的输入就是上一步的产物，流水线的好处是每一步都能独立重跑——批量跑挂了，回单跑定位；校验不过，回打包检查。工作流的命令要按这个顺序理解，不要只背五个名字。

解法的入口是 `npu_triton_impl(inputs) -> dict`，这个签名本身就是一道防作弊的设计。入口函数接收输入、返回一个字典，评测拿返回的字典与 PyTorch 的参考输出对照。关键约束写在源文件里：必须是一个真正的 Triton kernel，不能是 PyTorch 的取巧实现。为什么要立这条规矩？因为评测测的是“迁移”，不是“在 Ascend 上把结果算出来”。用 PyTorch 的算子拼出正确结果，在 Ascend 上当然也能跑，但它绕过了迁移本身——Triton kernel 的写法、tiling、访存，全部没有被检验。允许取巧的评测，测出来的分数与迁移能力无关，只与“会不会抄近路”有关。入口签名的存在，就是把近路堵死：评测只认 Triton kernel 写出来的结果，PyTorch 的捷径不在入口的验收范围里。这条约束不是对写法的挑剔，是对评测有效性的保护。

判分分两半：正确性与性能。正确性在 CPU 上对照 PyTorch oracle 判分，有容限分级——oracle 是 PyTorch 算出来的参考答案，判分是把 kernel 的输出与参考答案比，误差落在容限里算过，容限之外算不过，分级的意思是误差大小对应不同的档位，而不是一刀切的过与不过。正确性放在 CPU 上判，有两个好处：判分环境与被测硬件解耦，没有 NPU 也能判正确性；判分标准统一，所有人的 kernel 都对着同一个 oracle。在 CPU 上判正确性的另一面是：它判不了性能。性能需要在 910B 上实测——性能是硬件相关的量，离开目标芯片谈性能是没有意义的。正确性与性能分开判，是这个 benchmark 最诚实的一处设计：它不把“在 CPU 上算对了”等同于“在芯片上跑得快”，两个分数各有各的场地。

`tab run` 默认 `--backend npu`，`--backend gpu` 是 CUDA 自查。默认值的选择也有讲究：默认跑 NPU，说明这个 benchmark 的主场就是迁移之后的那一端；GPU 后端是自查用的——先确认 kernel 在 GPU 上是对的，再谈迁移。自查的顺序不能反：一个在 GPU 上就是错的 kernel，搬到 Ascend 上再错一次，排错的时候分不清是迁移引入的问题，还是 kernel 本来就有的问题。先在 GPU 上把正确性钉死，迁移之后如果错了，错的一定是迁移那一步引入的。这个顺序是排错的基本功：每次只动一个变量，动完立刻验证。没有 NPU 的时候，自查就是能做到的最远一步——GPU 上跑通，正确性在 CPU oracle 下通过，NPU 那一步标上标记，等有卡再补。

没有 NPU 时结果会标 `npu_antihack_unverified`，这个标记值得多看两眼。它的意思是：NPU 一侧的结果没有经过验证。标记的存在说明设计者想过一种作弊：没有 NPU 却声称 NPU 上跑通了。防作弊的办法是把无卡运行的结果打上标记——标记本身就是诚实设计的一部分。一个 benchmark 诚不诚实，不取决于它有没有防作弊的墙，取决于它怎么处理“验证不了”的情况：静默地把 GPU 结果当成 NPU 结果，是撒谎；明确地标上未验证，是诚实。读到这个标记的时候，不要把它当成失败，它只是在说“这一步还没做”，与“这一步做错了”是两回事。第 5 周动手的时候，这个标记还会再出现，到时候就知道它不是障碍，是路标。

看它的 worked example：`USAGE.md` 里的 `L2_04_silu_and_mul`——一个完整的 silu_and_mul Triton kernel。这个例子选得有深意：silu_and_mul 是 M2 第 6 周写过的同一个算子，第 5 周会移植它。例子在这里出现，本周的任务就是先读懂它：kernel 本身怎么写，评测怎么跑，判分怎么判。读懂的检验标准是：不看文档，能不能说出这个 kernel 的输入输出是什么、评测拿什么跟它比、哪一步需要 NPU、哪一步在 GPU 上就能完成。读懂之后，第 5 周直接用它练手——本周是读，第 5 周是做，读与做之间隔着一周，隔这一周的原因是：读懂一个完整的例子，比囫囵吞枣地读十个例子更有用。例子里的算子不是随便挑的：silu_and_mul 在第 2 节的矩阵里是插件实现的那一行，在 M2 第 6 周是手写过的 kernel，在第 5 周是移植的对象——同一个算子在三周里以三种身份出现，读懂它就等于把三周串在了一起。

注意版本：它的 NPU 环境参考配置在 `docker/Dockerfile.npu` 里（CANN 8.5.0、torch_npu 2.8.0.post1、triton_ascend 3.2.0）——与第 2 周的 vllm-ascend 锚点（CANN 9.2.1）并不属于同一套版本。不同工具钉不同版本是常态，记录时写清“这一步用的是哪套版本”，不要混在一起。版本这件事上最容易犯的错是“统一强迫症”：看到两个工具的 CANN 版本不一样，就想把它们统一到同一个版本。统一本身没有错，错的是不加验证地统一——TritonAscendBench 的评测结果是在 CANN 8.5.0 下测的，换到 9.2.1 下重测，分数可能变，也可能不变，没测过就不知道。记录的纪律是：每一步写清版本，复现的时候按记录里的版本复现，升级版本是另一项独立的工作，要单独排期、单独验证。版本不是装饰，是实验条件的一部分。

分数的读法也可以先练一组算例，免得第 5 周拿到真实分数时不会读。假设某次批量跑的结果是 L1 通过 30 个、L2 通过 28 个、L3 通过 12 个，逐项算通过率：L1 是 30 除以 36，约 83.3%；L2 是 28 除以 35，等于 80.0%；L3 是 12 除以 30，等于 40.0%。总通过是 30 加 28 加 12 等于 70 个，占 101 个任务的约 69.3%。这组数是读法算例，不是实测，它演示的是拆开读与合起来读的区别：总分 69.3% 看起来平平，拆开看是 L1 与 L2 都在八成上下、L3 只有四成——瓶颈在高难度档，不在基本功。反过来，如果 L1 只有四成、L3 也有四成，总分一样，结论完全不同：基本功就有问题，高难度的分数没有参考价值。同一张总分，拆开之后是两个故事，读 benchmark 的分数要先拆档，再谈总分。

一个常见的误解：“benchmark 的分数就是芯片的速度”。TritonAscendBench 的分数首先是迁移可用性的分数：kernel 能不能搬过去、搬过去对不对，速度是第二步才问的事。一个在 L1 全过、L3 全挂的解法，说明常规迁移没问题，难迁移还没过关；一个正确性全过、性能全挂的解法，说明东西能跑，但是慢。分数要按“正确性先行、性能随后”的顺序读，顺序一反，就会把“能跑但慢”误读成“芯片不行”，或者把“跑得快但不对”误读成“迁移成功”。第 5 周移植的时候，这个顺序还会再用一次：先求对，再求快。

没卡时能做到哪一步，现在可以收拢了：读懂工作流与判分规则，读懂 `L2_04_silu_and_mul` 这个例子，在 GPU 上把 kernel 的正确性自查通过，NPU 一侧的步骤标注 `npu_antihack_unverified`、等有卡再补。能做的部分与不能做的部分分得越清，待补测的清单就越干净——清单上每一项都知道自己在等什么，而不是笼统地写一句“没卡做不了”。

**产出**：一页笔记：TritonAscendBench 测什么、怎么判分、没卡时能做到哪一步。笔记里写清三档任务数（L1 36、L2 35、L3 30，共 101）、工作流命令、判分场地（正确性在 CPU 上对 oracle，性能在 910B 上实测）、版本锚点（CANN 8.5.0、torch_npu 2.8.0.post1、triton_ascend 3.2.0）与无卡标记（`npu_antihack_unverified`）。原始仓库：[TritonAscendBench](https://github.com/Gxj230958/TritonAscendBench)。

### 第 4 节 · 动手：对清单里最耗时的 3 个算子，查它们的芯片侧实现状态

**目标**：出一份“Top 3 算子覆盖分析”。

**怎么做**（60 分钟，无硬件可做；查不到的写“待核实”，不猜）：

第 2 节的矩阵是 Top 10 的全景，这一节是 Top 3 的特写。全景回答“整体覆盖怎么样”，特写回答“最贵的三个算子到底由谁承接”。特写的必要性在第 1 节的算例里已经算出来了：Top 3 合计 34.5 毫秒，占总时间 50.0 毫秒的 69.0%——近七成的时间花在三个算子身上。这三个算子的状态，基本上就决定了整张矩阵的结论；反过来，Top 10 里排在后七位的算子加起来才占 23.0%，它们的状态再差，端到端的账也有限。资源有限的时候，特写优先于全景的细化：先把最贵的三行查实，再回头补后七行。

动手的方法是第 3 周第 3 节的方法，对每个算子做三件事：在 vllm-ascend 的 `ops/`、`attention/`、`_cann_ops_custom/` 里找它的实现，在 CANN 文档里找它有没有原生实现，然后在覆盖矩阵里更新它的状态与实现位置。三件事的顺序也有讲究：先找插件，再找原生，最后更新矩阵。先找插件的原因是插件的位置最具体——文件路径就在那里，有就是有，没有就是没有，查起来最快。原生实现查的是文档，文档的说法可能滞后于代码，也可能把几个算子合在一起讲，查起来要慢一些。更新矩阵放在最后，是因为状态要等两边的证据都齐了再定：插件里找到了，就不用再纠结文档里有没有；插件里没找到、文档里找到了，状态就是原生实现；两边都没找到，写待核实。顺序反过来，先查文档再查插件，结论是一样的，时间花得更多——先查便宜的证据，再查贵的证据。

三个目录的分工要先分清，找的时候才不会走错地方。`ops/` 里是 vllm-ascend 的定制算子，`_cann_ops_custom/` 里也是定制算子一侧的实现，两个地方都是“插件实现”这个档位的证据来源。`attention/` 目录单独拎出来，是因为 attention 在 vLLM 里走后端（backend），不是 CustomOp——第 3 周第 3 节已经讲过这个边界，这里再确认一次。边界的意思是：attention 相关算子的实现不要到 CustomOp 的位置去找，找了也没有；要到 attention 后端里去找。找错地方的后果是“找错了地方之后误判为没有实现”——没找到与不存在之间，差的就是一个正确的查找位置。Top 3 里如果有 attention 相关算子，第一步就是确认它的后端路径，而不是把它当成普通算子去 `ops/` 里翻。

每个算子查完之后，写一句话结论，格式是固定的：“这个算子在 Ascend 上由某某实现承接（文件路径），预计不是瓶颈 / 可能是瓶颈 / 待核实”。一句话里有三个要素，缺一个都不完整：承接方是谁，证据在哪，结论是什么。承接方写实现的名字与位置，证据就是那个文件路径或者文档链接，结论只取三个档位里的一档。一句话结论的价值在于可复核：别人拿着这句话，顺着文件路径能找到同一个实现，顺着占比能算出同一个判断，复核通过，结论才成立。写两句话的结论、写一段话的结论，都不如一句话——多出来的话往往是修饰，修饰里最容易藏进没验证的判断。

三个档位的结论，各自需要什么证据，要先说清楚。预计不是瓶颈，需要两条证据同时成立：实现存在（原生或者插件），而且耗时占比不大，或者实现本身经过验证、性能有保障。只满足一条，结论下不了：实现存在但占比 42%，回退倍数哪怕只有 2 倍，端到端的账也是 1.42 倍（第 2 节算过），这种行不能叫“不是瓶颈”；占比很小但实现状态是待核实，结论只能写“还没查到”——“不是瓶颈”这四个字，证据不足时写不得。可能是瓶颈，同样需要两条证据：回退状态属实，而且耗时占比够大——第 2 节的乘法在这里再用一次，占比与回退倍数两个数都有了，才轮到说“可能是瓶颈”。待核实最简单：任何一条证据缺失，结论就是待核实。待核实不是结论的降级，是证据不足时的唯一诚实选项。

拿第 1 节的算例把 Top 3 的账算一遍，算完就知道特写要盯着谁。Top 3 是 21.0、9.0、4.5 毫秒，合计 34.5 毫秒，占 69.0%。假设排第三的算子占 4.5 毫秒、也就是 9.0%，它的状态是回退、回退后慢 5 倍，端到端的账是这样算的：总时间 50.0 毫秒里，4.5 毫秒变成 22.5 毫秒，其余 45.5 毫秒不变，总时间变成 45.5 加 22.5 等于 68.0 毫秒，是原来的 1.36 倍。一个只排第三的算子，回退 5 倍，端到端慢 36%——排第三尚且如此，排第一的更不用算：占 21.0 毫秒、也就是 42.0% 的算子，回退后只慢 2 倍，总时间是 29.0 加 42.0 等于 71.0 毫秒，是原来的 1.42 倍。两组算例都是算例，不是实测，它们的任务是把 Top 3 特写的优先级算清楚：特写不是因为“前三名”这个名次好听，是因为前三名的占比乘上回退倍数之后，端到端的账最大。查 Top 3 的状态，就是在查端到端时间里最大头的那几笔账由谁来付。

特写里还有一个容易踩的坑：把“清单上的名字”等同于“芯片侧的名字”。第 1 节讲过，profiler 导出的是 PyTorch 算子名，芯片侧的承接要落到“这个 PyTorch 算子由哪个实现承接”。`aten::mm` 在清单上是一行，在芯片侧可能对应多个实现：不同形状走不同 kernel，不同 dtype 走不同路径。查的时候要带着形状去查——清单上的 `aten::mm` 是在哪个形状下被调的，芯片侧的实现支不支持那个形状。形状不带的对照，等于默认“所有形状都有实现”，这个默认是没有证据的。矩阵里的实现位置列，写文件路径的时候最好把形状条件也带上一句，条件写不出来的，状态的置信度就要打折扣。

SiluAndMul 这个算子值得在特写里单独提一句，因为它在三周里出现了三次。第 2 节的矩阵里，它是插件实现的那一行；M2 第 6 周手写过同一个算子的 Triton kernel；第 3 节的 worked example `L2_04_silu_and_mul` 也是它；第 5 周要移植它。如果 Top 3 里有它，特写的工作量会小很多——M2 第 6 周的 kernel 还在，第 3 节的例子也读过了，查它的芯片侧状态有现成的线索可循。如果 Top 3 里没有它，也不用失望：特写的方法是通用的，换一个算子，三个目录、两边证据、一句话结论的流程不变。特写练的是方法，不是某一个算子的答案。

特写查完之后，要回头与第 2 节的矩阵对一次账：特写里写的状态，与矩阵里同一行的状态是不是一致。不一致的时候，以证据更新的那一边为准，矩阵里那一行的状态与核实日期一起改掉，改的理由写在行边上。矩阵不是写完就 frozen 的文档，特写就是一次针对 Top 3 行的复查，复查出不一致是正常的，藏起来才不正常。对账还有第二层意思：Top 3 的三行查实之后，矩阵里剩下七行的置信度也要重新评估——如果三行里有两行是待核实，矩阵整体的结论就要跟着降级，第 5 节归因的时候不能假装矩阵是完整的。

最后是这一节的产出纪律：每个算子只写查到的东西。实现位置写的是文件路径或者文档链接，路径写不出来就写待核实；一句话结论写的是三个档位里的一个，证据不足就写待核实。特写只有三个算子，三个算子的篇幅都很短，短有短的好处：短到每一句话都能被复核，长到藏不进一句没验证的判断。三个算子查完，矩阵里的 Top 3 行就从“填了状态”升级成“有证据的状态”，第 5 节的归因才有东西可贴。

**产出**：一份“Top 3 算子覆盖分析”（每个算子：实现位置 + 一句话结论），结论只取三档——预计不是瓶颈 / 可能是瓶颈 / 待核实；查不到实现位置的算子，结论只能是待核实。

### 第 5 节 · 小结：覆盖矩阵的结论——瓶颈在算子覆盖，还是在调度/显存

**目标**：写出一页结论，用 Four Walls 归因。

**怎么做**（60 分钟，总结无卡可做）：

回到第 1 周的 Four Walls：生态、显存、通信、精度。把覆盖矩阵的结论往四面墙上贴——瓶颈在生态墙（算子没实现、要回退）？还是显存墙（算子有实现，但显存带宽跟不上）？还是精度墙（量化实现有差异）？通信墙（多卡）本周不涉及，先空着。

贴墙的方法是按证据贴，不按感觉贴。生态墙的证据是矩阵里的回退行：某个占 20% 时间的算子回退了，端到端的账是 2.80 倍（第 2 节算过），这就是生态墙的证据——算子没实现，时间花在回退上。显存墙的证据是另一类：算子有实现，但搬运量大、带宽跟不上。本周没有实测带宽数据，显存墙贴不上去，就空着，不硬贴。精度墙同样如此：本周没有量化的实测误差，精度墙也没有证据，先不下结论。通信墙本周不涉及，空着是计划内的空着，不是遗漏。

一页结论的格式是固定的：“本次分析的瓶颈主要在 X 墙，依据是覆盖矩阵里的某几行（算子名 + 耗时占比 + 状态）；Y 墙没有证据，先不下结论”。格式里有两句话，第一句下结论，第二句划边界。只写第一句不写第二句的结论，是不完整的结论——它没说自己没查什么，读的人会以为四面墙都查过了。Y 墙那一句的价值就在于此：它把“没证据”写出来，免得结论被读成比实际更确定的样子。依据那几行要写全三要素：算子名、耗时占比、状态。三要素缺一个，依据就站不住：只有算子名没有占比，不知道它有多贵；只有占比没有状态，不知道它为什么慢；只有状态没有占比，不知道它值不值得治。

还有一句要写在结论的边上：本周的结论是覆盖层面的结论，不是性能层面的结论。矩阵回答的是“有没有实现”，没有回答“实现得快不快”。一个原生实现的算子，在某个形状下可能仍然很慢，慢的原因要等有卡实测才能定位——可能是 tile 配置，可能是访存，也可能是别的原因。本周的归因贴到生态墙上，意思是“覆盖是目前能看到的瓶颈”，不是“性能问题已经排除了”。两句话的结论，边界要划清。

把本周的检查清单并进总清单，加两条：“覆盖矩阵的 Top 10 全了吗”“每个状态都有核实日期或‘待核实’标记吗”。第一条查的是完整性：Top 10 有没有漏行，漏掉的那一行是不是正好在切口附近。第二条查的是诚实性：每一行状态的背后，是一个日期或者一个待核实标记，没有第三种情况。两条都过，本周才算收拢。

**产出**：一页结论（瓶颈归因 + 依据），格式是“瓶颈主要在 X 墙，依据是某几行；Y 墙没有证据，先不下结论”。
## 第 29 章 · 移植案例：把 Triton kernel 搬到 Ascend

把一个在 GPU 上跑通的 Triton kernel 搬到 Ascend，最先遇到的往往不是“算子不存在”。更常见的是三种症状：编译过了，结果和参考实现对不上；结果对了，速度慢一个数量级；或者一启动就卡在图编译上。它们指向同一类原因——代码里有一批“默认这里是 CUDA”的假设，换了硬件之后假设不再成立，但代码不会主动报错。

本章用一个示例仓库和一个小 kernel 把这批假设拆开。`nano-vllm-ascend` 是把小推理引擎移植到 Ascend 的示例仓库，体量小到可以读完；SiluAndMul 是第 24 章手写过的算子，在 TritonAscendBench 里有现成的迁移任务 L2_04。前者看移植会改动哪几层，后者把语义差异落到一行行代码上。最后收成一份可执行的检查清单。

**本章目标。** 给定一个小推理引擎和一份 Triton kernel，能列出它依赖 CUDA 的假设、判断哪些在 Ascend 上会失效、把失效点写成“现象—定位—改法—重测结果”的记录，并在没有 NPU 时把 NPU 步骤标成待补测，而不是标成通过。

### 第 1 节 · 示例：把一个小推理引擎移植到 Ascend

读一个引擎之前，先看它有多大。打开 [nano-vllm-ascend](https://github.com/linzm1007/nano-vllm-ascend)（把 nano-vllm 移植到 Ascend 的示例仓库，中文 README），README 自报了行数：20 个文件共 4652 行，其中 models 目录 5 个文件 2224 行，占 47.8%；models 之外的 15 个文件 2428 行，占 52.2%。三个数对得上：4652 减 2224 等于 2428；2224 除以 4652 约等于 47.80%；2428 除以 4652 约等于 52.20%，加起来 100.0%。这组数是仓库 README 自报的，没有逐文件复核；它只是某个时点的快照——当前仓库的 models 目录已经能数到 7 个 Python 文件，文件总数也早超过 20，README 的代码行数表与现在的目录对不上。读体量时拿它当量级，不拿它当审计。

体量的意义在分母。vLLM 是几十万行量级，读它先要决定从哪个目录进；2428 行是能从头读到尾的规模，读完能把推理链路画成一条线：请求进来、调度、组批、模型前向、采样、KV 缓存回收。2224 行模型目录是另一回事：它按模型结构重复，几个架构各写一遍注意力与 MLP，读它是查“某个模型怎么接进来”，不是查“引擎怎么运转”。先读 models 之外的 2428 行，再按需读模型目录，是这份体量给出的读法。

模型清单回答定位，不回答能力。README 的模型表按 6 个架构列：Qwen3、Qwen2、Llama、Qwen3-MoE、Qwen3-VL、MiniCPM，覆盖稠密、MoE 与多模态三类结构。这是挑选的结果：三类结构各有一个能跑通的例子，说明仓库的目标不止“跑通一个模型”；但清单里没有长上下文模型、没有超大模型，MoE 与 VL 两行还标着“暂时不支持入图”。读支持列表时把“没支持什么”一起读，比只数“支持几个”更能看清边界。

移植改动可以收进四类，每一类落在软件栈的一层上：

| 类别 | 改什么 | 失效时的症状 |
|---|---|---|
| 框架适配 | `device="cuda"` 换 `npu`，`torch.cuda.*` 换 `torch_npu` 的对应接口；分页 KV 写回换 `_npu_reshape_and_cache` | 引擎起不来，或算子找不到实现 |
| 图编译 | torchair 把一次前向抓成图，编译后重放；形状变了要重编 | MoE 与 VL 标“暂时不支持入图”；动态 batch 反复重编 |
| 注意力后端 | FlashAttention 库版、NPU 原生版、PyTorch 原生版三选一 | 跑得对但慢，或后端不可用 |
| 自定义算子 | rms_norm 的三种写法：PyTorch C++ 扩展、AscendC、`torch_npu` 内置 | 注册了但调用不到，或编译不过 |

四类里，图编译与注意力后端最容易被当成“配置项”，它们各有一条边界。torchair（Ascend 上的图编译接口）把一次前向的计算图抓下来、编译成可重放的单元，省的是几十个小算子逐个下发的启动开销，计算量不变。省下来的开销有前提：图编好之后形状固定。形状一变就要重编，重编的代价可能比省下的启动开销还大——MoE 与 VL 标“暂时不支持入图”，边界就在形状上：MoE 的形状随路由结果变，VL 的形状随图像输入变，进图要反复重编。这个标注说的是图编译的适用边界，不是一个待修的 bug。

注意力后端是同一笔计算的三份实现。README 用星级对比：性能上，FlashAttention 库版与 NPU 原生版都是五星，PyTorch 原生版两星；可读性上顺序相反，PyTorch 原生版五星，库版两星。星级是仓库自评，不是实测，它表达的是取舍方向：跑得快的写法难读，好读的写法慢。生产环境 GPU 选库版、NPU 选原生版，学习调试选 PyTorch 原生版——三条路并存，不是三选一之后另外两个作废。换到新芯片，注意力后端要重新选一次，依据是新芯片上有哪个后端可用、数值行为与性能如何。

PagedAttention 的账可以单独算，因为它是仓库里自带算例的优化。序列长 1000、最大长度 4096，按最大长度预分配，浪费是 `(4096-1000)/4096`，约 75.6%。按 block 分配，block 取 16，`1000/16 = 62.5`，向上取整 63 块，实际分配 `63×16 = 1008` 个位置，浪费是 `(1008-1000)/1008`，约 0.8%。75.6% 对 0.8%，省的是显存，不是算力：分页不加速任何一次矩阵乘，它让同样大的显存装下更多并发序列。读实现时盯住 block 表，它把逻辑位置映射到物理块，同一序列的 KV 才能分散在不连续的块里。

自定义算子那一类，rms_norm 是个合适的样本：一个归一化算子，输入输出都在一层内，没有跨层状态，三种写法（PyTorch C++ 扩展、AscendC、`torch_npu` 内置）都能装下。三种写法对应三条路：通用扩展、芯片原生、厂商内置。读它的目的是看清“定义—编译—注册—调用”这个闭环，第 26 章第 4 节画过同一条链路；搬到 Ascend 上，闭环的形状不变，变的是每一段由谁实现。

这个仓库的定位是示例，不是生产引擎。它自己的 bench 表标着“仅供参考”，模型清单只覆盖那 6 个架构，MoE 与 VL 不支持入图——这些标注不是缺点清单，是读它时该带的边界。把它当生产引擎读，会得出“Ascend 只能跑小模型”的错误结论；把它当示例读，得到的是四类改动点和一份可以对照的代码。

对照第 27 章的 vllm-ascend，两者是同一套改动点的两种落点。vllm-ascend 是平台级插件，四类改动落在 `ops/`、`attention/`、`_cann_ops_custom/`、`compilation/` 这些目录里；nano-vllm-ascend 是手写的小引擎，改动直接落在引擎代码里。改动点的名字一样，落点不一样：先读小引擎，知道每一类改的是什么；再读插件，知道这些改动在大引擎里被放在哪。

### 第 2 节 · Triton kernel 迁移：从 GPU 到 Ascend 要改什么

迁移的第一步不是改 kernel，是把工具链的版本钉死。同一个 Ascend 环境里，CANN、torch_npu、triton_ascend 三个版本互相咬合；版本换一套，前面验证过的结论就可能失效。

| 环境 | 锚点 | 出处 |
|---|---|---|
| TritonAscendBench 的 NPU 环境 | CANN 8.5.0、torch_npu 2.8.0.post1、triton_ascend 3.2.0 | 仓库 `docker/Dockerfile.npu` |
| 第 27 章的 vllm-ascend 环境 | vllm-ascend v0.30.0 对应的一套，CANN 9.2.1 | 第 27 章的安装记录 |

两套 CANN 一个 8.5.0、一个 9.2.1。同一个仓库名、不同的版本组合，结论就可能不同；记录里写“这一步用的是哪套环境”，比写“Ascend 环境”有用。具体版本要求以各仓库当前文档为准，上面的数字是写作时的快照。

版本之外，迁移是一套循环：GPU 自查、NPU 实测、逐条记录。循环的必要性在于不兼容点没法靠读文档想全：文档能划出可能出问题的范围，具体哪一行报错，要跑起来才知道。每跑一圈，把“可能”变成“是”或“不是”，记录表里多一行带证据的结论。

GPU 自查在最前面，作用是分开两类 bug：kernel 本身的错，与迁移引入的错。先用 `tab run --backend gpu` 把第一类清零（第 28 章第 3 节讲过为什么顺序不能反），NPU 上报的错才只剩第二类。这与第 24 章先 standalone 验证、再接入引擎是同一条分层的规矩。

真正要逐条对照的是六类语义差异。它们不都出现在每个 kernel 里，但每一类都要先判断“这个 kernel 用不用得上”：

| 语义面 | GPU 上的默认 | Ascend 上要面对 | 典型失败 |
|---|---|---|---|
| 内存层级 | shared memory 加寄存器，tile 大小按 SM 的 shared memory 定 | UB、L1，矩阵乘的 A/B tile 还要进 L0A/L0B，累加器在 L0C | 按 GPU 经验取大 BLOCK，编译期报 `ub overflow` |
| 同步 | `__syncthreads` 或 warp 内锁步；跨核靠全局内存加原子 | 搬运与计算分属不同流水，靠事件同步；Triton-Ascend 一般由编译器插同步 | 用全局内存加原子做跨核累加；int64 与 fp64 原子不支持 |
| block 与核映射 | grid 里的 program 由硬件调度到几百个 SM | AI Core 只有几十个，Vector 核与 Cube 核约 2:1；超出的 program 分多轮下发 | 照搬几千个 program：能跑，但核启动开销吃掉收益；更大 grid 报 `coreDim` 超过 65535 |
| `tl.dot` | 走 Tensor Core，形状由 MMA 指令约束 | 走 Cube；M/N/K tile、dtype、累加器决定能不能落到 Cube 与 L0 | INT8 的 dot 在当前支持表里没有对应项；fp32 累加要显式 |
| dtype 与对齐 | int8 matmul 可用；对齐主要影响合并访存 | 尾轴对齐：纯 Vector 32 字节，CV 融合 512 字节；不对齐由硬件补齐；int64 与 fp64 支持有限 | 结果对但慢；cast 顺序变了，`max_abs` 超容限 |
| 动态 shape | 运行时 n 变化，autotune 按 key 选配置 | JIT 按 constexpr 特化；图编译按捕获时的形状编，形状变了要重编 | batch 或序列长度每步变，图反复重编；或 padding 成固定形状，浪费带宽 |

表里的容量与支持项随版本变化，具体数字与 API 支持状态以 Triton-Ascend 当前文档为准。

内存层级的差别决定 tile 怎么选。GPU 上 tile 大小按 shared memory 与寄存器预算；Ascend 上向量算子看 UB，矩阵乘看 L1 与 L0。UB（Unified Buffer，统一缓冲区：AI Core 上的片上暂存区，向量计算的输入输出先搬到这里）在 Atlas A2/A3 上是 192 KB，L1 是 512 KB；编译器默认开 multi-buffer（存算并行），同一块 tile 要留两份，可用容量按一半算。BLOCK 取大了，编译期就报 `ub overflow`，报错里会写需要多少 bit、可用多少 bit——这是最直接的一条线索。

同步的差别不在 API 名字，在“谁负责插同步”。GPU 上跨线程的依赖由 `__syncthreads` 一类原语显式表达；Ascend 上搬运与计算分属不同的流水：MTE2（内存搬运引擎 2）把数据从全局内存搬到 UB，MTE3 把结果搬回去，Vector 与 Cube 各自计算，彼此靠事件同步。Triton-Ascend 的编译器会在 lowering 时插同步，用户通常不用手写 barrier；`tl.debug_barrier` 是块内屏障，不是跨核同步。跨核依赖要么避免，要么用支持的内存序与原子操作表达。当前文档里 `atomic_add` 不支持 int64 与 fp64，也不支持“多核累加后保存中间结果”的用法——一段在 GPU 上靠原子汇总的归约，搬到 Ascend 上要改写成单核汇总，或换一种归约结构。

block 与核数的映射最容易照搬错。GPU 上 grid 里几千个 program 是常态，硬件会把它们铺到 SM 上；Ascend 的 AI Core 只有几十个，超出的 program 分多轮下发，每一轮都有核启动与初始化开销。Triton-Ascend 的建议是先把 grid 固定成物理核数，再在核内循环处理多个 tile：纯 Vector 算子按 Vector 核数分，含 `tl.dot` 的按 Cube 核数分。核数用 `triton.runtime.driver.active.utils.get_device_properties` 查 `num_vectorcore` 与 `num_aicore`，不要猜。二维 grid 在 NPU 上会被合并成一维，`(4, 5)` 与 `(20,)` 等价——写二维不会报错，但核怎么映射不由你控制。

`tl.dot` 的差别决定一个 kernel 能不能整体搬。GPU 上 dot 落到 Tensor Core，形状约束由 MMA 指令给；Ascend 上落到 Cube，A/B tile 要进 L0A/L0B，累加器在 L0C，M/N/K 的取法同时受 L0 容量与 UB 容量约束。当前 Triton-Ascend 的算子支持表里，dot 支持 fp16、bf16、fp32，int8 标不支持——第 22 章写的 INT8 dot 路径，搬到 Ascend 上不是改几行，而是这条 lowering 路径当前不存在。碰到这种情况，记录里要写清“哪一行、哪个 API、当前版本的什么限制”，不要写成“Ascend 不支持低精度计算”。

dtype 与对齐是两条慢路径。尾轴对齐要求纯 Vector 算子 32 字节，CV 融合（一个 kernel 里同时用 Cube 与 Vector 单元）算子 512 字节；不够的部分硬件自动补齐，结果不错，带宽多花。fp16 的 1024 列是 2048 字节，正好对齐；1000 列是 2000 字节，按 32 字节补齐到 2016 字节，每行多搬 16 字节，约 0.8%。对齐的账要按“每行多搬多少字节乘行数”算，形状越不规则，补齐的比例越高。另一种慢来自 dtype：int64 索引、fp64 计算在 Ascend 上支持有限，kernel 里用 int64 算偏移、用 fp64 累加，可能编译不过，也可能落到低效路径；能用 int32 的地方不要用 int64。

动态 shape 的差别在编译时机。GPU 上同一个 kernel 可以用运行时的 n 处理不同形状，autotune 按 key 选配置；Ascend 上 JIT 按 constexpr 特化，图编译按捕获时的形状编。LLM 推理的 batch 与序列长度每步都在变，图编译要么把形状 padding 成固定档位，要么反复重编。nano-vllm-ascend 的 MoE 与 VL 标“暂时不支持入图”，就是这条边界的一个例子：形状变化太大，进图的收益抵不过重编的成本。

循环里的记录只有四列：现象（报错原文）、定位（kernel 里的哪一行、哪个 API）、改法、重测结果。四列缺一列，这一行就不成立：没有报错原文，事后对不上号；没有行号与 API，改法无从谈起；没有重测结果，改法只是猜测。改完一个测一个，不要攒一堆改动再测——攒着测，失败时不知道是哪一个改动引起的。

把这六类差异带到 L2_04 上，先看参考 kernel 的四个特征：二维 grid、mask、fp32 里算 SiLU、cast 回输入精度再乘。四个特征分别对应核映射、对齐与边界、dtype、数值语义；`tl.dot` 与跨核同步在这份 kernel 里不出现——不是它们不重要，是这个 kernel 用不到。判断“哪些差异适用”，本身就是迁移的一部分。

六类差异加三步循环，是第 5 节清单的底稿。清单不是把六类抄一遍，而是把每一类变成一句能用“是/否”回答的检查项。

### 第 3 节 · 统一算子库：FlagGems 如何让一份 kernel 跑多家芯片

先分清三个东西。第 25 章第 4 节已经把关系画过：FlagGems 是算子库，FlagCX 是通信库，vllm-plugin-FL（把两者接进 vLLM 的插件）是把统一层接进引擎的那一环。这一节只讲 FlagGems 的机制。FlagGems 里的算子用 Triton 写，注册进 PyTorch 的 ATen 分发（PyTorch 的算子分发层），模型代码不用改就能用到；它不做编译器的活，编译发生在每个后端自己的 Triton 编译器里。

一份源码怎么变成多家芯片上的实现？分两段。第一段是写：一个算子只有一份 Triton 源码，描述“算什么”，不写“在哪块芯片上怎么算”。第二段是编译：运行时按当前平台选后端，由该后端的 Triton 编译器把同一份源码 lowering 成目标芯片的二进制。源码一份，二进制多份；多出来的那些由编译器生成，不需要人维护。

第二段之后是分发。vllm-plugin-FL 的分发链是三级：先找 FlagGems 的统一实现，没有再找厂商实现，再没有回退到 PyTorch 参考实现，自动 fallback。启动时先识别芯片、选后端，三级都按这个后端查。

文档列的平台是 12 家芯片：NVIDIA、Ascend、T-Head、天数智芯、沐曦、摩尔线程、清微智能、海光、曦望、燧原、昆仑芯、阿里巴巴 PPU，另外支持 ARM64 CPU 上的纯 CPU 推理。名单与数量以当前文档为准，平台增删比芯片型号变化快。

人力账可以这样算：手写路线，12 家芯片乘 1 个算子等于 12 份要人维护的实现；统一路线，1 份源码在 12 个后端上各编译一次，仍然只有 1 份要人维护的源码。两边都叫“12 份实现”，性质不同：前者改一个 bug 要改 12 处，后者改源码一处、重新编译。这笔账省的是“每多一家芯片就多写一份代码”，不是把移植变成零成本——某个后端编不过、某个算子没有统一实现，仍然要人补。

FlagGems 能解决的是覆盖面与语义对齐：常见算子有一份多后端实现，注册进 ATen 之后 PyTorch 调用点不变，三级分发保证缺实现时还能跑。它不解决四件事：

| 不解决 | 为什么 |
|---|---|
| 跨算子自动融合 | 它提供算子级的融合实现（silu_and_mul 本身就是一个），但模型里相邻的两个 PyTorch 调用不会因为它自动合成一个 kernel |
| 调度 | 不管 batch 怎么组、stream 怎么排、核数怎么分，这些在引擎与运行时 |
| 图优化 | 不做图捕获、图级融合、内存规划；它 eager 模式就绪，不依赖 `torch.compile` |
| 单芯片峰值性能 | 一份写法要在 12 个后端跑对，取的是交集；要榨干单芯片，仍要原生算子或特化 |

同一生态里还有 FlagTree：fork 自 Triton 的多后端编译器，每个后端一条受保护分支，Ascend 分支钉在 Triton 3.5（README 的分支表，2026-06-08 升级）。它是编译器，FlagGems 是算子库，两者不是一回事；它的版本也和 TritonAscendBench 用的 triton_ascend 3.2.0 不同，不能混用。

把分发链放进一次前向：模型调到 SiluAndMul，插件先问 FlagGems 有没有这个算子的统一实现；有就走统一实现，没有找厂商实现，再没有回退 PyTorch 参考实现。整个过程模型代码不动。代价是回退不报错：走到第三级时结果仍然对，速度是参考实现的速度，只有分发日志或 profiler 能看出来。统一层承诺覆盖面，不承诺性能下限。

对照第 2 节的手写迁移：统一路线省人力、牺牲特化；手写路线费人力、能榨单芯片性能。两条路不是替代关系：统一层决定移植的起点，手写移植决定天花板。一个算子该走哪条，看它是不是热点、目标芯片上有没有统一实现、维护人力有多少——这三个问题比“哪种技术更先进”有用。

29.4 要手写的 SiluAndMul，FlagGems 里也有一个统一实现。同一个算子，一条路是直接用库里的，一条路是自己写一遍、看它怎么落到 Ascend 上；两条路不冲突：先看库里的实现怎么组织，再自己写，读与写互相印证。

### 第 4 节 · 动手：移植第三部分的 SiluAndMul 到 TritonAscendBench 的 L2_04

移植第一步是确认语义，不是打开编辑器。SiluAndMul 在第 24 章写过：输入 `[M, 2D]`，前一半过 SiLU，再乘以后一半，输出 `[M, D]`。SiLU 的定义是 `silu(x) = x · sigmoid(x)`。手算几个值做基准：`silu(1) = 1/(1+e^-1) ≈ 0.7311`，`silu(2) ≈ 1.7616`，`silu(0) = 0`，`silu(-2) ≈ -0.2384`；再乘第二半，`0.7311×3 ≈ 2.1932`，`1.7616×0.5 ≈ 0.8808`。这些是算例，用来在对照 oracle（PyTorch 参考答案）之前先确认对照链路本身没写错。

TritonAscendBench 的 USAGE 里给了 L2_04 的参考解法，核心是一个 2D grid 的逐元素 kernel：

```python
# 摘自 TritonAscendBench 的 docs/USAGE.md；省略了 import torch / triton / triton.language as tl
@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, hidden_dim, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < hidden_dim
    x1 = tl.load(x_ptr + row * n_cols + col, mask=mask, other=0.0)
    x3 = tl.load(x_ptr + row * n_cols + hidden_dim + col, mask=mask, other=0.0)
    x1f = x1.to(tl.float32)
    silu = x1f * tl.sigmoid(x1f)
    out = x3 * silu.to(x3.dtype)
    tl.store(out_ptr + row * hidden_dim + col, out, mask=mask)


def npu_triton_impl(inputs: dict) -> dict:
    hs = inputs["hidden_states"]          # 形状 [M, 2D]
    bs, n_cols = hs.shape
    hidden_dim = n_cols // 2
    out = torch.empty((bs, hidden_dim), dtype=hs.dtype, device=hs.device)
    BLOCK = 128
    grid = (bs, triton.cdiv(hidden_dim, BLOCK))
    _silu_and_mul_kernel[grid](hs, out, hidden_dim, n_cols, BLOCK=BLOCK)
    return {"output": out}
```

这份参考 kernel 有四个特征，每一个都要在 Ascend 上重新确认：二维 grid（核映射）、mask（边界与对齐）、fp32 里算 SiLU（dtype 与精度）、cast 回输入精度再乘（数值语义）。它没有 `tl.dot`，没有跨核同步，没有原子操作——六类差异里 `tl.dot` 与同步不出现，其余四类都要看。先判断哪些差异适用，比把六类全查一遍省时间。

先算规模。取 M=512、D=1024、fp16：输入 `512×2048 = 1,048,576` 个元素，每个 2 字节，2 MiB；输出 `512×1024 = 524,288` 个元素，1 MiB。融合写法读 2 MiB、写 1 MiB，共 3 MiB；不融合要多写一次、多读一次中间结果，各 1 MiB，共 5 MiB。省下 2 MiB，`2/5 = 40%`。这是访存量的算例，不是端到端加速比。省的是中间结果的显存往返，不是计算量：这个算子每个元素只有几次到十几次浮点运算，算术强度在个位数 FLOP/字节，远低于第 1 章算例里的脊点，成本在搬运上。

grid 的账跟着形状走。参考 kernel 取 BLOCK=128，`hidden_dim=1024` 时每行 8 个列块，M=512 时 grid 是 `512×8 = 4096` 个 program。GPU 上这个数不算大；Ascend 的 AI Core 是几十个，4096 个 program 要么被后端折叠到物理核，要么分多轮下发。想控制核映射，就把 grid 改成 1D、固定成核数，在核内循环遍历这 4096 个 tile。改完之后每个核处理约 `4096/核数` 个 tile——核数用 `get_device_properties` 查，不要按“910B 有多少个核”的传闻写死。

BLOCK 取多大，由 UB 定。fp16 下 BLOCK 个元素是 `2×BLOCK` 字节，输入两半加输出是 3 份，fp32 中间量再占两份 `4×BLOCK`。BLOCK=4096 时粗算 56 KiB，双缓冲按两份算是 112 KiB，低于 Atlas A2/A3 的 192 KiB；BLOCK=8192 时同样算法是 224 KiB，超过 192 KiB，编译期可能直接报 `ub overflow`。这是上界估算，实际占用还要看编译器复用与对齐，但量级足够用来选 BLOCK：先按这个账选一个不溢出的值，再用 autotune 扫。

对齐的账也要算。`hidden_dim=1024` 时，一行 1024 个 fp16 是 2048 字节，32 字节的整数倍；`hidden_dim=1000` 时是 2000 字节，按 32 字节补齐到 2016 字节，每行多搬 16 字节，约 0.8%。形状越不规则，补齐越多；mask 挡住的是越界读写，挡不住补齐的带宽。`other` 取 0 只是让无效位置的计算不产生脏值，这些位置不会写回。

TritonAscendBench 的评测设计、101 个任务的分档、五个命令与判分口径，第 28 章第 3 节已经讲过；这里只做 L2_04 这一段：`tab init-solution L2_04_silu_and_mul` 生成 `solution.py`，填好 `npu_triton_impl` 之后用 `tab run L2_04_silu_and_mul --solution .../solution.py` 判分。入口必须是真 Triton kernel（`@triton.jit`、`tl.load`/`tl.store`），纯 PyTorch 实现会被静态检查拒绝；正确性在 CPU 上对照 oracle 判，性能要 910B 实测；任务编号、命令与判分字段以仓库当前版本为准。

按仓库当前状态，性能链路尚未端到端接通，所以这一节不报加速比，只报正确性与待补测的性能。正确性可以先在 CPU 上判，性能的数字留到有卡的那一天。

无卡时跑到哪一步为止？`tab run --backend gpu` 做 CUDA 自查；默认后端（NPU）在没有 `torch_npu` 时放不了设备，结果标 `npu_antihack_unverified`，记录里原样保留。那一节解释过这个标记的含义，这里只把它当记录纪律：它不是通过，是“没验证过 NPU”；没有 NPU 时也不会静默回退到 GPU，所以标记可信。

无卡时，本节要交的是 GPU 自查结果加一份逐行移植计划。计划按 kernel 的每一行写“可能有什么问题、准备怎么验证”，不写“这里一定有问题”或“这里肯定没问题”：

| kernel 里的位置 | 可能的问题 | 怎么验证 |
|---|---|---|
| 二维 grid `(bs, cdiv(D, BLOCK))` | 4096 个 program 在几十个核上分多轮，核映射不可控 | 改成 1D grid 加核内循环，有卡时对比两种写法的设备侧耗时 |
| `mask = col < hidden_dim` | D 不整除 BLOCK 时尾块越界或补齐 | 用 D=1000、1023、1024 三组形状对 oracle |
| `x1.to(tl.float32)` 与 `tl.sigmoid` | sigmoid 的 lowering 与精度行为不一致 | 用 0、±1、±2 和 ±80 的输入对 oracle 的 `max_abs` |
| `silu.to(x3.dtype)` 再乘 | cast 顺序或舍入与 oracle 不一致 | 对比“先 cast 再乘”与“fp32 直接乘”的 `max_abs`，确认前者在容限内 |
| `hidden_dim`、`n_cols` 作为运行时整数 | 特化规则改变时重编，或结果不变但变慢 | 用 M=1 与 M=512 两组形状各跑一次，看编译次数与结果 |

每行都只写“可能有什么问题”和“准备怎么验证”，因为没卡就没有结论。有卡之后跑一次、记一行，把“可能”改成“是”或“不是”，把验证改成带重测结果的记录。记录表头与第 2 节相同：现象、定位、改法、重测结果。

这一节要交的不是“移植成功”，是一份能复核的记录：语义怎么对、规模怎么算、每一步验证了什么、哪一步还没验证。移植记录里最容易被跳过的是最后一项，而它决定记录可不可信。

### 第 5 节 · 小结：移植的通用检查清单

把第 2 节的六类差异和第 4 节的记录合起来，得到七类检查项。每一类都要能用一个“是/否”回答，答不上来就是还没验证。

| 检查项 | 通过标准 | 不通过时先看什么 |
|---|---|---|
| 语义等价 | 输入输出形状、dtype、cast 顺序与 oracle 一致；边界形状也一致 | 先看契约与 cast 位置，再看 mask |
| 内存层级 | 单 tile 的 UB 占用（双缓冲按两份）不超上限；dot 的 tile 进得了 L1 与 L0 | 先看 BLOCK 与中间张量数量，再考虑关 multi-buffer |
| 同步 | 没有靠全局内存加原子做跨核汇总；块内 barrier 没被当成跨核同步 | 先看是否有多个核写同一地址，再看原子 dtype 是否支持 |
| 对齐 | 尾轴按 32 字节（纯 Vector）或 512 字节（CV）对齐；补齐的带宽算过 | 先看 D 是否整除 BLOCK、尾块补了多少字节 |
| 边界 | D 不整除 BLOCK、M=1、空张量、超大 N 都有对应测试 | 先看 mask、grid 上限与 `coreDim` |
| 精度 | 容限类内 `max_abs` 达标；fp32 计算与累加；cast 顺序不变 | 先看 exp 与 sigmoid 的输入范围、cast 位置 |
| 性能回归 | 同形状、同 dtype、同 warmup 对比基线；没有基线的写“待补测” | 先看 grid 是否按核数、BLOCK 是否吃满 UB、有没有标量访存 |

七项里，语义与精度管结果对不对；内存、同步、对齐、边界管会不会编译失败、跑出越界或错值；性能回归管能不能交付。

每一项都要有证据。语义等价的证据是对 oracle 的逐形状结果；内存层级的证据是编译日志（有没有 `ub overflow`）与 UB 占用估算；同步的证据是“没有跨核依赖”的代码阅读，不是“没报错”；对齐与边界的证据是边界形状的测试；精度的证据是 `max_abs` 与容限类；性能的证据是同条件基线，没基线就写待补测。

清单之外还有两条记录纪律。第一，版本锚点写清：CANN、torch_npu、triton_ascend 各是哪一版，来自哪套环境（TritonAscendBench 的 CANN 8.5.0 那一套，还是 vllm-ascend 的 9.2.1 那一套）。第二，没有 NPU 的结果保留 `npu_antihack_unverified`，不要改写成“通过”；失败的尝试也要留在记录里，它告诉后人哪条路走不通。

这份清单的用法是逐项打勾，不是读完就算。七项里任何一项空着，移植就没有收口；空着并标“待补测”，比填一个没验证的“通过”有用。移植的工作量不在把 API 名字换掉，而在把代码里那些“默认这里是 CUDA”的假设逐条翻出来、逐条给出证据——清单只是把这些假设排成一行，方便一条一条核。

**延伸**：[nano-vllm-ascend](https://github.com/linzm1007/nano-vllm-ascend)、[TritonAscendBench](https://github.com/Gxj230958/TritonAscendBench)（README 与 `docs/USAGE.md`）、[Triton-Ascend 迁移指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md)、[FlagGems](https://github.com/FlagOpen/FlagGems) 与 [FlagTree](https://github.com/flagos-ai/FlagTree)、[vllm-plugin-FL 文档](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)。

## 第 30 章 · 精度回归与调优报告

**本章目标。** 同一模型、同一评测集在 NVIDIA 与 Ascend 上各跑一遍，量出精度与性能差异，把差异归因到具体算子，按「某年某月某版本实测」的格式写成一份可复现的报告。

同一模型、同一评测集，在两边各跑一遍，会得到两组数字。数字摆在一起，问题才刚开始：准确率差 0.8 个百分点算不算差？首 token 延迟在并发 8 时差 40 毫秒，是芯片慢、调度差，还是权重精度不同？三个月后这套数字还作不作数？

这份报告要回答四件事：两边的精度差多少，差异有没有超出样本量能解释的范围；同一负载下首 token 延迟、token 间延迟与吞吐差多少，差在哪一档并发；差异能归因到哪个算子或哪条路径，证据是什么；结论在哪些版本与条件下成立，什么时候需要重测。要让答案站得住，需要四个条件：两边各有能跑同一模型的软硬件环境；评测集、模型、tokenizer 与采样参数在两边完全一致；版本、命令与原始结果都留下来；每个差异有 profiler 读数或对照实验撑着。缺一边环境时，报告框架、条件表与命令可以先写完，实测格子写「待补测」——待补测不是失败，是尚未取得证据的状态。

### 精度回归：同一评测集，在 NVIDIA 与 Ascend 上各跑一遍

精度回归是回归测试，不是统计回归：换一套软硬件之后，重新跑同一组评测，确认输出质量变了多少。它的前提是控制变量：同一评测集（同一版本、同一 prompt 模板、同一判分口径）、同一模型（同一权重来源、同一量化方式）、同一 tokenizer、同一采样参数（temperature=0 或固定种子）、同一 max_tokens，只换芯片与对应的服务。任何一条变了，表里的差异就同时混进两个变量：评测集换了，0.8 个百分点分不清是芯片的差别还是题目的差别；一边 FP16（16 位浮点）、一边 INT8（8 位整数），量出来的是量化的差别，芯片的差别被盖住。量化方式确实不同时，表头写清「FP16 vs INT8」，结论改写成「各自默认量化下的对比」，不能写成「芯片对比」。

tokenizer 也在清单里：同一个模型名不等于同一个 tokenizer，切分不同，prompt 的 token 数、截断位置与判分口径都会变。max_tokens 固定的是生成上限，实际长度按服务端返回的 `usage.completion_tokens` 记；服务端不返回 usage 时，脚本按字符数估算并在 JSON 里标 `tokens_estimated`，这种结果不能和 usage 口径混着比。上限不同，被截断的样本比例不同，accuracy 与 F1 就不可比。

指标按评测集的定义取，报告里写清指标名、样本数与判分脚本。常用的有四个：准确率（accuracy，答对的样本占总样本的比例）、精确匹配（Exact Match，EM，生成答案与参考答案完全一致的样本比例）、F1（F1 score，精确率与召回率的调和平均，衡量答案 token 与参考答案的重叠）、困惑度（Perplexity，PPL，模型对文本的平均负对数似然的指数，越低越好）。准确率是精确匹配还是宽容匹配（大小写、空白、标点算不算错），写进表头或脚注；两边的判分脚本不同，数字差的是判分，不是芯片。

样本量决定这张表能分辨多细的差异。1000 条样本里，NVIDIA 答对 850 条、Ascend 答对 842 条，准确率分别是 85.0% 与 84.2%，差 0.8 个百分点，折算成样本是 8 条；换成 100 条样本，1 条就是 1 个百分点，0.8 个百分点连 1 条都不到。这组数是算例，不是实测，用来看分辨率：同样叫 0.8 个百分点，在 1000 条的表里是 8 条样本，在 100 条的表里是不到 1 条。

还可以再算一步。1000 条样本、准确率 85% 时，比例的标准误（standard error，样本比例偏离真值的典型幅度）约 `sqrt(0.85 × 0.15 / 1000) ≈ 0.011`，即 1.1 个百分点；保守地按独立样本估，差值的标准误约 `sqrt(2 × 0.85 × 0.15 / 1000) ≈ 1.6` 个百分点，95% 区间约 ±3.1 个百分点。0.8 个百分点的差距落在区间内，单看准确率不能判定哪边更好。样本越小，区间越宽：100 条样本时差值的标准误约 5.0 个百分点，95% 区间约 ±9.9 个百分点，0.8 个百分点在里面可以忽略。报告里写样本量，读表的人才能自己算这一步。

困惑度的差也可以换算：两边的平均负对数似然（Negative Log-Likelihood，NLL，模型给正确答案的负对数概率的平均值，越小越好）差 0.01，PPL 是 NLL 的指数，e^0.01 ≈ 1.01005，两边 PPL 的比值约 1.01，差 1% 左右。1% 的 PPL 差和 0.8 个百分点的准确率差是同一个量级：能测出来，但解释它要去第 3 节做归因，本节只负责量。

输出抽查是另一路证据：至少抽 5 条并排对照，看有没有系统性差异。5 条算不出 0.8 个百分点，它的用途是看「错的样子」——指标说差多少，样本说差在哪类题上。5 条里有 3 条是同一类问题（比如都需要多步推理），这个模式给归因指了一个方向；5 条错得各不相同，就什么也说明不了。5 条的用途是发现模式，不是证明差异：样本只有 5 条，任何比例都不足以单独下结论。抽查要分层：1000 条里 800 条是选择题、200 条是开放问答时，随机抽 5 条全是选择题的概率是 0.8^5 ≈ 0.328，开放问答可能一条都抽不到；按题型分层、每类至少抽几条，某类题的差异才有机会露面。抽法写进报告，别人按同一个抽法才能复现。

表格的列带版本号与日期，因为表里每个数字只在某个软件栈版本下成立：同一块芯片，CANN 换一个版本，算子实现或量化路径可能就变了。版本号是数字的锚，日期是保质期的起点，第 5 节会用到。条件允许时把两边的版本写进表头，完整版本清单按模板的环境表填。

| 评测项 | 指标与口径 | NVIDIA（版本 + 日期） | Ascend（版本 + 日期） | 差异 |
|---|---|---|---|---|
| 评测集 A（样本数 n） | accuracy / EM / F1，判分口径 | … | … | … |
| 困惑度代理集 | PPL（NLL 的指数） | … | … | … |
| 输出抽查（至少 5 条并排，分层抽） | 人眼对照 | … | … | 有/无系统性差异 |

表中的「…」等实测填，没测的写「待补测」。精度回归的结论是差异的去向，不是谁高谁低：比出高低只要一行数字，回归要的是这 0.8 个百分点落在哪类题、对应哪个算子。第 25 章讲插件机制时说过，模型代码不用为每块芯片改一版；评测集也一样，它不关心跑在哪块芯片上，只负责量，解释是人的事。缺一边环境时，先把方法与表格框架定下来，实测格子留空；框架本身可以被检查，别人能看出方法对不对，哪怕数字还没填。

### 性能对比：TTFT / ITL / 吞吐，用第一部分的压测脚本在两边各跑三级并发

三个指标量三件事。TTFT（Time to First Token，首 token 延迟）是第一个 token 出来的时间，背后是 prefill（预填充：把提示一次性算完、建立 KV 缓存）；ITL（Inter-Token Latency，token 间延迟）是之后相邻 token 的间隔，背后是 decode（解码：逐 token 生成）；吞吐是单位时间整机产出的 token 数。只报吞吐会漏掉交互体验：吞吐相同的两边，TTFT 可能差一倍；只报 TTFT 会漏掉越吐越慢：首 token 很快，ITL 一路涨，用户照样等。三个数都要记，缺一个画像就缺一块。

三个数可以互相换算。一个请求生成 L 个 token，总延迟约等于 TTFT + ITL × (L − 1)：第一个 token 花 TTFT，剩下 L − 1 个间隔每个花 ITL。单请求 token 速率是 L ÷ 总延迟；并发为 C、延迟不变时，吞吐约等于 C × L ÷ 总延迟。

算例：TTFT = 100 毫秒、ITL = 20 毫秒、L = 101，总延迟 = 100 + 20 × 100 = 2100 毫秒 = 2.1 秒，单请求速率 = 101 ÷ 2.1 ≈ 48.1 token/秒，并发 8 时吞吐 ≈ 8 × 48.1 ≈ 384.8 token/秒。这组数是算例，不是实测；它给出一条无争用基准线：并发 8 的吞吐远低于 384.8，说明延迟已经变了，争用发生了。

再看延迟翻倍的情形：ITL 从 20 毫秒涨到 40 毫秒、TTFT 不变，总延迟 = 100 + 40 × 100 = 4100 毫秒 = 4.1 秒，单请求速率 ≈ 24.6 token/秒，并发 8 的吞吐 ≈ 8 × 24.6 ≈ 196.8 token/秒。并发涨 8 倍，吞吐只涨 196.8 ÷ 48.1 ≈ 4.1 倍，差的那部分就是争用的代价。

测量条件先钉死：用第 4 章的 `scripts/bench_openai.py`，不在两边各写一套脚本；并发档位至少三级（1 / 8 / 32 是例子，按硬件容量选），每级请求数、prompt、max_tokens、temperature 两边相同；每档先预热（第一次跑混着编译与缓存预热的成本），再重复多轮；每一轮按第 11 章的口径报 p50 与 p95，需要看尾巴时加 p99，跨轮取中位数。20 个请求的 p95 基本由最慢的一两个请求决定，p99 基本就是最慢的那一个；样本量写进表里，别把 20 个请求的 p99 当成精确分位。实际生成长度按 `usage.completion_tokens` 记（读不到时脚本按字符数估算，JSON 里会标 `tokens_estimated`）：两边平均长度差得多，ITL 与吞吐的差异里就混进了长度差，先按长度归一或分长度段比较，再谈芯片。

版本写全：硬件型号加显存、驱动与固件、Python / PyTorch / TorchNPU（PyTorch 的昇腾适配）、CANN、vLLM 与 vllm-ascend，或对应的芯片插件，每项带核实日期。版本锚点见第 27 章的安装链，本章不写死具体版本号——版本一换，数字就过期，报告按「某年某月某版本实测」记录。条件缺一项，别人复现出来的曲线就对不上，对不上又分不清是系统变了还是条件变了。

表里加一列比值，统一写成 Ascend ÷ NVIDIA，并注明方向：TTFT 与 ITL 越大越慢，吞吐越大越快。比值要和绝对值一起读：TTFT 从 10 毫秒变 14 毫秒，比值 1.4，绝对差 4 毫秒，用户几乎感觉不到；从 100 毫秒变 140 毫秒，比值还是 1.4，绝对差 40 毫秒，交互体感已经不同。比值接近 1 也不等于两边一样：TTFT 比值 1.02、ITL 比值 1.30 的两边，首 token 差不多，decode 差一截，画像完全不同。

三级并发看的是曲线形状：曲线越陡，扩展性越好；越早变平，瓶颈越早到。只看并发 1 的一个点不能下结论：单点量的是单请求延迟，调度、批处理与 KV 管理都没被量到；只看高并发吞吐、不看 TTFT，同样会漏掉交互体验。如果并发 32 的吞吐反而低于并发 8，说明进了过载区，TTFT 被排队撑大，吞吐不升反降；这一档记「过载」与原因，不拿它当性能结论——第 11 章讲过，饱和与抢占点上的延迟混进了排队与重算，不是服务的正常表现。跑不下的档位写「跑不下 + 原因」，不要硬跑，也不要悄悄降档再填表。

| 并发 | 指标 | NVIDIA（版本 + 日期） | Ascend（版本 + 日期） | 比值（Ascend / NVIDIA） |
|---|---|---|---|---|
| 1 | TTFT p50 / p95、ITL p50 / p95、吞吐 | … | … | … |
| 8 | 同上 | … | … | … |
| 32 | 同上 | … | … | … |

单位写进表头：TTFT 与 ITL 用毫秒，吞吐用 token/秒。每档带成功数 / 请求数与实际平均生成长度；成功数不满，先查服务，不要拿残缺样本的 p95 当结论。缺一边环境时，方法、档位、条件表与表格框架先写好，实测格子写「待补测」。

### 差异分析：数字不同的原因——归因到具体算子

归因的终点是一个算子名，或者一面墙。精度差异问「输出为什么不一样」，源头在计算本身：算子精度、量化方案、舍入方式；性能差异问「速度为什么不一样」，源头在资源：算子跑在哪、搬了多少数据、调度排得怎么样。两类混在一起，最容易得到一句「国产芯片还有差距」——这句话没法指导下一步：差距在哪、改哪个算子、换哪个实现，一个都答不上来。

归因按五步走：现象、定位、机制、对照、边界。

**现象。** 写清哪个指标、哪档并发、差多少、在什么版本下。精度写清是 accuracy 差 0.8 个百分点、PPL 差 1%，还是抽查里某类题总错；性能写清是并发 8 的 TTFT p95 差 40 毫秒、ITL p95 差 30 毫秒，还是吞吐比值 1.4。没有现象记录，后面的步骤没有输入。

**定位。** 精度差异先查三处：量化方案的实现（scale 是缩放系数，按 per-channel 每通道一个，还是 per-token 每 token 一个；舍入是就近舍入还是截断）、低精度算子的实现（一边 FP16、一边 FP8 或 INT8）、库版本（同一个算子名，版本一变，kernel 可能就换了）。性能差异也先查三处：算子覆盖（operator coverage，模型所需算子中芯片侧已有可用实现的比例）——第 28 章的矩阵里有没有回退到 CPU 或低效实现；显存带宽有没有变；调度排得怎么样。第 28 章的算子清单与覆盖矩阵是底图：它告诉你每个算子占多少时间、由谁承接；第 3 章的 profiler 方法用来看时间线。曲线形状能缩小范围：ITL 随并发涨得快，先查 decode 侧的调度与显存管理；TTFT 涨得快，先查 prefill 侧的批处理。

**机制。** 四类机制各自对应不同的现象。

- 精度降级：量化 scale 粒度不同、舍入方向不同，误差分布就不同；FP8 与 FP16 的输出差一点是意料之中，差多少才需要解释。
- 算子实现：没有芯片侧实现时回退到 CPU 或 PyTorch 参考实现，能跑但慢；有实现也可能缺融合、只在特定形状走优化路径。
- KV 布局：block 大小、KV 缓存的数据类型、PagedAttention 的实现不同，长上下文下的显存占用与搬运量就不同。
- 调度：批处理策略、分块 prefill、最大并发数不同，高并发下的排队与 ITL 曲线就不同。

回退的代价可以估：某算子占端到端时间 10%，回退后慢 5 倍，新时间 = 0.9 + 0.1 × 5 = 1.40 倍，整体慢 40%；占比 5%、慢 5 倍是 1.20 倍；占比 20%、慢 3 倍也是 1.40 倍。占比小的算子回退，端到端可能没感觉；占比大的回退一点，端到端掉一大截。同一笔账的另一面是第 24 章的 Amdahl 定律：占比 1% 的算子无限优化也只省 1%，占比 10% 的算子回退 5 倍却能让整体慢 40%。搬运量同理：7B 模型 FP16 权重 14 GB、INT8 权重 7 GB，decode 若受带宽限制，权重精度一变，ITL 就跟着变。边界是权重只是搬运的一部分，KV 缓存也要搬；权重省一半不等于总搬运省一半，更不等于 ITL 省一半，变多少要实测。

**对照。** 一次只换一个变量：把可疑算子换成 PyTorch 参考实现、把量化换回 FP16、关掉前缀缓存或分块 prefill、把并发固定在同一档，看差异是否消失。差异消失，这个因素就被证实；差异还在，换下一个。候选多时用二分：8 个候选 8 → 4 → 2 → 1，三轮；16 个四轮；10 个 ceil(log2 10) = 4 轮，因为 2^3 = 8 不够、2^4 = 16 才够。每轮代价是一次重跑，候选集列得越准，轮数越少。

两种失败模式先认出来。一是候选集漏了真凶：二分法只在候选集里找，真凶不在，几轮下来差异还在，谁也定位不到；候选集要从第 28 章的矩阵、第 1 节的抽查（哪类题错）与第 2 节的曲线形状（TTFT 先涨还是 ITL 先涨）三路拼出来。二是一次换多个变量：差异消失，不知道是谁的功劳；差异没消失，也不知道是谁的问题。验证顺序按成本与可能性排：占比大的先查、实现明显不同的先查、版本对不上的先对版本。

**边界。** 一个解释只在特定模型、长度、版本与并发范围内成立。版本换了，结论过期；并发跨过拐点，机制可能从调度换成排队；上下文长度变了，KV 布局的影响可能放大。归因记录里写日期，日期标明结论的有效期；验证不了的写「待验证」，不要写成定论。「待验证」不是报告的污点，它标明了证据链断在哪、下一步补什么。

| 差异（现象） | 候选原因（2–3 个） | 验证方法（对照实验） | 结论（算子 / 墙 + 证据，日期） |
|---|---|---|---|
| … | … | … | …（验证不了的写「待验证」） |

候选原因写 2–3 个：只写 1 个等于先下结论再找证据，超过 3 个说明候选集还没收敛。模板的归因表只有四列，日期写进结论格；结论只写验证过的，证据写清是哪个对照实验、差了多少。墙要写明是哪一套：第 25 章的四面墙是生态、显存、通信、精度，第 1 章 Roofline 的墙是算力与访存，两套不要混用。归因记录同时是调优清单：每一行指向一个算子或一条路径，下一步改什么、预期端到端动多少，都从这一行出发。

### 写报告：按「某年某月某版本实测」的格式，把环境、版本、命令、结果写全

报告是证据的合集，写只是把证据按顺序摆好。这份报告是第四部分的可交付物：跨芯片对比报告（初稿）。模板在 [cross-chip-report-template.md](cross-chip-report-template.md)，一共 11 节：0 号报告头，加 1 到 10 共十节。初稿的标准是框架完整、已有数据已填、缺失项都写「待补测」并给出补测条件；初稿和终稿的差别不在厚度，在所有「待补测」是否填上、所有「待验证」是否有了结论。

| 节 | 要填什么 | 为什么 |
|---|---|---|
| 0 报告头 | 报告日期、报告人、模型、评测集、结论保质期 | 读者第一眼要判断这份报告还管不管用 |
| 1 环境 | 两边各一列：硬件型号加显存、驱动与固件、Python / PyTorch / TorchNPU（PyTorch 的昇腾适配）、CANN、vLLM / vllm-ascend、核实日期 | 跨芯片对比的变量不止芯片，驱动、框架、插件每一项都能让数字变几个百分点 |
| 2 模型与评测集 | 模型名称 + 权重来源 + 量化方式；评测集名称 + 版本 + 样本数；temperature / top_p / max_tokens 固定值 | 复现的前提；样本数决定表的分辨率 |
| 3 命令 | 两边启动命令全文（含并行度、最大并发数、显存比例）、压测命令全文（含并发档位与每级请求数） | 命令是「只换芯片与服务」的证据，少一个环境变量复现就断 |
| 4 精度对比 | 第 1 节的表，带版本号与日期 | 质量差异的证据 |
| 5 性能对比 | 第 2 节的表，三级并发 × TTFT / ITL / 吞吐 | 性能差异的证据 |
| 6 差异归因 | 第 3 节的归因记录，验证不了的保持「待验证」 | 从数字到机制的桥 |
| 7 算子覆盖矩阵摘要 | 第 28 章矩阵的 Top 10 状态（原生 / 插件 / 回退 / 待核实）与核实日期 | 定位回退与实现位置 |
| 8 移植记录摘要 | 第 29 章记录的不兼容点：现象 + 定位 + 改法 + 重测结果 | 解释实现差异从哪来 |
| 9 结论 | 有证据的结论（数据 + 归因）与待补测项（缺失 + 下一步） | 只写证据；计划线写清补测的触发条件 |
| 10 保质期说明 | 会过期的结论与不会过期的方法论，分开列 | 报告的使用说明，第 5 节填 |

第 7、8 节是引用，不是重写：矩阵与记录的正本在第 28、29 章，报告里放摘要加指向；正本改了，摘要要跟着改。报告的读者不只有自己：三个月后的自己、组里同事、下一批做移植的人；写给接手的人，要把「为什么这么测」写进去，否则对方只会照抄命令，命令一出问题就不知道怎么调。报告头的结论保质期写「本报告基于某年某月某版本，过期需重测」——这一句告诉读者数字的有效期，也告诉复现的人该对齐哪套版本。模板有两处要补。模型与评测集一栏没有 tokenizer 与 prompt 模板的位置，填的时候补上，或者把 prompt 全文贴进命令一栏；报告头没有版本号一栏，补一行：初稿记 v0.1，补测一次升一版，并在报告头记一笔变更——改了什么、为什么改、哪天改的。没有版本号，两份报告摆在一起不知道哪份新；没有变更记录，新版的数字从哪来没人说得清。

版本写法是「某年某月某版本实测」：写具体版本号加核实日期，不写「最新版」——「最新」只是一个时间点，三个月后没人知道它指哪一天。核实日期写到天，版本号是静态的，核实日期说明这个号是亲手确认过的，不是从文档里抄的。本章不写死具体版本号，版本锚点见第 27 章；换版本就是换实验条件，数字要重测。

可复现的标准是同一量级，不是逐位相同：调度抖动、系统负载都会让数字浮动几个百分点。报告说 TTFT 是 100 毫秒级，复现出来也是 100 毫秒级，算复现；复现出 300 毫秒级，说明有变量没钉死——版本、命令、采样参数，按这三处回查。

写报告的顺序是：先填有数据的章节，再写归因，最后写结论。结论最后写，因为它依赖前面的数据与归因；先写结论再填数据，数据不好看就会想改结论。写完自查三问：命令能复现吗、版本都写全了吗、每个结论都有证据吗。

### 小结：这份报告里哪些结论会随版本过期、哪些方法论不会

一条结论只要带着具体版本号、具体数字或具体覆盖状态，就属于会过期的一类；只描述判据、步骤与记录格式的，不会过期。第二条规则：换一块芯片后，只需重填数字、步骤不动的，是方法论；必须重新测、重新归因才能得到的，是结论。第三条规则：报告里的每个数字必须能追到版本、日期、命令与原始结果；追不到的，不是结论。

会过期的结论：

- 精度数字：accuracy、EM、F1、PPL，以及输出抽查里发现的系统性差异。CANN、vllm-ascend、PyTorch 任一版本变化，算子实现或量化路径都可能变。
- 性能数字：TTFT、ITL、吞吐、曲线形状与拐点位置。调度策略、算子库、驱动都会改这些数。
- 覆盖状态：原生实现、插件实现、回退、待核实，以及实现位置。今天的原生实现，下个版本可能换路径；今天查不到的算子，下个版本可能补了插件。
- 环境与命令：版本清单、安装步骤、启动参数；换版本就要重跑，旧命令只对旧版本负责。

保守的过期时间点是下一次版本发布：只要驱动、CANN、引擎或插件升级，数字就要重测；没重测之前，旧数字只在旧版本下成立。引用前先看报告头的日期与环境表，版本一变，旧数字作废，重测后升一版，不把新旧数字放进同一张表。

不会过期的方法论：

- 控制变量清单：同一评测集、同一模型与权重来源、同一 tokenizer、同一采样参数、同一 max_tokens，只换芯片与服务。
- 精度与样本量的关系：1000 条里 0.8 个百分点是 8 条，100 条里是不到 1 条；样本量决定表的分辨率。
- 性能三指标与换算：TTFT + ITL × (L − 1)、单请求速率与吞吐的算式、三级并发与分位数口径。
- 归因五步与二分法：现象、定位、机制、对照、边界；一次只换一个变量，候选集 2–3 个，轮数 ceil(log2 n)。
- 记录格式：归因表（差异 / 候选原因 / 验证方法 / 结论）、移植记录（现象 / 定位 / 改法 / 重测结果）、报告 11 节与「待补测 / 待验证」标记。
- 第 25 章的四面墙与第 1 章的 Roofline：硬件参数会变，按墙定位、按算力与带宽判断瓶颈的思路不变。

本部分的检查清单有五项：软件栈分层图与插件机制、Ascend 跑通或写清待补测的原因、Top 3 耗时算子的覆盖矩阵、一次移植记录、一份按「某年某月某版本实测」格式写的报告初稿。打勾靠证据，不靠印象：每勾一项，报告里要能找到对应的实测数或验证记录；没勾的写清缺什么、什么条件下补。

第四部分到这里收口。它的标志不是「Ascend 跑得和 NVIDIA 一样快」，而是每个数字都知道怎么来、每个差异都知道归到哪、缺的都知道缺什么、什么时候补。下一块芯片（摩尔线程、海光、昆仑芯等，切入点见本部分末尾的「其他芯片的切入点」）来了，分层图要重画、覆盖矩阵要重填、数字要重测，但归因的步骤、记录的格式、二分法的轮数不用改。第 31 章会把这份初稿放进综合项目，补数据契约、证据等级与优化记录，条件仍然只有一条：每个数字追得到版本、日期、命令与原始结果。

**延伸**：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（固定 prompt、few-shot 与指标的评测框架，可用来核对精度回归的条件是否写全）、[MLPerf Inference](https://docs.mlcommons.org/inference/)（推理 benchmark 的条件、精度验收与复现规则）、[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)（算子级时间线与 kernel 时间，第 28 章用它导出算子清单）。

## 其他芯片的切入点

本模块以 Ascend 为主线，因为它的生态最成熟、资料最多。其他芯片的切入点供学完主线后扩展（状态以 [awesome-china-gpu](https://github.com/bytetora/awesome-china-gpu/blob/HEAD/README.md) 的最新追踪为准，不要照抄这里的版本信息）：

- **摩尔线程 MUSA**：vllm-musa 插件，另有 llama.cpp 官方支持
- **海光 DCU**：DTK（基于 ROCm），vLLM 容器镜像，迁移成本相对低
- **昆仑芯 XPU**：vllm-kunlun 插件
- **天数智芯 Iluvatar**：DeepSparkInference / vllm-iluvatar
- **壁仞 MetaX**：vllm-metax 插件
- **多芯片统一接入**：[vllm-plugin-FL](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)（FlagGems 统一算子库 + FlagCX 统一通信库）

## 本模块检查清单

- [ ] 能画出国产芯片软件栈的分层图，并说清插件机制
- [ ] 同一模型在 Ascend 上跑通过（或明确标注了待补测的原因）
- [ ] 有一份算子覆盖矩阵，指出 Top 3 耗时算子的芯片侧状态
- [ ] 走过一次小规模移植流程并留下了记录
- [ ] 有一份按「某年某月某版本实测」格式写的跨芯片对比报告（初稿）
