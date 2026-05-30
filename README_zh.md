# TileLang Notebooks

一套系统化的 Tilelang 内核编程教学 notebook，通过对比 **TileLang**、**Triton** 和 **PyTorch** 三种框架的实现，从基础内存操作逐步进阶到 LLM 推理中的核心算子。

每个 notebook 均包含：问题背景、PyTorch 参考实现、Triton 实现、TileLang 实现、正确性验证、以及延迟/带宽/TFLOPS 的性能表格对比（含 vs PyTorch 和 vs Triton 的加速比）。


## 目录结构

```
tilelang_notebooks/
├── zh/          # 中文版
└── en/          # 英文版
```

两个目录下各有 10 个 notebook，内容完全对应。

## Notebook 列表

| 编号 | 主题 | 核心概念 |
|------|------|---------|
| 01 | **Copy**（数据拷贝） | Block / Thread 并行模型，内存带宽利用率 |
| 02 | **Vector Add**（向量加法与算子融合） | 元素级运算，Kernel Fusion，寄存器缓冲 |
| 03 | **Outer Vector Add**（外积向量加法） | 二维 Grid，2D 广播，`T.Parallel(BN, BM)` |
| 04 | **Backward Op**（反向传播算子） | 自定义前向 + 反向，链式法则，广播乘 + ReLU |
| 05 | **Reduce Sum**（归约求和） | 跨线程归约，Warp Shuffle，`T.reduce_sum` |
| 06 | **Softmax**（数值稳定 Softmax） | 数值稳定性，三遍扫描，`T.reduce_max` + `T.exp2` |
| 07 | **Scalar Flash Attention** | Online Softmax，IO-Aware 分块，FlashAttention 思想 |
| 08 | **Matrix Computation**（GEMM） | Tensor Core（MMA），共享内存，软件流水线（Pipelined） |
| 09 | **Convolution**（1D 卷积） | 边界处理，权重共享，单/多输出通道 |
| 10 | **Dequantized MatMul**（INT4 反量化矩阵乘） | W4A16 量化，INT4 打包解包，融合内核 |

## 各章详解

### 01 — Copy
GPU 编程入门。通过对同一份数据拷贝实现串行（1块×1线程）、多线程（1块×256线程）、多块并行三种版本，直观展示并行度对内存带宽利用率的影响。

### 02 — Vector Add & 算子融合
以向量加法为基础，进一步实现"乘法 + ReLU"融合算子。演示如何用 `T.alloc_fragment` 分配寄存器缓冲，将多次全局内存读写压缩为一次，避免 Memory-Bound 算子的带宽浪费。

### 03 — Outer Vector Add
介绍二维 GPU 内核网格。给定向量 `A[N]` 和 `B[M]`，计算矩阵 `C[N,M] = A[:,None] + B[None,:]`。是理解矩阵运算并行化（位置编码加法、偏置广播等）的基础。

**iGPU coalescing 优化（gfx1151）**：`T.Parallel(BN, BM)` 大 `BN` 让每个线程跨行访问（stride = M = 8 KB），统一内存下 cache miss 严重。解决：**BN=1**（每个 block/program 处理 1 行），所有线程写同一行连续列，完全 coalesced。TileLang 和 Triton 均采用此自适应分块策略。效果：Triton 2.5 ms → **0.29 ms**，TileLang **0.28 ms**。

### 04 — Backward Op
以"广播乘 + ReLU"为例，同时实现前向和反向内核（次梯度 gate 控制）。展示 TileLang / Triton 手写自定义梯度的完整流程，适用于需要精细控制梯度计算的场景。

### 05 — Reduce Sum
归约运算的 GPU 实现。计算矩阵每行之和 `B[i] = sum(A[i,:], dim=1)`。重点讲解跨线程数据依赖的处理，以及 `T.Serial`（含依赖的串行循环）与 `T.Parallel`（无依赖并行循环）的区别。

### 06 — Softmax
数值稳定 Softmax 的两种实现：
- **3-pass**：求行最大值 → 计算 exp(x−max) → 归一化。gfx1100 上配置 `threads=256, BM=256`，比 Triton 快约 24%。
- **Online Softmax**：单遍融合扫描，同时维护 running max 和 rescaled sum，省去独立的 max pass。gfx1100 最优配置 `threads=512, BM=4096`（每线程每轮加载 8 个 float）。引出 FlashAttention IO-aware 设计的核心动机。

### 07 — Scalar Flash Attention
FlashAttention 思想的简化版实现：用逐元素乘积 `Q*K` 代替矩阵乘 `QK^T`，完整保留 Online Softmax + 分块计算结构，无需显式存储 N×N 注意力矩阵，内存复杂度从 O(N²) 降至 O(N)。

TileLang 根据硬件特性自适应选择算法：
- **gfx1100 / gfx1201**（独立显存，高带宽）：**3-pass** — 将 exp(QK−max) 写入 DRAM，Pass 3 读回。带宽充足，额外一遍的代价小于 2-pass 的寄存器压力。
- **gfx1151**（iGPU 统一内存，DRAM 慢）：**2-pass online** — Pass 1 将 Q,K 预热到 L2 缓存；Pass 2 从 L2 重新读取 QK（约比 DRAM 快 10 倍），省去中间结果的 DRAM 写+读。效果：gfx1151 上 TileLang **+30% vs PyTorch，+20% vs Triton**。

gfx1100/gfx1201 与 Triton 的差距根源：07_attn 的 BS=512/1024 使用 T.Parallel 加载，该约束已通过 PR #2210 解除。

### 08 — Matrix Computation（GEMM）
矩阵乘法从朴素到硬件加速的完整演进：
- **朴素版**：`alloc_fragment`（寄存器）+ `T.Serial` K 循环
- **WMMA 版**（ROCm/RDNA3）：通过 `WMMAIntrinEmitter` 发射 `v_wmma_f32_16x16x16_f16` 指令（warp-size=32）。B 需预转置为 `(N×K)` 布局。最优配置 `brw=bcw=2, wrt=wct=64, chunk=64`，在 RX 7900 XTX 上达到约 85 TFLOPS（rocBLAS ~90 TFLOPS，Triton ~68 TFLOPS）。

### 09 — Convolution
单通道和多通道 1D 卷积（same padding）实现。重点：无分支边界处理（`T.if_then_else` 代替条件跳转，避免 Warp Divergence）、三维 Block 网格（batch × length × channel）的设计。

### 10 — Dequantized MatMul（W4A16）
LLM 推理场景的 INT4 权重量化矩阵乘。每个 uint8 字节存储两个 INT4 值，在 K 维循环内即时解包（低 4 位 / 高 4 位分别还原），避免显式展开为 FP16 B 矩阵，节省约 4× 权重内存和带宽。

## 性能测试结果（RX 7900 XTX / gfx1100）

所有内核在 **AMD Radeon RX 7900 XTX**（RDNA3 架构，gfx1100，24 GB GDDR6）上完成测试。

**软件版本**：TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> 延迟单位为毫秒（越低越好）。带宽单位 TB/s，计算吞吐单位 TFLOPS（越高越好）。

| 编号 | 内核 | 配置 | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行） | `BLOCK_N=1024` | 0.0070ms | 0.0071ms | **0.0054ms** | **+30%** | **+31%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0219ms | 0.0163ms | **0.0131ms** | **+67%** | **+24%** |
| 02 | Mul + ReLU（融合） | `BLOCK_N=1024` | 0.0231ms | **0.0121ms** | **0.0119ms** | **+94%** | ≈ |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1241ms | 0.0749ms | **0.0556ms** | **+123%** | **+35%** |
| 04 | Backward fwd | `BN=BM=128, shared B` | 0.3385ms | 0.3556ms | **0.2388ms** | **+42%** | **+49%** |
| 04 | Backward bwd | `BN=BM=128, shared B` | 0.6957ms | 0.6660ms | **0.4793ms** | **+45%** | **+39%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=128` | 0.3010ms | **0.2959ms** | 0.2980ms | ≈ | ≈ |
| 06 | Softmax（online） | `BM=4096, TH=512` | 0.7199ms | 0.8587ms | **0.6483ms** | **+11%** | **+32%** |
| 07 | Scalar Flash Attn | `BB=1, TH=64, BS=1024` | 0.0802ms | 0.0463ms | **0.0461ms** | **+74%** | ≈ |
| 08 | GEMM（WMMA） | `wrt=wct=64, panel=10` | **1.5145ms**<br>**90.7 TFLOPS** | 1.9526ms<br>70.4 TFLOPS | 1.5994ms<br>85.9 TFLOPS | −5% | **+22%** |
| 09 | Conv 1D（单通道） | `BN=4, BL=64` | 0.0227ms | 0.0091ms | **0.0054ms** | **+320%** | **+69%** |
| 09 | Conv 1D（多通道） | `BN=4, BL=32, BF=32` | 0.0228ms | 0.0093ms | **0.0044ms** | **+418%** | **+111%** |
| 10 | Dequant MM（W4A16） | `BM=BN=128, BK=32` | 1.7983ms<br>76.4 TFLOPS | 2.8004ms<br>49.1 TFLOPS | **1.7978ms**<br>**76.4 TFLOPS** | ≈ | **+56%** |



## 性能测试结果（Radeon 8060S / gfx1151）

所有内核在 **AMD Radeon 8060S**（RDNA3.5 iGPU，gfx1151，40 CU，128 GB 统一内存）上完成测试。

**软件版本**：TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> 延迟单位为毫秒（越低越好）。BW = 有效内存带宽（TB/s）。计算密集型内核标注 TFLOPS。  
> iGPU 统一内存带宽（~0.21 TB/s）远低于独立显卡，带宽瓶颈内核 PyTorch 有优势。
> `‡` 表示 PyTorch 基准不是公平的 GPU 对比（串行 Python 循环）。

| 编号 | 内核 | 配置（gfx1151） | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行） | `BLOCK_N=8192` | 0.0075ms | 0.0084ms | **0.0053ms** | **+42%** | **+58%** |
| 02 | Vector Add | `BLOCK_N=2048` | 0.0391ms | 0.0375ms | **0.0376ms** | **+4%** | ≈ |
| 02 | Mul + ReLU（融合） | `BLOCK_N=2048` | 0.0361ms ★ | 0.0373ms | 0.0377ms | ≈ | ≈ |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.3099ms | 0.2916ms | **0.2822ms** | **+10%** | **+3%** |
| 04 | Backward fwd | `BN=1, BM=2048` (1-row) | 1.1763ms | **0.6110ms** | **0.5938ms** | **+98%** | ≈ |
| 04 | Backward bwd | `BN=1, BM=2048` (1-row) | 2.5885ms | **0.9044ms** | **0.8792ms** | **+194%** | ≈ |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | **1.1152ms** | 1.1177ms | **1.1144ms** | ≈ | ≈ |
| 06 | Softmax（online） | `BM=256, TH=256` | **2.5430ms** | 4.4869ms | 2.8319ms | −10% | **+58%** |
| 07 | Scalar Flash Attn | `BB=1, BS=256` (2-pass) | 0.5821ms | 0.5076ms | **0.4233ms** | **+38%** | **+20%** |
| 08 | GEMM（WMMA） | `wrt=wct=32, panel=8` | **4.0485ms**<br>**33.9 TFLOPS** | 9.7002ms<br>14.2 TFLOPS | 5.8037ms<br>23.7 TFLOPS | −30% | **+67%** |
| 09 | Conv 1D（单通道） | `BN=4, BL=64` | 0.0349ms ★ | 0.0106ms | **0.0038ms** | **+818%** | **+179%** |
| 09 | Conv 1D（多通道） | `BN=4, BL=32, BF=32` | 0.0182ms | 0.0107ms | **0.0039ms** | **+367%** | **+174%** |
| 10 | Dequant MM（W4A16） | `BM=BN=128, BK=32` | 6.4980ms<br>21.2 TFLOPS | 19.2121ms<br>7.2 TFLOPS | **3.9170ms**<br>**35.1 TFLOPS** | **+66%** | **+390%** |

**gfx1151（RDNA3.5 iGPU）特性：**
- 与 gfx1100/gfx1201 相同 WMMA ISA（`v_wmma_f32_16x16x16_f16`）和 warp_size=32，`WMMAIntrinEmitter` 无需修改
- 统一内存（~0.21 TB/s 有效带宽）：CPU 与 GPU 共享同一内存总线，cache miss 代价远高于独立显卡
- **iGPU 关键优化原则——BN=1 行并行**：`T.Parallel(BN, BM)` / Triton 2D tile 大 `BN` 导致跨行访问（stride = M = 8 KB）→ cache miss。解决：`BN=1`（每块处理 1 行），所有线程写同一行连续列，完全 coalesced。已应用于 **03_outer** 和 **04_backward** 的 TileLang 和 Triton 实现
- `T.reduce_sum`：PR #2210 修复后无 BM 上限，配置更新为 `BN=1, BM=1024, TH=256`。旧配置 BM=128 有 50% 线程空闲（128 个工作项对应 256 条线程），带宽仅 0.13 TB/s。BM=1024 时带宽达 0.241 TB/s，与 PyTorch/Triton 持平
- TileLang 在计算密集型和 coalescing 友好型任务上表现突出：**03_outer**（+7% vs PyTorch，+2% vs Triton）、**04_fwd**（+98%，+2% vs Triton）、**04_bwd**（+193%，+3% vs Triton）、**07_attn**（+30%）、**09_conv**（+1337%）、**10_dequant_mm**（+68%）

## 性能测试结果（R9700 / gfx1201）

所有内核在 **AMD Radeon AI PRO R9700**（RDNA4，gfx1201，64 CU，32 GB，峰值带宽 ~0.58 TB/s）上完成测试。

**软件版本**：TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

notebook 在运行时自动检测 GPU 架构，并选择对应配置：
```python
arch = torch.cuda.get_device_properties(0).gcnArchName  # "gfx1100" 或 "gfx1201"
if arch.startswith("gfx1201"):
    ...  # R9700 配置
else:
    ...  # RX 7900 XTX 配置（原始）
```

> 延迟单位为毫秒（越低越好）。BW = 有效内存带宽（TB/s），计算密集型内核标注 TFLOPS（越高越好）。
> `†` 标记与 gfx1100 最优配置不同的条目。

| 编号 | 内核 | 配置（gfx1201） | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行） | `BLOCK_N=2048` † | 0.0089ms | 0.0088ms | **0.0053ms** | **+68%** | **+66%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0574ms | 0.0416ms | **0.0405ms** | **+42%** | ≈ |
| 02 | Mul + ReLU（融合） | `BLOCK_N=1024` | 0.0790ms | **0.0401ms** | 0.0443ms | **+78%** | −10% |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.3099ms | 0.2916ms | **0.2822ms** | **+10%** | **+3%** |
| 04 | Backward fwd | `BN=BM=128, shared B` | 0.5179ms | 0.6294ms | **0.3927ms** | **+32%** | **+60%** |
| 04 | Backward bwd | `BN=BM=128, shared B` | 1.1961ms | 0.9901ms | **0.6845ms** | **+75%** | **+45%** |
| 05 | Reduce Sum | `BN=1, BM=64, TH=128` † | 0.5266ms | **0.4981ms** | 0.5234ms | ≈ | −5% |
| 06 | Softmax（online） | `BM=4096, TH=512` | 1.1674ms | 1.7510ms | **1.2033ms** | ≈ | **+46%** |
| 07 | Scalar Flash Attn | `BB=1, TH=128, BS=1024` | 0.2051ms | 0.1321ms | **0.0930ms** | **+121%** | **+42%** |
| 08 | GEMM（WMMA） | `wrt=wct=64, panel=10` | 1.4262ms<br>96.4 TFLOPS | 1.6032ms<br>85.7 TFLOPS | **1.4174ms**<br>**97.0 TFLOPS** | ≈ | **+13%** |
| 09 | Conv 1D（单通道） | `BN=4, BL=64, TH=128` | 0.0226ms | 0.0114ms | **0.0047ms** | **+381%** | **+143%** |
| 09 | Conv 1D（多通道） | `BN=4, BL=32, BF=32` | 0.0683ms | 0.0117ms | **0.0072ms** | **+849%** | **+62%** |
| 10 | Dequant MM（W4A16） | `BM=BN=128, BK=32` | 2.1519ms<br>63.9 TFLOPS | 2.5908ms<br>53.0 TFLOPS | **2.0895ms**<br>**65.8 TFLOPS** | ≈ | **+24%** |

† 配置与 gfx1100 最优不同。  

**gfx1201 与 gfx1100 主要差异：**
- WMMA ISA（`v_wmma_f32_16x16x16_f16`）和 warp size = 32 完全相同——`WMMAIntrinEmitter` 无需修改
- `01_copy`：最优 `BLOCK_N=2048`（gfx1100 为 1024）——64 CU 用更大分块更充分填满
- `05_reduce`：`BN=1, BM=64, TH=128`——此配置为 PR #2210 前测量所得；修复后更大 BM 同样正确
- `T.reduce_max/sum` 作用于任意 `(BB, BS)` 分块时：三个 RDNA 架构均要求 `BS ≤ 128`——限制了 07_attn 的 tile 大小（Triton 无此约束，可用 `BLOCK_S=1024`）
- 更高峰值内存带宽（~0.58 TB/s vs ~0.96 TB/s）使所有带宽瓶颈内核受益

## 环境依赖

- Python 3.10+
- PyTorch（CUDA/ROCm）
- [TileLang](https://github.com/tile-ai/tilelang)
- [Triton](https://github.com/openai/triton)
- Jupyter Notebook / JupyterLab

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 学习路径建议

```
内存操作基础          归约与多遍扫描          矩阵计算与优化
01 Copy
02 Vector Add    →   05 Reduce Sum    →   08 GEMM
03 Outer Add         06 Softmax           10 Dequant MM
04 Backward Op       07 Flash Attn
                     09 Conv
```


