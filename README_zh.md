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

**软件版本**：TileLang 0.1.10+rocm · PyTorch 2.10.0+rocm7.2 · ROCm 7.2 · Triton（ROCm 分支）

> 加速比相对于同等问题规模下的 PyTorch 和 Triton 基准。

| 编号 | 内核 | TileLang 最优配置 | vs PyTorch | vs Triton |
|------|------|-------------------|:----------:|:---------:|
| 01 | Copy | `BLOCK_N=1024, threads=256` | **+19%** | **+31%** |
| 02 | Vector Add | `BLOCK_N=1024, threads=256` | **+18%** | **+7%** |
| 03 | Outer Vector Add | `BN=256, BM=64, threads=256` | **+78%** | **+22%** |
| 04 | Backward fwd | `BN=BM=128, threads=256, shared B` | **+39%** | **+50%** |
| 04 | Backward bwd | `BN=BM=128, threads=256, shared B` | **+71%** | **+39%** |
| 05 | Reduce Sum | `BN=2, BM=128, threads=128` | **+1%** | ≈ |
| 06 | Softmax（online） | `BM=4096, threads=512` | **+11%** | **+24%** |
| 07 | Scalar Flash Attn | `BB=8, BS=128, threads=256` | ≈ | ≈ |
| 08 | GEMM（WMMA） | `brw=bcw=2, wrt=wct=64, chunk=64` | **+52%** | **+26%** |
| 09 | Conv 1D | `BN=4, BL=64` | **+256%** | **+89%** |
| 10 | Dequant MM | `BM=BN=128, BK=32` | ≈ | **+53%** |

**gfx1100 关键约束**（影响上述配置选取）：
- `T.copy` 要求 `BM ≤ threads`，否则退化为标量加载
- `T.reduce_sum/max` 要求 `BN × BM = threads`，否则 warp 归约结果不正确
- WMMA 使用 `v_wmma_f32_16x16x16_f16`（RDNA3），**不是** MFMA——应用 `WMMAIntrinEmitter`，而非 `MatrixCoreIntrinEmitter`
- warp size = 32（RDNA3），不是 64（CDNA）

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


