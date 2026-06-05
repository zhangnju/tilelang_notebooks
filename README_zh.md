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

**iGPU coalescing 优化（gfx1151）**：`T.Parallel(BN, BM)` 大 `BN` 让每个线程跨行访问（stride = M = 8 KB），统一内存下 cache miss 严重。解决：**BN=1**（每个 block/program 处理 1 行），所有线程写同一行连续列，完全 coalesced。TileLang 和 Triton 均采用此自适应分块策略。效果：Triton 2.5 ms → **0.28 ms**，TileLang **0.27 ms**。

### 04 — Backward Op
以"广播乘 + ReLU"为例，同时实现前向和反向内核（次梯度 gate 控制）。展示 TileLang / Triton 手写自定义梯度的完整流程，适用于需要精细控制梯度计算的场景。

### 05 — Reduce Sum
归约运算的 GPU 实现。计算矩阵每行之和 `B[i] = sum(A[i,:], dim=1)`。重点讲解跨线程数据依赖的处理，以及 `T.Serial`（含依赖的串行循环）与 `T.Parallel`（无依赖并行循环）的区别。

### 06 — Softmax
数值稳定 Softmax 的两种实现：
- **3-pass**：求行最大值 → 计算 exp(x−max) → 归一化。gfx1100 上配置 `threads=256, BM=256`，比 Triton 快约 31%。
- **Online Softmax**：单遍融合扫描，同时维护 running max 和 rescaled sum，省去独立的 max pass。gfx1100 最优配置 `threads=512, BM=8192`（每线程每轮加载 16 个 float）；gfx1201 最优配置 `threads=512, BM=4096`（8 个 float）。gfx1151 在 [PR #2313](https://github.com/tile-ai/tilelang/pull/2313) 修复 HIPMath 向量类型降级后可使用 `BM=1024`（修复前受限于 256）。引出 FlashAttention IO-aware 设计的核心动机。

### 07 — Scalar Flash Attention
FlashAttention 思想的简化版实现：用逐元素乘积 `Q*K` 代替矩阵乘 `QK^T`，完整保留 Online Softmax + 分块计算结构，无需显式存储 N×N 注意力矩阵，内存复杂度从 O(N²) 降至 O(N)。

TileLang 根据硬件特性自适应选择算法：
- **gfx1100 / gfx1201**（独立显存，高带宽）：**3-pass** — 将 exp(QK−max) 写入 DRAM，Pass 3 读回。带宽充足，额外一遍的代价小于 2-pass 的寄存器压力。
- **gfx1151**（iGPU 统一内存，DRAM 慢）：**2-pass online** — Pass 1 将 Q,K 预热到 L2 缓存；Pass 2 从 L2 重新读取 QK（约比 DRAM 快 10 倍），省去中间结果的 DRAM 写+读。最优配置：`BB=1, BS=2048, TH=256`。效果：gfx1151 上 TileLang **+92% vs PyTorch，+61% vs Triton**。

### 08 — Matrix Computation（GEMM）
矩阵乘法从朴素到硬件加速的完整演进：
- **朴素版**：`alloc_fragment`（寄存器）+ `T.Serial` K 循环
- **WMMA 版**（ROCm/RDNA3+）：通过 `WMMAIntrinEmitter` 发射 `v_wmma_f32_16x16x16_f16` 指令（warp-size=32）。WMMA 要求 B operand 以转置布局提供（`b_transposed=True`）；可通过预先物理转置或在共享内存搬运阶段构造。gfx1100 最优配置 `wrt=wct=64, panel=10`，达到 **87.2 TFLOPS**（rocBLAS 93.0 TFLOPS，Triton 66.6 TFLOPS）。**gfx1201 上 TileLang WMMA 达到 122.6 TFLOPS，超越 rocBLAS（121.8 TFLOPS）**，Triton 为 52.7 TFLOPS。

### 09 — Convolution
单通道和多通道 1D 卷积（same padding）实现。重点：无分支边界处理（`T.if_then_else` 代替条件跳转，避免 Warp Divergence）、三维 Block 网格（batch × length × channel）的设计。

**参考精度说明**：`ref_conv1d_mc` 使用 `unfold + float32 matmul` 代替 `torch.conv1d`（MIOpen），避免算法差异导致的精度偏差（gfx1151 上 MIOpen 与直接累加之间最大误差约 0.015，超出 `atol=1e-2` 阈值）。Triton 和 TileLang 均使用直接 float32 累加，与 unfold+matmul 语义一致。

### 10 — Dequantized MatMul（W4A16）
LLM 推理场景的 INT4 权重量化矩阵乘。每个 uint8 字节存储两个 INT4 值，在 K 维循环内即时解包（低 4 位 / 高 4 位分别还原），避免显式展开为 FP16 B 矩阵，节省约 4× 权重内存和带宽。

## 性能测试结果（RX 7900 XTX / gfx1100）

所有内核在 **AMD Radeon RX 7900 XTX**（RDNA3 架构，gfx1100，96 CU，24 GB GDDR6）上完成测试。

**软件版本**：TileLang 0.1.10+rocm.gitfe7cda86 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> 延迟单位为毫秒（越低越好）。计算密集型内核（GEMM、Dequant MM）标注 TFLOPS（越高越好）。

| 编号 | 内核 | 配置 | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行★） | `BLOCK_N=2048, TH=256` | 0.0061ms | 0.0113ms | **0.0057ms** | **+7%** | **+97%** |
| 02 | Vector Add | `BN=1024, TH=256` | 0.0207ms | 0.0175ms | **0.0140ms** | **+48%** | **+25%** |
| 02 | Mul + ReLU（融合） | `BLOCK_N=1024` | 0.0318ms | 0.0166ms | **0.0160ms** | **+98%** | **+4%** |
| 03 | Outer Vector Add | `BN=512, BM=64, TH=128` | 0.1155ms | 0.0824ms | **0.0502ms** | **+130%** | **+64%** |
| 04 | Backward（前向） | `BN=512, BM=128, TH=256` | 0.3341ms | 0.3171ms | **0.1792ms** | **+86%** | **+77%** |
| 04 | Backward（反向） | `BN=BM=128` | 0.6854ms | 0.5857ms | **0.5004ms** | **+37%** | **+17%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | 0.2998ms | **0.2961ms** | **0.2952ms** | **+2%** | ≈ |
| 06 | Softmax（online） | `BM=8192, TH=512` | 0.7984ms | 0.8695ms | **0.6487ms** | **+23%** | **+34%** |
| 07 | Scalar Flash Attn | `TH=64, BS=1024` | 0.0798ms | **0.0596ms** | 0.0595ms | **+34%** | ≈ |
| 08 | GEMM（WMMA） | `wrt=wct=64, panel=10` | **1.4782ms**<br>**93.0 TFLOPS** | 2.0647ms<br>66.6 TFLOPS | 1.5769ms<br>87.2 TFLOPS | −6% | **+31%** |
| 09 | Conv 1D（单通道） | `BN=1, BL=256, TH=256` | 0.0145ms | 0.0089ms | **0.0062ms** | **+133%** | **+42%** |
| 09 | Conv 1D（多通道） | `BN=1, BL=32, BF=32` | 0.0369ms | 0.0118ms | **0.0064ms** | **+476%** | **+84%** |
| 10 | Dequant MM（W4A16） | `BM=256, BN=128, BK=32, TH=256` | 1.7735ms<br>77.5 TFLOPS | 2.7837ms<br>49.4 TFLOPS | **1.7173ms**<br>**80.0 TFLOPS** | **+3%** | **+62%** |


## 性能测试结果（Radeon 8060S / gfx1151）

所有内核在 **AMD Radeon 8060S**（RDNA3.5 iGPU，gfx1151，40 CU，128 GB 统一内存）上完成测试。

**软件版本**：TileLang 0.1.10+rocm.gitfe7cda86 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> 延迟单位为毫秒（越低越好）。计算密集型内核（GEMM、Dequant MM）标注 TFLOPS（越高越好）。

| 编号 | 内核 | 配置（gfx1151） | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行★） | `BLOCK_N=256, TH=128` | **0.0037ms** | 0.0160ms | 0.0039ms | −5% | **+310%** |
| 02 | Vector Add | `BN=2048, TH=256` | 0.0382ms | 0.0377ms | 0.0379ms | ≈ | ≈ |
| 02 | Mul + ReLU（融合） | `BLOCK_N=1024` | 0.0420ms ★ | 0.0351ms | **0.0275ms** | **+53%** | **+28%** |
| 03 | Outer Vector Add | `BN=1, BM=4096, TH=256`（单行） | 0.3017ms | 0.2966ms | **0.2789ms** | **+8%** | **+6%** |
| 04 | Backward（前向） | `BN=1, BM=4096, TH=256`（单行） | 1.1796ms | 0.6136ms | **0.5905ms** | **+100%** | **+4%** |
| 04 | Backward（反向） | `BN=1, BM=4096`（单行） | 2.5977ms | 0.9063ms | **0.7179ms** | **+262%** | **+26%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | 1.1262ms | 1.1351ms | **1.1156ms** | **+1%** | **+2%** |
| 06 | Softmax（online） | `BM=1024, TH=256` | 3.0127ms | 4.7458ms | **2.5555ms** | **+18%** | **+86%** |
| 07 | Scalar Flash Attn | `BB=1, BS=2048, TH=256`（2-pass） | 0.6159ms | 0.5170ms | **0.3207ms** | **+92%** | **+61%** |
| 08 | GEMM（WMMA） | `wrt=wct=64, panel=4` | **4.1245ms**<br>**33.3 TFLOPS** | 19.6792ms<br>7.0 TFLOPS | 4.8667ms<br>28.2 TFLOPS | −15% | **+304%** |
| 09 | Conv 1D（单通道） | `BN=8, BL=64, TH=256` | 0.0396ms ★ | 0.0105ms | **0.0037ms** | **+965%** | **+186%** |
| 09 | Conv 1D（多通道） | `BN=4, BL=16, BF=16` | 0.0192ms | 0.0105ms | **0.0089ms** | **+117%** | **+18%** |
| 10 | Dequant MM（W4A16） | `BM=BN=128, BK=32, TH=128` | 6.4409ms<br>21.3 TFLOPS | 7.2821ms<br>18.9 TFLOPS | **3.9470ms**<br>**34.8 TFLOPS** | **+63%** | **+84%** |

★ gfx1151 上 PyTorch Mul+ReLU 使用 `torch.compile` 融合为单个 Inductor kernel。
★ PyTorch 单通道卷积使用 `unfold+matmul` 代替 MIOpen conv1d（iGPU 上小 N 规模启动开销更低）。

**gfx1151（RDNA3.5 iGPU）特性：**
- 与 gfx1100/gfx1201 相同 WMMA ISA（`v_wmma_f32_16x16x16_f16`）和 warp_size=32，`WMMAIntrinEmitter` 无需修改
- 统一内存（~0.26 TB/s 峰值带宽）：CPU 与 GPU 共享同一内存总线，cache miss 代价远高于独立显卡
- **iGPU 关键优化原则——BN=1 行并行**：`T.Parallel(BN, BM)` / Triton 2D tile 大 `BN` 导致跨行访问（stride = M = 8 KB）→ cache miss。解决：`BN=1`（每块处理 1 行），所有线程写同一行连续列，完全 coalesced。已应用于 **03_outer** 和 **04_backward** 的 TileLang 和 Triton 实现
- `T.reduce_sum`：[PR #2313](https://github.com/tile-ai/tilelang/pull/2313) warpSize 修复后无 BM 上限，配置 `BN=1, BM=1024, TH=256` 带宽达 ~0.24 TB/s，与 PyTorch/Triton 持平
- **06_softmax**：HIPMath 向量类型修复后 BM=1024 正常工作，TileLang **+18% vs PyTorch，+86% vs Triton**
- **08_GEMM**：rocBLAS 在 gfx1151 上优于 TileLang（iGPU CU 数量限制）；TileLang 仍比 Triton 快 **+304%**
- **10_dequant_mm**：TileLang **+63% vs PyTorch**（34.8 vs 21.3 TFLOPS），融合 W4→FP16 解包与 GEMM 于共享内存
- TileLang 在各类内核上全面领先：**04_bwd**（+262%）、**07_flash**（+92%）、**09_conv 单通道**（+965%）、**09_conv 多通道**（+117%）、**10_dequant**（+63%）
- **01_copy** 在 N=512K 规模下：`T.copy BN=256, TH=128` → PyTorch 快 5%（iGPU 带宽瓶颈）；TileLang 比 Triton 快 **+310%**
- **02_vector_add**：`BN=2048, TH=256` — ≈ PyTorch 和 Triton（~0.038ms）。iGPU 统一内存带宽是三者共同上限，均已饱和。BN=2048（128-bit load）是 TileLang 最优配置；BN=1024（64-bit）明显更慢

## 性能测试结果（R9700 / gfx1201）

所有内核在 **AMD Radeon AI PRO R9700**（RDNA4，gfx1201，64 CU，32 GB，峰值带宽 ~0.58 TB/s）上完成测试。

**软件版本**：TileLang 0.1.10+rocm.gitc9f72459 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

notebook 在运行时自动检测 GPU 架构，并选择对应配置：
```python
arch = torch.cuda.get_device_properties(0).gcnArchName  # "gfx1100" 或 "gfx1201"
if arch.startswith("gfx1201"):
    ...  # R9700 配置
else:
    ...  # RX 7900 XTX 配置（原始）
```

> 延迟单位为毫秒（越低越好）。计算密集型内核（GEMM、Dequant MM）标注 TFLOPS（越高越好）。

| 编号 | 内核 | 配置（gfx1201） | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|------|------|----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy（多块并行★） | `BLOCK_N=2048, TH=128` | **0.0057ms** | 0.0075ms | 0.0049ms | **+16%** | **+53%** |
| 02 | Vector Add | `BN=2048, TH=128` | 0.0210ms | 0.0197ms | **0.0166ms** | **+27%** | **+19%** |
| 02 | Mul + ReLU（融合） | `BLOCK_N=1024` | 0.0287ms | **0.0169ms** | 0.0191ms | **+51%** | ≈ |
| 03 | Outer Vector Add | `BN=512, BM=256, TH=128` | 0.1547ms | 0.0904ms | **0.0428ms** | **+262%** | **+112%** |
| 04 | Backward（前向） | `BN=512, BM=256, TH=128` | 0.5130ms | 0.5366ms | **0.3073ms** | **+67%** | **+75%** |
| 04 | Backward（反向） | `BN=BM=128` | 1.1053ms | 0.8229ms | **0.5942ms** | **+86%** | **+38%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=64` | 0.4929ms | 0.4951ms | **0.4898ms** | **+1%** | **+1%** |
| 06 | Softmax（online） | `BM=4096, TH=512` | 1.0681ms | 1.6793ms | **1.0596ms** | **+1%** | **+58%** |
| 07 | Scalar Flash Attn | `TH=128, BS=1024` | 0.2044ms | 0.1217ms | **0.0815ms** | **+151%** | **+49%** |
| 08 | GEMM（WMMA） | `wrt=wct=64, panel=10` | 1.1284ms<br>**121.8 TFLOPS** | 2.6103ms<br>52.7 TFLOPS | **1.1209ms**<br>**122.6 TFLOPS** | **+1%** | **+133%** |
| 09 | Conv 1D（单通道） | `BN=4, BL=32, TH=256` | 0.0111ms | 0.0092ms | **0.0047ms** | **+138%** | **+97%** |
| 09 | Conv 1D（多通道） | `BN=4, BL=32, BF=16` | 0.0319ms | 0.0141ms | **0.0048ms** | **+563%** | **+193%** |
| 10 | Dequant MM（W4A16） | `BM=256, BN=128, BK=32, TH=256` | 1.7045ms<br>80.6 TFLOPS | 2.4969ms<br>55.0 TFLOPS | **1.4543ms**<br>**94.5 TFLOPS** | **+17%** | **+72%** |

**gfx1201 与 gfx1100 主要差异：**
- WMMA ISA（`v_wmma_f32_16x16x16_f16`）和 warp size = 32 完全相同——`WMMAIntrinEmitter` 无需修改
- **08_GEMM：TileLang WMMA 达到 122.6 TFLOPS，超越 rocBLAS（119.9 TFLOPS）**——RDNA4 WMMA 路径经 PR #2313 完整启用（warpSize 修复 + gfx12 注册）
- `01_copy`：`BLOCK_N=2048, TH=128` — 0.0049ms，**+16% vs PyTorch，+53% vs Triton**
- `02_vector_add`：`tl_add_gfx1201, BN=2048, TH=128` — **0.0166ms，+27% vs PyTorch，+19% vs Triton**（1.01 TB/s）；此前为 −3% vs PyTorch。优化：`T.alloc_fragment` + `T.copy` 替换 `T.Parallel` 直接索引 → 128-bit 向量化 load（`global_load_dwordx4`）
- `03_outer`：`BN=512, BM=256, TH=128` — **+262% vs PyTorch，+112% vs Triton**
- `07_flash_attn`：`TH=128, BS=1024` — **+151% vs PyTorch，+49% vs Triton**（3-pass 标量注意力）
- `09_conv`：单通道 **+138%**，多通道 **+563%** vs PyTorch（单通道参考为 MIOpen conv1d）
- `10_dequant`：**94.5 TFLOPS**（+17% vs PyTorch，+72% vs Triton）——gfx1201 最高计算吞吐受益

## 环境依赖

- Python 3.10+
- PyTorch（ROCm）
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
