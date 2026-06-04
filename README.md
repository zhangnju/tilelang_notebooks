# TileLang Notebooks

A structured series of Tilelang GPU kernel programming tutorials that compare **TileLang**, **Triton**, and **PyTorch** side by side, progressing from fundamental memory operations to advanced LLM inference kernels.

Each notebook covers: problem background, PyTorch reference implementation, Triton implementation, TileLang implementation, correctness validation, and a printed performance table (latency, bandwidth/TFLOPS, and speedup vs PyTorch and Triton).


> Chinese version: [README_zh.md](./README_zh.md)

## Repository Structure

```
tilelang_notebooks/
├── zh/          # Chinese notebooks
└── en/          # English notebooks
```

Both directories contain the same 10 notebooks with identical content.

## Notebook Index

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | **Copy** | Block/Thread parallelism, memory bandwidth utilization |
| 02 | **Vector Add & Kernel Fusion** | Element-wise ops, operator fusion, register buffers |
| 03 | **Outer Vector Add** | 2D kernel grid, broadcast, `T.Parallel(BN, BM)` |
| 04 | **Backward Op** | Custom forward + backward, chain rule, broadcast Mul+ReLU |
| 05 | **Reduce Sum** | Cross-thread reduction, warp shuffle, `T.reduce_sum` |
| 06 | **Softmax** | Numerical stability, three-pass scan, `T.reduce_max` + `T.exp2` |
| 07 | **Scalar Flash Attention** | Online Softmax, IO-aware tiling, FlashAttention principles |
| 08 | **Matrix Computation (GEMM)** | Tensor Core (MMA), shared memory, software pipelining |
| 09 | **Convolution (1D)** | Boundary handling, weight sharing, single/multi-channel |
| 10 | **Dequantized MatMul (W4A16)** | INT4 quantization, nibble unpacking, fused kernel |

## Notebook Descriptions

### 01 — Copy
Introduction to GPU programming. Implements data copy in three variants — serial (1 block × 1 thread), multi-threaded (1 block × 256 threads), and multi-block parallel — to illustrate how parallelism directly impacts memory bandwidth utilization.

### 02 — Vector Add & Kernel Fusion
Builds on vector addition to implement a fused Mul+ReLU kernel. Demonstrates how `T.alloc_fragment` allocates register buffers to consolidate multiple global memory round-trips into one, reducing bandwidth waste in memory-bound workloads.

### 03 — Outer Vector Add
Introduces 2D kernel grids. Given vectors `A[N]` and `B[M]`, computes matrix `C[N,M] = A[:,None] + B[None,:]`. Serves as the foundation for understanding 2D-parallel matrix operations such as positional encoding addition and bias broadcast.

**iGPU coalescing insight** (gfx1151): `T.Parallel(BN, BM)` with large `BN` assigns each thread elements spread across multiple rows (stride = M = 8 KB) — catastrophic for unified-memory cache. Fix: **BN=1** (one row per block/program), all threads write consecutive columns of the same row → fully coalesced. Both TileLang and Triton apply this arch-adaptive block shape. Result: Triton 2.5 ms → **0.28 ms**, TileLang **0.27 ms** on gfx1151.

### 04 — Backward Op
Implements both the forward and backward pass for a broadcast Mul+ReLU kernel (subgradient gate for ReLU). Demonstrates the full workflow for writing custom gradient kernels in TileLang and Triton, applicable wherever fine-grained gradient control is needed.

**iGPU optimisations** (gfx1151):
- **Triton**: switch from `BN=64, BM=64` to `BN=1, BM=2048` (same row-parallel principle as 03). Forward: 7.7 ms → **0.61 ms**; Backward: 11 ms → **0.91 ms**.
- **TileLang**: `BN=1, BM=2048`, no shared-B (B fits in L2 after first access). Forward: **0.59 ms**; Backward: **0.89 ms**.
- **PyTorch ref_bwd**: replaced `dC * B * (A*B>0).to(fp16)` (5 kernel launches on iGPU) with `dC.mul(B).mul_(A.mul(B).gt(0))` (3 launches via in-place ops) → 3.2 ms → **2.6 ms**.

### 05 — Reduce Sum
GPU implementation of row-wise reduction: `B[i] = sum(A[i,:], dim=1)`. Focuses on cross-thread data dependencies and the distinction between `T.Serial` (dependent iterations) and `T.Parallel` (independent iterations).

### 06 — Softmax
Numerically stable Softmax in two variants:
- **3-pass**: find row max → compute exp(x−max) → normalize. Beats Triton by ~31% on gfx1100 (config: `threads=256, BM=256`).
- **Online Softmax**: single-pass fused scan that maintains a running max and rescaled sum simultaneously. Best config on gfx1100: `threads=512, BM=8192` (16 floats per thread per iteration); gfx1201: `threads=512, BM=4096` (8 floats). On gfx1151: `threads=256, BM=1024` after [PR #2313](https://github.com/tile-ai/tilelang/pull/2313) fixed the HIPMath vector-dtype lowering (previously limited to BM=256). Motivates FlashAttention's IO-aware design.

### 07 — Scalar Flash Attention
A simplified FlashAttention implementation using element-wise `Q*K` (scalar product) in place of the full `QK^T` matrix multiply, while preserving the complete Online Softmax + tiled structure. Avoids materializing the N×N attention matrix, reducing memory complexity from O(N²) to O(N).

TileLang uses hardware-adaptive algorithm selection:
- **gfx1100 / gfx1201** (fast discrete memory): **3-pass** — write intermediate exp(QK−max) to DRAM, read back in Pass 3. DRAM bandwidth is high enough that the extra pass costs less than the register pressure of 2-pass.
- **gfx1151** (iGPU, unified memory, slow DRAM): **2-pass online** — Pass 1 warms L2 cache with Q,K; Pass 2 re-reads Q,K which often remain in L2 cache, reducing external memory traffic compared to the extra DRAM round-trip. Result: TileLang **+39% vs PyTorch, +25% vs Triton** on gfx1151.


### 08 — Matrix Computation (GEMM)
Full progression from naive to hardware-accelerated GEMM:
- **Naive**: `alloc_fragment` (registers) + `T.Serial` K-loop
- **WMMA** (ROCm/RDNA3+): uses `WMMAIntrinEmitter` to emit `v_wmma_f32_16x16x16_f16` instructions (warp-size=32). The WMMA emitter uses `b_transposed=True` to interpret B in transposed layout; this can be achieved either by physical transposition (`B.T.contiguous()`) or by staging B in transposed form during shared-memory loading. Best config on gfx1100: `wrt=wct=64, panel=10` → **87.2 TFLOPS** vs rocBLAS 90.8 TFLOPS and Triton 70.5 TFLOPS. On gfx1201: **122.6 TFLOPS**, surpassing rocBLAS (119.9 TFLOPS) and Triton (98.3 TFLOPS).

### 09 — Convolution (1D)
Single-channel and multi-channel 1D convolution with same padding. Key techniques: branch-free boundary handling (`T.if_then_else` instead of conditional branches, avoiding warp divergence) and 3D block grid design (batch × length × channel).

**Reference precision note**: `ref_conv1d_mc` uses `unfold + float32 matmul` instead of `torch.conv1d` (MIOpen) to avoid algorithm-dependent precision differences (~0.015 max error on gfx1151 that would fail the `atol=1e-2` check). Both Triton and TileLang kernels use direct float32 accumulation, matching the unfold+matmul semantics.

### 10 — Dequantized MatMul (W4A16)
INT4 weight-only quantization matrix multiply for LLM inference. Each `uint8` byte stores two INT4 values; the kernel unpacks nibbles on-the-fly inside the K-loop (low 4 bits for even columns, high 4 bits for odd columns), avoiding explicit FP16 expansion and saving ~4× weight memory and bandwidth.

## Benchmark Results (RX 7900 XTX / gfx1100)

All kernels were benchmarked on an **AMD Radeon RX 7900 XTX** (RDNA3, gfx1100, 96 CU, 24 GB GDDR6).

**Software**: TileLang 0.1.10+rocm.gitfe7cda86 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|--------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block★) | `BLOCK_N=1024, TH=128` | 0.0060ms | 0.0071ms | **0.0055ms** | **+9%** | **+29%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0209ms | 0.0168ms | **0.0145ms** | **+44%** | **+16%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0241ms | **0.0118ms** | **0.0116ms** | **+108%** | **+2%** |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1259ms | 0.0753ms | **0.0562ms** | **+124%** | **+34%** |
| 04 | Backward (fwd) | `BN=BM=128` | 0.3415ms | 0.3148ms | **0.2174ms** | **+57%** | **+45%** |
| 04 | Backward (bwd) | `BN=BM=128` | 0.6746ms | 0.6497ms | **0.5004ms** | **+35%** | **+30%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=128` | 0.3002ms | **0.2959ms** | 0.3032ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=8192, TH=512` | 0.7177ms | 0.8600ms | **0.6487ms** | **+11%** | **+32%** |
| 07 | Scalar Flash Attn | `BB=1, TH=64, BS=1024` | 0.0813ms | 0.0463ms | **0.0458ms** | **+77%** | **+1%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | **1.5137ms**<br>**90.8 TFLOPS** | 1.9506ms<br>70.5 TFLOPS | 1.5769ms<br>87.2 TFLOPS | −4% | **+24%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0228ms | 0.0092ms | **0.0061ms** | **+274%** | **+51%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0263ms | 0.0094ms | **0.0045ms** | **+484%** | **+109%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 1.7960ms<br>76.5 TFLOPS | 2.7837ms<br>49.4 TFLOPS | **1.7935ms**<br>**76.6 TFLOPS** | ≈ | **+55%** |



## Benchmark Results (Radeon 8060S / gfx1151)

All kernels were benchmarked on an **AMD Radeon 8060S** (RDNA3.5 iGPU, gfx1151, 40 CU, 128 GB unified memory).

**Software**: TileLang 0.1.10+rocm.gitfe7cda86 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config (gfx1151) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block★) | `BLOCK_N=1024, TH=256` | **0.0037ms** | 0.0083ms | 0.0045ms | −18% | **+84%** |
| 02 | Vector Add | `BLOCK_N=2048` | 0.0388ms | 0.0374ms | **0.0373ms** | ≈ | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=2048` | 0.0360ms ★ | 0.0372ms | 0.0372ms | ≈ | ≈ |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.2952ms | 0.2843ms | **0.2746ms** | **+7%** | **+4%** |
| 04 | Backward (fwd) | `BN=1, BM=2048` (1-row) | 1.1759ms | 0.6092ms | **0.5915ms** | **+99%** | **+3%** |
| 04 | Backward (bwd) | `BN=1, BM=2048` (1-row) | 2.5778ms | 0.9064ms | **0.8851ms** | **+191%** | **+2%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | **1.1164ms** | 1.1159ms | 1.1180ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=1024, TH=256` | **2.5469ms** | 4.4237ms | **2.5555ms** | ≈ | **+73%** |
| 07 | Scalar Flash Attn | `BB=1, BS=256` (2-pass) | 0.5800ms | 0.5198ms | **0.4162ms** | **+39%** | **+25%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=4` | **4.3431ms**<br>**31.6 TFLOPS** | 9.3689ms<br>14.7 TFLOPS | 4.8667ms<br>28.2 TFLOPS | −12% | **+92%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0348ms ★ | 0.0108ms | **0.0039ms** | **+792%** | **+177%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0200ms | 0.0109ms | **0.0040ms** | **+400%** | **+173%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 6.4889ms<br>21.2 TFLOPS | 7.2821ms<br>18.9 TFLOPS | **3.9591ms**<br>**34.7 TFLOPS** | **+64%** | **+84%** |

★ PyTorch Mul+ReLU on gfx1151 uses `torch.compile` to fuse mul+relu into a single Inductor kernel.
★ PyTorch single-ch conv uses `unfold+matmul` instead of MIOpen conv1d (lower launch overhead for small N).

**gfx1151 (RDNA3.5 iGPU) characteristics:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp_size=32 as gfx1100/gfx1201 — `WMMAIntrinEmitter` works unchanged
- Unified memory (spec peak ~0.26 TB/s): CPU and GPU share one memory bus; lower bandwidth amplifies the cost of inefficient access patterns compared to discrete GPUs
- **Key iGPU optimisation — BN=1 row-parallel**: `T.Parallel(BN, BM)` / Triton 2D tiles with large `BN` cause stride-M (8 KB) writes → cache miss. Fix: `BN=1` (one row per block/program), all threads write the same row's consecutive columns → coalesced. Applied to **03_outer** and **04_backward** for both TileLang and Triton
- `T.reduce_sum`: no BM limit after warpSize fix ([PR #2313](https://github.com/tile-ai/tilelang/pull/2313)). Config `BN=1, BM=1024, TH=256` saturates iGPU bandwidth at ~0.24 TB/s, matching PyTorch/Triton
- **06_softmax**: BM=1024 now works after HIPMath vector-dtype fix; matches PyTorch (≈0%). Old BM=256 gave −10%
- **08_GEMM**: rocBLAS outperforms TileLang on gfx1151 (fewer CUs limit WMMA occupancy); TileLang still beats Triton by **+92%**
- **10_dequant_mm**: TileLang **+64%** vs PyTorch (34.7 vs 21.2 TFLOPS). Fuses W4→FP16 unpack with GEMM in shared memory
- TileLang excels on compute-bound kernels: **04_bwd** (+206%), **07_attn** (+45%), **09_conv** (+775%), **10_dequant** (+70%)
- **01_copy** at N=8M: TileLang wins! `par-256, BN=1024` generates 64-bit loads → 8192 blocks saturate 40 CU → **+7% vs PyTorch, +1% vs Triton**

## Benchmark Results (R9700 / gfx1201)

All kernels were benchmarked on an **AMD Radeon AI PRO R9700** (RDNA4, gfx1201, 64 CU, 32 GB, ~0.58 TB/s peak BW).

**Software**: TileLang 0.1.10+rocm.gitc9f72459 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

The notebooks auto-detect the GPU architecture at runtime and select the appropriate config:
```python
arch = torch.cuda.get_device_properties(0).gcnArchName  # "gfx1100" or "gfx1201"
if arch.startswith("gfx1201"):
    ...  # R9700 config
else:
    ...  # RX 7900 XTX config (original)
```

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config (gfx1201) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block★) | `BLOCK_N=1024, TH=128` | 0.0047ms | 0.0140ms | **0.0041ms** | **+15%** | **+241%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0214ms | 0.0167ms | **0.0165ms** | **+30%** | **+1%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0301ms | **0.0166ms** | **0.0163ms** | **+85%** | **+2%** |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1522ms | 0.0886ms | **0.0678ms** | **+124%** | **+31%** |
| 04 | Backward (fwd) | `BN=BM=128` | 0.5110ms | 0.6283ms | **0.3781ms** | **+35%** | **+66%** |
| 04 | Backward (bwd) | `BN=BM=128` | 1.0937ms | 0.9134ms | **0.5942ms** | **+84%** | **+54%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=64` | 0.4925ms | **0.4948ms** | 0.5201ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=4096, TH=512` | 1.0686ms | 1.6693ms | **1.0596ms** | **+1%** | **+57%** |
| 07 | Scalar Flash Attn | `BB=1, TH=128, BS=1024` | 0.1924ms | 0.0796ms | **0.0789ms** | **+144%** | **+1%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | 1.1466ms<br>119.9 TFLOPS | 1.3983ms<br>98.3 TFLOPS | **1.1209ms**<br>**122.6 TFLOPS** | **+2%** | **+25%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0136ms | 0.0121ms | **0.0041ms** | **+232%** | **+195%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0338ms | 0.0144ms | **0.0048ms** | **+604%** | **+200%** |
| 10 | Dequant MM (W4A16) | `BM=256, BN=128, BK=32` | 1.6993ms<br>80.9 TFLOPS | 2.4969ms<br>55.0 TFLOPS | **1.4647ms**<br>**93.8 TFLOPS** | **+16%** | **+71%** |

**gfx1201 vs gfx1100 key differences:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp size = 32 — `WMMAIntrinEmitter` works unchanged
- **08_GEMM**: TileLang WMMA **122.6 TFLOPS surpasses rocBLAS** (119.9 TFLOPS) on gfx1201 — RDNA4 WMMA fully enabled by PR #2313 (warpSize fix + gfx12 registration)
- `01_copy`: optimal `BLOCK_N=1024, TH=128` (multi-block★) — 1.73 TB/s, +3% vs Triton, +37% vs PyTorch
- `05_reduce`: `BN=2, BM=128, TH=64` — 64-thread blocks maximise bandwidth on 64 CU
- `10_dequant`: **93.8 TFLOPS** (+16% vs PyTorch, +71% vs Triton) — fused W4 unpack benefits from gfx1201's higher compute throughput
- `09_conv`: TileLang **+232%** (single-ch) and **+604%** (multi-ch) vs PyTorch

## Requirements

- Python 3.10+
- PyTorch (ROCm)
- [TileLang](https://github.com/tile-ai/tilelang)
- [Triton](https://github.com/openai/triton)
- Jupyter Notebook / JupyterLab

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


## Suggested Learning Path

```
Memory Fundamentals        Reduction & Multi-pass         Matrix Compute & Optimization
01 Copy
02 Vector Add        -->   05 Reduce Sum          -->   08 GEMM
03 Outer Add               06 Softmax                   10 Dequant MM
04 Backward Op             07 Flash Attn
                           09 Conv
```
