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

### 04 — Backward Op
Implements both the forward and backward pass for a broadcast Mul+ReLU kernel (subgradient gate for ReLU). Demonstrates the full workflow for writing custom gradient kernels in TileLang and Triton, applicable wherever fine-grained gradient control is needed.

### 05 — Reduce Sum
GPU implementation of row-wise reduction: `B[i] = sum(A[i,:], dim=1)`. Focuses on cross-thread data dependencies and the distinction between `T.Serial` (dependent iterations) and `T.Parallel` (independent iterations).

### 06 — Softmax
Numerically stable Softmax in two variants:
- **3-pass**: find row max → compute exp(x−max) → normalize. Beats Triton by ~24% on gfx1100 (config: `threads=256, BM=256`).
- **Online Softmax**: single-pass fused scan that maintains a running max and rescaled sum simultaneously, eliminating the separate max pass. Best config on gfx1100: `threads=512, BM=4096` (8 floats per thread per iteration). Motivates FlashAttention's IO-aware design.

### 07 — Scalar Flash Attention
A simplified FlashAttention implementation using element-wise `Q*K` in place of the full `QK^T` matrix multiply, while preserving the complete Online Softmax + tiled structure. Avoids materializing the N×N attention matrix, reducing memory complexity from O(N²) to O(N).

### 08 — Matrix Computation (GEMM)
Full progression from naive to hardware-accelerated GEMM:
- **Naive**: `alloc_fragment` (registers) + `T.Serial` K-loop
- **WMMA** (ROCm/RDNA3): uses `WMMAIntrinEmitter` to emit `v_wmma_f32_16x16x16_f16` instructions (warp-size=32). B must be pre-transposed to `(N×K)` layout. Best config: `brw=bcw=2, wrt=wct=64, chunk=64` → ~85 TFLOPS on RX 7900 XTX vs rocBLAS ~90 TFLOPS and Triton ~68 TFLOPS.

### 09 — Convolution (1D)
Single-channel and multi-channel 1D convolution with same padding. Key techniques: branch-free boundary handling (`T.if_then_else` instead of conditional branches, avoiding warp divergence) and 3D block grid design (batch × length × channel).

### 10 — Dequantized MatMul (W4A16)
INT4 weight-only quantization matrix multiply for LLM inference. Each `uint8` byte stores two INT4 values; the kernel unpacks nibbles on-the-fly inside the K-loop (low 4 bits for even columns, high 4 bits for odd columns), avoiding explicit FP16 expansion and saving ~4× weight memory and bandwidth.

## Benchmark Results (RX 7900 XTX / gfx1100)

All kernels were benchmarked on an **AMD Radeon RX 7900 XTX** (RDNA3, gfx1100, 24 GB GDDR6).

**Software**: TileLang 0.1.10+rocm · PyTorch 2.10.0+rocm7.2 · ROCm 7.2 · Triton (ROCm fork)

> Speedup is relative to the respective PyTorch and Triton baselines at the same problem size.

| # | Kernel | Best TileLang Config | vs PyTorch | vs Triton |
|---|--------|----------------------|:----------:|:---------:|
| 01 | Copy | `BLOCK_N=1024, threads=256` | **+19%** | **+31%** |
| 02 | Vector Add | `BLOCK_N=1024, threads=256` | **+18%** | **+7%** |
| 03 | Outer Vector Add | `BN=256, BM=64, threads=256` | **+78%** | **+22%** |
| 04 | Backward fwd | `BN=BM=128, threads=256, shared B` | **+39%** | **+50%** |
| 04 | Backward bwd | `BN=BM=128, threads=256, shared B` | **+71%** | **+39%** |
| 05 | Reduce Sum | `BN=2, BM=128, threads=128` | **+1%** | ≈ |
| 06 | Softmax (online) | `BM=4096, threads=512` | **+11%** | **+24%** |
| 07 | Scalar Flash Attn | `BB=8, BS=128, threads=256` | ≈ | ≈ |
| 08 | GEMM (WMMA) | `brw=bcw=2, wrt=wct=64, chunk=64` | **+52%** | **+26%** |
| 09 | Conv 1D | `BN=4, BL=64` | **+256%** | **+89%** |
| 10 | Dequant MM | `BM=BN=128, BK=32` | ≈ | **+53%** |

**gfx1100-specific constraints** that informed the configs above:
- `T.copy` requires `BM ≤ threads` to generate a vectorised loop
- `T.reduce_sum/max` requires `BN × BM = threads` for correct warp reduction
- WMMA uses `v_wmma_f32_16x16x16_f16` (RDNA3), **not** MFMA — use `WMMAIntrinEmitter`, not `MatrixCoreIntrinEmitter`
- warp size = 32 (RDNA3), not 64 (CDNA)

## Requirements

- Python 3.10+
- PyTorch (CUDA/ROCm)
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

