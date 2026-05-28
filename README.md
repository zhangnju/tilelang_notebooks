# TileLang Notebooks

A structured series of GPU kernel programming tutorials that compare **TileLang**, **Triton**, and **PyTorch** side by side, progressing from fundamental memory operations to advanced LLM inference kernels.

Each notebook covers: problem background, PyTorch reference implementation, Triton implementation, TileLang implementation, correctness validation, and visualized latency/throughput benchmarks.

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
Numerically stable Softmax via three-pass scan (find row max → compute exp → normalize). Motivates FlashAttention's core idea: fusing multi-pass scans into a streaming Online Softmax to eliminate redundant global memory traffic.

### 07 — Scalar Flash Attention
A simplified FlashAttention implementation using element-wise `Q*K` in place of the full `QK^T` matrix multiply, while preserving the complete Online Softmax + tiled structure. Avoids materializing the N×N attention matrix, reducing memory complexity from O(N²) to O(N).

### 08 — Matrix Computation (GEMM)
Full progression from naive to optimized GEMM:
- **Naive**: `alloc_fragment` (registers) + `T.Serial`
- **Optimized**: `alloc_shared` (shared memory) + `T.Pipelined(num_stages=3)` (software pipeline to hide memory latency)

TFLOPS benchmarks show the gap to cuBLAS and the impact of each optimization.

### 09 — Convolution (1D)
Single-channel and multi-channel 1D convolution with same padding. Key techniques: branch-free boundary handling (`T.if_then_else` instead of conditional branches, avoiding warp divergence) and 3D block grid design (batch × length × channel).

### 10 — Dequantized MatMul (W4A16)
INT4 weight-only quantization matrix multiply for LLM inference. Each `uint8` byte stores two INT4 values; the kernel unpacks nibbles on-the-fly inside the K-loop (low 4 bits for even columns, high 4 bits for odd columns), avoiding explicit FP16 expansion and saving ~4× weight memory and bandwidth.

## Requirements

- Python 3.10+
- PyTorch (CUDA/ROCm)
- [TileLang](https://github.com/tile-ai/tilelang)
- [Triton](https://github.com/openai/triton)
- Jupyter Notebook / JupyterLab
- matplotlib

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

An NVIDIA Ampere or newer GPU (A100 / H100) is recommended to fully exercise Tensor Core and TMA features.

## Suggested Learning Path

```
Memory Fundamentals        Reduction & Multi-pass         Matrix Compute & Optimization
01 Copy
02 Vector Add        -->   05 Reduce Sum          -->   08 GEMM
03 Outer Add               06 Softmax                   10 Dequant MM
04 Backward Op             07 Flash Attn
                           09 Conv
```

Reading in order provides the most coherent learning experience. If you already have Triton experience, jumping to notebook 05 or 08 is a reasonable starting point.
