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

**iGPU coalescing insight** (gfx1151): `T.Parallel(BN, BM)` with large `BN` assigns each thread elements spread across multiple rows (stride = M = 8 KB) — catastrophic for unified-memory cache. Fix: **BN=1** (one row per block/program), all threads write consecutive columns of the same row → fully coalesced. Both TileLang and Triton apply this arch-adaptive block shape. Result: Triton 2.5 ms → **0.29 ms**, TileLang **0.28 ms** on gfx1151.

### 04 — Backward Op
Implements both the forward and backward pass for a broadcast Mul+ReLU kernel (subgradient gate for ReLU). Demonstrates the full workflow for writing custom gradient kernels in TileLang and Triton, applicable wherever fine-grained gradient control is needed.

**iGPU optimisations** (gfx1151):
- **Triton**: switch from `BN=64, BM=64` to `BN=1, BM=2048` (same row-parallel principle as 03). Forward: 7.7 ms → **0.61 ms**; Backward: 11 ms → **0.91 ms**.
- **TileLang**: `BN=1, BM=2048`, no shared-B (B fits in L2 after first access). Forward: **0.59 ms**; Backward: **0.88 ms**.
- **PyTorch ref_bwd**: replaced `dC * B * (A*B>0).to(fp16)` (5 kernel launches on iGPU) with `dC.mul(B).mul_(A.mul(B).gt(0))` (3 launches via in-place ops) → 3.2 ms → **2.6 ms**.

### 05 — Reduce Sum
GPU implementation of row-wise reduction: `B[i] = sum(A[i,:], dim=1)`. Focuses on cross-thread data dependencies and the distinction between `T.Serial` (dependent iterations) and `T.Parallel` (independent iterations).

### 06 — Softmax
Numerically stable Softmax in two variants:
- **3-pass**: find row max → compute exp(x−max) → normalize. Beats Triton by ~24% on gfx1100 (config: `threads=256, BM=256`).
- **Online Softmax**: single-pass fused scan that maintains a running max and rescaled sum simultaneously. Best config on gfx1100/gfx1201: `threads=512, BM=4096` (8 floats per thread per iteration). On gfx1151: `threads=256, BM=256` (T.exp2 vectorization limit for RDNA3.5 iGPU). Motivates FlashAttention's IO-aware design.

### 07 — Scalar Flash Attention
A simplified FlashAttention implementation using element-wise `Q*K` (scalar product) in place of the full `QK^T` matrix multiply, while preserving the complete Online Softmax + tiled structure. Avoids materializing the N×N attention matrix, reducing memory complexity from O(N²) to O(N).

TileLang uses hardware-adaptive algorithm selection:
- **gfx1100 / gfx1201** (fast discrete memory): **3-pass** — write intermediate exp(QK−max) to DRAM, read back in Pass 3. DRAM bandwidth is high enough that the extra pass costs less than the register pressure of 2-pass.
- **gfx1151** (iGPU, unified memory, slow DRAM): **2-pass online** — Pass 1 warms L2 cache with Q,K; Pass 2 recomputes QK from L2 (≈10× faster than DRAM), eliminating the intermediate DRAM write+read. Result: TileLang **+30% vs PyTorch, +20% vs Triton** on gfx1151.


### 08 — Matrix Computation (GEMM)
Full progression from naive to hardware-accelerated GEMM:
- **Naive**: `alloc_fragment` (registers) + `T.Serial` K-loop
- **WMMA** (ROCm/RDNA3): uses `WMMAIntrinEmitter` to emit `v_wmma_f32_16x16x16_f16` instructions (warp-size=32). B must be pre-transposed to `(N×K)` layout. Best config: `brw=bcw=2, wrt=wct=64, chunk=64` → ~85 TFLOPS on RX 7900 XTX vs rocBLAS ~90 TFLOPS and Triton ~68 TFLOPS.

### 09 — Convolution (1D)
Single-channel and multi-channel 1D convolution with same padding. Key techniques: branch-free boundary handling (`T.if_then_else` instead of conditional branches, avoiding warp divergence) and 3D block grid design (batch × length × channel).

### 10 — Dequantized MatMul (W4A16)
INT4 weight-only quantization matrix multiply for LLM inference. Each `uint8` byte stores two INT4 values; the kernel unpacks nibbles on-the-fly inside the K-loop (low 4 bits for even columns, high 4 bits for odd columns), avoiding explicit FP16 expansion and saving ~4× weight memory and bandwidth.

## Benchmark Results (RX 7900 XTX / gfx1100)

All kernels were benchmarked on an **AMD Radeon RX 7900 XTX** (RDNA3, gfx1100, 96 CU, 24 GB GDDR6).

**Software**: TileLang 0.1.10+rocm.gitf791a9ca · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|--------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=1024` | 0.0070ms | 0.0071ms | **0.0067ms** | **+4%** | **+6%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0210ms | 0.0172ms | **0.0149ms** | **+41%** | **+15%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0244ms | **0.0120ms** | **0.0120ms** | **+103%** | ≈ |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1238ms | 0.0751ms | **0.0555ms** | **+123%** | **+35%** |
| 04 | Backward (fwd) | `BN=BM=128` | 0.3277ms | 0.3060ms | **0.2124ms** | **+54%** | **+44%** |
| 04 | Backward (bwd) | `BN=BM=128` | 0.6813ms | 0.6242ms | **0.4722ms** | **+44%** | **+32%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=128` | 0.3010ms | **0.2962ms** | 0.3031ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=4096, TH=512` | 0.7194ms | 0.8570ms | **0.6481ms** | **+11%** | **+32%** |
| 07 | Scalar Flash Attn | `BB=1, TH=64, BS=1024` | 0.0832ms | 0.0462ms | **0.0461ms** | **+80%** | ≈ |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | **1.5149ms**<br>**90.7 TFLOPS** | 1.9587ms<br>70.2 TFLOPS | 1.6028ms<br>85.7 TFLOPS | −6% | **+22%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0235ms | 0.0093ms | **0.0057ms** | **+312%** | **+63%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0243ms | 0.0095ms | **0.0044ms** | **+452%** | **+116%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 1.7982ms<br>76.4 TFLOPS | 2.7973ms<br>49.1 TFLOPS | **1.7937ms**<br>**76.6 TFLOPS** | ≈ | **+56%** |



## Benchmark Results (Radeon 8060S / gfx1151)

All kernels were benchmarked on an **AMD Radeon 8060S** (RDNA3.5 iGPU, gfx1151, 40 CU, 128 GB unified memory).

**Software**: TileLang 0.1.10+rocm.gitf791a9ca · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config (gfx1151) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=8192` | 0.0062ms | 0.0084ms | **0.0039ms** | **+59%** | **+115%** |
| 02 | Vector Add | `BLOCK_N=2048` | 0.0388ms | 0.0376ms | **0.0374ms** | ≈ | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=2048` | 0.0363ms ★ | 0.0373ms | 0.0373ms | ≈ | ≈ |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.3035ms | 0.2899ms | **0.2837ms** | **+7%** | **+2%** |
| 04 | Backward (fwd) | `BN=1, BM=2048` (1-row) | 1.1772ms | 0.6096ms | **0.5931ms** | **+98%** | **+3%** |
| 04 | Backward (bwd) | `BN=1, BM=2048` (1-row) | 2.5792ms | 0.9070ms | **0.8843ms** | **+192%** | **+3%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | **1.1165ms** | 1.1154ms | **1.1146ms** | ≈ | ≈ |
| 06 | Softmax (online) | `BM=1024, TH=256` | **2.5475ms** | 4.2689ms | **2.5600ms** | ≈ | **+67%** |
| 07 | Scalar Flash Attn | `BB=1, BS=256` (2-pass) | 0.5759ms | 0.5302ms | **0.4112ms** | **+40%** | **+29%** |
| 08 | GEMM (WMMA) | `wrt=wct=32, panel=8` | **4.0077ms**<br>**34.3 TFLOPS** | 9.3389ms<br>14.7 TFLOPS | 6.0123ms<br>22.9 TFLOPS | −50% | **+55%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0352ms ★ | 0.0107ms | **0.0039ms** | **+803%** | **+174%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0182ms | 0.0108ms | **0.0040ms** | **+355%** | **+170%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 6.4200ms<br>21.4 TFLOPS | 29.7754ms<br>4.6 TFLOPS | **3.9159ms**<br>**35.1 TFLOPS** | **+64%** | **+660%** |

★ PyTorch Mul+ReLU on gfx1151 uses `torch.compile` to fuse mul+relu into a single Inductor kernel.
★ PyTorch single-ch conv uses `unfold+matmul` instead of MIOpen conv1d (lower launch overhead for small N).

**gfx1151 (RDNA3.5 iGPU) characteristics:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp_size=32 as gfx1100/gfx1201 — `WMMAIntrinEmitter` works unchanged
- Unified memory (~0.21 TB/s effective): CPU and GPU share one memory bus, so cache-miss patterns are much more costly than on discrete GPUs
- **Key iGPU optimisation — BN=1 row-parallel**: `T.Parallel(BN, BM)` / Triton 2D tiles with large `BN` cause stride-M (8 KB) writes → cache miss. Fix: `BN=1` (one row per block/program), all threads write the same row's consecutive columns → coalesced. Applied to **03_outer** and **04_backward** for both TileLang and Triton
- `T.reduce_sum`: no BM limit after warpSize fix. Config `BN=1, BM=1024, TH=256` saturates iGPU bandwidth at ~0.24 TB/s, matching PyTorch/Triton
- **06_softmax**: BM=1024 now works after HIPMath vector-dtype fix (PR #2313); matches PyTorch (≈0%). Old BM=256 gave −10%
- **08_GEMM**: rocBLAS outperforms TileLang on gfx1151 (iGPU CU count limit); TileLang still beats Triton by +55%
- **10_dequant_mm**: TileLang **+64%** vs PyTorch (35.1 vs 21.4 TFLOPS). Fuses W4→FP16 unpack with GEMM in shared memory
- TileLang excels on compute-bound kernels: **04_bwd** (+192%), **07_attn** (+40%), **09_conv** (+803%), **10_dequant** (+64%)

## Benchmark Results (R9700 / gfx1201)

All kernels were benchmarked on an **AMD Radeon AI PRO R9700** (RDNA4, gfx1201, 64 CU, 32 GB, ~0.58 TB/s peak BW).

**Software**: TileLang 0.1.10+rocm.gitf791a9ca · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

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
| 01 | Copy (multi-block) | `BLOCK_N=2048` | 0.0038ms | 0.0085ms | **0.0039ms** | ≈ | **+118%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0215ms | 0.0167ms | **0.0167ms** | **+29%** | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0300ms | **0.0166ms** | **0.0165ms** | **+82%** | ≈ |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1537ms | 0.0887ms | **0.0688ms** | **+123%** | **+29%** |
| 04 | Backward (fwd) | `BN=BM=128, shared B` | 0.5144ms | 0.6272ms | **0.3773ms** | **+36%** | **+66%** |
| 04 | Backward (bwd) | `BN=BM=128, shared B` | 1.1017ms | 0.9050ms | **0.5757ms** | **+91%** | **+57%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=64` | 0.4925ms | **0.4951ms** | 0.5171ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=4096, TH=512` | 1.0685ms | 1.6677ms | **1.0597ms** | **+1%** | **+57%** |
| 07 | Scalar Flash Attn | `BB=1, TH=128, BS=1024` | 0.1903ms | 0.0798ms | **0.0790ms** | **+141%** | **+1%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | **1.1568ms**<br>**118.8 TFLOPS** | 1.3994ms<br>98.2 TFLOPS | 1.1598ms<br>118.5 TFLOPS | ≈ | **+21%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0136ms | 0.0105ms | **0.0041ms** | **+232%** | **+156%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0220ms | 0.0139ms | **0.0044ms** | **+400%** | **+216%** |
| 10 | Dequant MM (W4A16) | `BM=256, BN=128, BK=32` | 1.7055ms<br>80.6 TFLOPS | 2.4946ms<br>55.1 TFLOPS | **1.4763ms**<br>**93.1 TFLOPS** | **+16%** | **+69%** |

**gfx1201 vs gfx1100 key differences:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp size = 32 — `WMMAIntrinEmitter` works unchanged
- `01_copy`: optimal `BLOCK_N=2048` (vs 1024) — 64 CU fills better with larger per-block tiles
- `05_reduce`: `BN=2, BM=128, TH=64` — warpSize fix enables BM>64; 64-thread blocks maximise bandwidth on 64 CU
- `10_dequant`: `BM=256, BN=128, BK=32, TH=256` — larger block + 256 threads → 93.1 TFLOPS, +16% vs PyTorch
- `08_GEMM`: TileLang 118.5 TFLOPS matches rocBLAS 118.8 TFLOPS (RDNA4 WMMA fully enabled by PR #2313)

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

