# TileLang Notebooks

A structured series of Tilelang GPU kernel programming tutorials that compare **TileLang**, **Triton**, and **PyTorch** side by side, progressing from fundamental memory operations to advanced LLM inference kernels.

Each notebook covers: problem background, PyTorch reference implementation, Triton implementation, TileLang implementation, correctness validation, and a printed performance table (latency, bandwidth/TFLOPS, and speedup vs PyTorch and Triton).


> **Optimization Guide**: See [RADEON_OPTIMIZATION_GUIDE.md](./RADEON_OPTIMIZATION_GUIDE.md) for a comprehensive summary of all Radeon GPU optimization lessons learned from this project.
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

Remaining gap vs Triton on gfx1100/gfx1201: `T.reduce_max/sum` requires `BS ≤ 128` on RDNA hardware, while Triton uses `BLOCK_S=1024` (8× larger tile) — a current TileLang framework limitation.

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

**Software**: TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). Bandwidth/throughput in TB/s or TFLOPS (higher is better).

| # | Kernel | Config | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|--------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=1024` | 0.0070ms | 0.0071ms | **0.0054ms** | **+30%** | **+31%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0219ms | 0.0163ms | **0.0131ms** | **+67%** | **+24%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0231ms | **0.0121ms** | **0.0119ms** | **+94%** | ≈ |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1241ms | 0.0749ms | **0.0556ms** | **+123%** | **+35%** |
| 04 | Backward fwd | `BN=BM=128, shared B` | 0.3385ms | 0.3556ms | **0.2388ms** | **+42%** | **+49%** |
| 04 | Backward bwd | `BN=BM=128, shared B` | 0.6957ms | 0.6660ms | **0.4793ms** | **+45%** | **+39%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=128` | 0.3010ms | **0.2959ms** | 0.2980ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=4096, TH=512` | 0.7199ms | 0.8587ms | **0.6483ms** | **+11%** | **+32%** |
| 07 | Scalar Flash Attn | `BB=1, TH=64, BS=1024` | 0.0802ms | 0.0463ms | **0.0461ms** | **+74%** | ≈ |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | **1.5145ms** | 1.9526ms | 1.5994ms | −5% | **+22%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0227ms | 0.0091ms | **0.0054ms** | **+320%** | **+69%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0228ms | 0.0093ms | **0.0044ms** | **+418%** | **+111%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 1.7983ms | 2.8004ms | **1.7978ms** | ≈ | **+56%** |


† PR #2210 applied: warpSize fix in reduce.h, gfx12 WMMA support in rdna.py, k_pack indexing fix in wmma_macro_generator.py.
  07_attn BB=1 design: each block handles 1 row; T.Parallel load (no BM≤threads constraint); BS=1024 matches Triton tile size; TH=64 (2 wavefronts).

**gfx1100-specific constraints** that informed the configs above:
- `T.copy` requires `BM ≤ threads` to generate a vectorised loop
- `T.reduce_sum/max` requires `BLOCK_M ≤ 128` for correct warp reduction on RDNA3; gfx1201/gfx1151 have stricter limits (see their sections)
- WMMA uses `v_wmma_f32_16x16x16_f16` (RDNA3), **not** MFMA — use `WMMAIntrinEmitter`, not `MatrixCoreIntrinEmitter`
- warp size = 32 (RDNA3), not 64 (CDNA)

## Benchmark Results (Radeon 8060S / gfx1151)

All kernels were benchmarked on an **AMD Radeon 8060S** (RDNA3.5 iGPU, gfx1151, 40 CU, 128 GB unified memory).

**Software**: TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). Effective memory bandwidth in TB/s. TFLOPS for compute kernels.  
> iGPU unified memory delivers lower peak BW than discrete GPUs (~0.21 TB/s effective vs ~0.58 TB/s on R9700).
> `‡` marks where PyTorch baseline is not a fair GPU comparison (serial Python loop).

| # | Kernel | Config (gfx1151) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=8192` | 0.0075ms | 0.0084ms | **0.0053ms** | **+42%** | **+58%** |
| 02 | Vector Add | `BLOCK_N=2048` | 0.0391ms | 0.0375ms | **0.0376ms** | **+4%** | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=2048` | 0.0361ms ★ | 0.0373ms | 0.0377ms | ≈ | ≈ |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.3099ms | 0.2916ms | **0.2822ms** | **+10%** | **+3%** |
| 04 | Backward fwd | `BN=1, BM=2048` (1-row) | 1.1763ms | **0.6110ms** | **0.5938ms** | **+98%** | ≈ |
| 04 | Backward bwd | `BN=1, BM=2048` (1-row) | 2.5885ms | **0.9044ms** | **0.8792ms** | **+194%** | ≈ |
| 05 | Reduce Sum | `BN=1, BM=128, TH=256` (T.Parallel) | **1.1148ms** | **1.1188ms** | 2.0959ms | −47% | −47% |
| 06 | Softmax (online) | `BM=256, TH=256` | **2.5430ms** | 4.4869ms | 2.8319ms | −10% | **+58%** |
| 07 | Scalar Flash Attn | `BB=1, BS=256` (2-pass) | 0.5821ms | 0.5076ms | **0.4233ms** | **+38%** | **+20%** |
| 08 | GEMM (WMMA) | `wrt=wct=32, panel=8` | **4.0485ms** | 9.7002ms | 5.8037ms | −30% | **+67%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0349ms ★ | 0.0106ms | **0.0038ms** | **+818%** | **+179%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0182ms | 0.0107ms | **0.0039ms** | **+367%** | **+174%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 6.4980ms | 19.2121ms | **3.9170ms** | **+66%** | **+390%** |

★★ PyTorch Mul+ReLU on gfx1151 uses `torch.compile` to fuse mul+relu into a single Inductor kernel (vs 2 separate kernels in eager mode → 2 DRAM round-trips → 0.142 ms). Compile call in `code-pytorch`: `torch.compile(lambda a,b: torch.relu(a*b))`.
★ PyTorch single-ch on gfx1151 uses `unfold+matmul` instead of MIOpen conv1d (which has high launch overhead for N=64, L=1024) — 34% faster than MIOpen.

**gfx1151 (RDNA3.5 iGPU) characteristics:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp_size=32 as gfx1100/gfx1201 — `WMMAIntrinEmitter` works unchanged
- Unified memory (~0.21 TB/s effective): CPU and GPU share one memory bus, so cache-miss patterns are much more costly than on discrete GPUs
- **Key iGPU optimisation — BN=1 row-parallel**: `T.Parallel(BN, BM)` / Triton 2D tiles with large `BN` cause stride-M (8 KB) writes → cache miss. Fix: `BN=1` (one row per block/program), all threads write the same row’s consecutive columns → coalesced. Applied to **03_outer** and **04_backward** for both TileLang and Triton
- `T.reduce_sum` constraint: **T.copy → BM ≤ 64**; **T.Parallel → BM ≤ 128**. Using T.Parallel for 05_reduce enables BM=128 (→ 2.10 ms; still 2× slower than PyTorch/Triton due to iGPU bandwidth; T.Parallel load approach is correct, but unified memory bottleneck limits throughput)
- TileLang excels on compute-bound and coalescing-friendly kernels: **03_outer** (+7% vs PyTorch, +2% vs Triton), **04_fwd** (+98%, +2% vs Triton), **04_bwd** (+193%, +3% vs Triton), **07_attn** (+30% vs PyTorch), **09_conv** (+1337% vs PyTorch), **10_dequant_mm** (+68% vs PyTorch)

## Benchmark Results (R9700 / gfx1201)

All kernels were benchmarked on an **AMD Radeon AI PRO R9700** (RDNA4, gfx1201, 64 CU, 32 GB, ~0.58 TB/s peak BW).

**Software**: TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

The notebooks auto-detect the GPU architecture at runtime and select the appropriate config:
```python
arch = torch.cuda.get_device_properties(0).gcnArchName  # "gfx1100" or "gfx1201"
if arch.startswith("gfx1201"):
    ...  # R9700 config
else:
    ...  # RX 7900 XTX config (original)
```

> Latency in milliseconds (lower is better). BW = effective memory bandwidth (TB/s). TFLOPS for compute-bound kernels.
> `†` marks configs that differ from the gfx1100 optimal.

| # | Kernel | Config (gfx1201) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=2048` | 0.0089ms | 0.0088ms | **0.0053ms** | **+68%** | **+66%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0574ms | 0.0416ms | **0.0405ms** | **+42%** | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0790ms | **0.0401ms** | 0.0443ms | **+78%** | −10% |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1912ms | 0.1099ms | **0.0692ms** | **+176%** | **+59%** |
| 04 | Backward fwd | `BN=BM=128, shared B` | 0.5179ms | 0.6294ms | **0.3927ms** | **+32%** | **+60%** |
| 04 | Backward bwd | `BN=BM=128, shared B` | 1.1961ms | 0.9901ms | **0.6845ms** | **+75%** | **+45%** |
| 05 | Reduce Sum | `BN=1, BM=64, TH=128` † | 0.5266ms | **0.4981ms** | 0.5234ms | ≈ | −5% |
| 06 | Softmax (online) | `BM=4096, TH=512` | 1.1674ms | 1.7510ms | **1.2033ms** | ≈ | **+46%** |
| 07 | Scalar Flash Attn | `BB=1, TH=128, BS=1024` | 0.2051ms | 0.1321ms | **0.0930ms** | **+121%** | **+42%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | 1.4262ms | 1.6032ms | **1.4174ms** | ≈ | **+13%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64, TH=128` | 0.0226ms | 0.0114ms | **0.0047ms** | **+381%** | **+143%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0683ms | 0.0117ms | **0.0072ms** | **+849%** | **+62%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 2.1519ms | 2.5908ms | **2.0895ms** | ≈ | **+24%** |

† Config differs from gfx1100 optimal.  

**gfx1201 vs gfx1100 key differences:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp size = 32 — `WMMAIntrinEmitter` works unchanged
- `01_copy`: optimal `BLOCK_N=2048` (vs 1024) — 64 CU fills better with larger per-block tiles
- `07_attn`: TH=128 (4 wavefronts/block) with BS=1024 achieves 0.081ms vs Triton 0.128ms (+58%). PR #2210 WMMA fix enabled gfx1201 kernel path
- `05_reduce`: `BN=1, BM=64, TH=128` — gfx1201 requires `BM ≤ 64` for `T.reduce_sum` correctness; gfx1100 tolerates `BM=128`
- `T.reduce_max/sum` on any `(BB, BS)` fragment: `BS ≤ 128` on all three RDNA archs — limits tile size for 07_attn (Triton uses `BLOCK_S=1024` without this constraint)
- Higher peak memory bandwidth (~0.58 TB/s vs ~0.96 TB/s) benefits all bandwidth-bound kernels

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

