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
- **gfx1151** (iGPU, unified memory, slow DRAM): **2-pass online** — Pass 1 warms L2 cache with Q,K; Pass 2 re-reads Q,K which often remain in L2 cache, reducing external memory traffic compared to the extra DRAM round-trip. Best config: `BB=1, BS=2048, TH=256`. Result: TileLang **+92% vs PyTorch, +61% vs Triton** on gfx1151.


### 08 — Matrix Computation (GEMM)
Full progression from naive to hardware-accelerated GEMM:
- **Naive**: `alloc_fragment` (registers) + `T.Serial` K-loop
- **WMMA** (ROCm/RDNA3+): uses `WMMAIntrinEmitter` to emit `v_wmma_f32_16x16x16_f16` instructions (warp-size=32). The WMMA emitter uses `b_transposed=True` to interpret B in transposed layout; this can be achieved either by physical transposition (`B.T.contiguous()`) or by staging B in transposed form during shared-memory loading. Best config on gfx1100: `wrt=wct=64, panel=3, num_stages=5` → **87.6 TFLOPS** (≈ rocBLAS 93.0 TFLOPS, Triton 66.6 TFLOPS). On gfx1201: **122.6 TFLOPS**, surpassing rocBLAS (121.8 TFLOPS) and Triton (52.7 TFLOPS).

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
| 01 | Copy (multi-block★) | `BLOCK_N=2048, TH=256` | 0.0061ms | 0.0113ms | **0.0057ms** | **+7%** | **+97%** |
| 02 | Vector Add | `BN=1024, TH=256` | 0.0207ms | 0.0175ms | **0.0140ms** | **+48%** | **+25%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0318ms | 0.0166ms | **0.0160ms** | **+98%** | **+4%** |
| 03 | Outer Vector Add | `BN=512, BM=64, TH=128` | 0.1155ms | 0.0824ms | **0.0502ms** | **+130%** | **+64%** |
| 04 | Backward (fwd) | `BN=512, BM=128, TH=256` | 0.3341ms | 0.3171ms | **0.1792ms** | **+86%** | **+77%** |
| 04 | Backward (bwd) | `BN=BM=128` | 0.6854ms | 0.5857ms | **0.5004ms** | **+37%** | **+17%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | 0.2998ms | **0.2961ms** | **0.2952ms** | **+2%** | ≈ |
| 06 | Softmax (online) | `BM=8192, TH=512` | 0.7984ms | 0.8695ms | **0.6487ms** | **+23%** | **+34%** |
| 07 | Scalar Flash Attn | `TH=64, BS=1024` | 0.0798ms | **0.0596ms** | 0.0595ms | **+34%** | ≈ |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=3, ns=5` | **1.4782ms**<br>**93.0 TFLOPS** | 2.0647ms<br>66.6 TFLOPS | **1.5687ms**<br>**87.6 TFLOPS** | **≈** | **+31%** |
| 09 | Conv 1D (single-ch) | `BN=1, BL=256, TH=256` | 0.0145ms | 0.0089ms | **0.0062ms** | **+133%** | **+42%** |
| 09 | Conv 1D (multi-ch) | `BN=1, BL=32, BF=32` | 0.0369ms | 0.0118ms | **0.0064ms** | **+476%** | **+84%** |
| 10 | Dequant MM (W4A16) | `BM=256, BN=128, BK=32, TH=256` | 1.7735ms<br>77.5 TFLOPS | 2.7837ms<br>49.4 TFLOPS | **1.7173ms**<br>**80.0 TFLOPS** | **+3%** | **+62%** |



## Benchmark Results (Radeon 8060S / gfx1151)

All kernels were benchmarked on an **AMD Radeon 8060S** (RDNA3.5 iGPU, gfx1151, 40 CU, 128 GB unified memory).

**Software**: TileLang 0.1.10+rocm.gitfe7cda86 · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config (gfx1151) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block★) | `BLOCK_N=256, TH=128` | **0.0037ms** | 0.0160ms | 0.0039ms | −5% | **+310%** |
| 02 | Vector Add | `BN=2048, TH=256` | 0.0382ms | 0.0377ms | 0.0379ms | ≈ | ≈ |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0420ms ★ | 0.0351ms | **0.0275ms** | **+53%** | **+28%** |
| 03 | Outer Vector Add | `BN=1, BM=4096, TH=256` (1-row) | 0.3017ms | 0.2966ms | **0.2789ms** | **+8%** | **+6%** |
| 04 | Backward (fwd) | `BN=1, BM=4096, TH=256` (1-row) | 1.1796ms | 0.6136ms | **0.5905ms** | **+100%** | **+4%** |
| 04 | Backward (bwd) | `BN=1, BM=4096` (1-row) | 2.5977ms | 0.9063ms | **0.7179ms** | **+262%** | **+26%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | 1.1262ms | 1.1351ms | **1.1156ms** | **+1%** | **+2%** |
| 06 | Softmax (online) | `BM=1024, TH=256` | 3.0127ms | 4.7458ms | **2.5555ms** | **+18%** | **+86%** |
| 07 | Scalar Flash Attn | `BB=1, BS=2048, TH=256` (2-pass) | 0.6159ms | 0.5170ms | **0.3207ms** | **+92%** | **+61%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=4` | **4.1245ms**<br>**33.3 TFLOPS** | 19.6792ms<br>7.0 TFLOPS | 4.8667ms<br>28.2 TFLOPS | −15% | **+304%** |
| 09 | Conv 1D (single-ch) | `BN=8, BL=64, TH=256` | 0.0396ms ★ | 0.0105ms | **0.0037ms** | **+965%** | **+186%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=16, BF=16` | 0.0192ms | 0.0105ms | **0.0089ms** | **+117%** | **+18%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32, TH=128` | 6.4409ms<br>21.3 TFLOPS | 7.2821ms<br>18.9 TFLOPS | **3.9470ms**<br>**34.8 TFLOPS** | **+63%** | **+84%** |

★ PyTorch Mul+ReLU on gfx1151 uses `torch.compile` to fuse mul+relu into a single Inductor kernel.
★ PyTorch single-ch conv uses `unfold+matmul` instead of MIOpen conv1d (lower launch overhead for small N on iGPU).

**gfx1151 (RDNA3.5 iGPU) characteristics:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp_size=32 as gfx1100/gfx1201 — `WMMAIntrinEmitter` works unchanged
- Unified memory (spec peak ~0.26 TB/s): CPU and GPU share one memory bus; lower bandwidth amplifies the cost of inefficient access patterns compared to discrete GPUs
- **Key iGPU optimisation — BN=1 row-parallel**: `T.Parallel(BN, BM)` / Triton 2D tiles with large `BN` cause stride-M (8 KB) writes → cache miss. Fix: `BN=1` (one row per block/program), all threads write the same row's consecutive columns → coalesced. Applied to **03_outer** and **04_backward** for both TileLang and Triton
- `T.reduce_sum`: no BM limit after warpSize fix ([PR #2313](https://github.com/tile-ai/tilelang/pull/2313)). Config `BN=1, BM=1024, TH=256` saturates iGPU bandwidth at ~0.24 TB/s, matching PyTorch/Triton
- **06_softmax**: BM=1024 now works after HIPMath vector-dtype fix; TileLang **+18% vs PyTorch, +86% vs Triton**
- **08_GEMM**: rocBLAS outperforms TileLang on gfx1151 (fewer CUs limit WMMA occupancy); TileLang still beats Triton by **+304%**
- **10_dequant_mm**: TileLang **+63% vs PyTorch** (34.8 vs 21.3 TFLOPS). Fuses W4→FP16 unpack with GEMM in shared memory
- TileLang excels across kernels: **04_bwd** (+262%), **07_flash** (+92%), **09_conv single** (+965%), **09_conv multi** (+117%), **10_dequant** (+63%)
- **01_copy** at N=512K: `T.copy BN=256, TH=128` → PyTorch wins by 5% (iGPU BW bound); TileLang beats Triton by **+310%**
- **02_vector_add**: `BN=2048, TH=256` — ≈ PyTorch and Triton (~0.038ms). iGPU unified memory bandwidth is the shared ceiling; all three frameworks saturate it equally. BN=2048 (128-bit loads) is the best TileLang config; BN=1024 (64-bit) is measurably slower

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
| 01 | Copy (multi-block★) | `BLOCK_N=2048, TH=128` | **0.0057ms** | 0.0075ms | 0.0049ms | **+16%** | **+53%** |
| 02 | Vector Add | `BN=2048, TH=128` | 0.0210ms | 0.0197ms | **0.0166ms** | **+27%** | **+19%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0287ms | **0.0169ms** | 0.0191ms | **+51%** | ≈ |
| 03 | Outer Vector Add | `BN=512, BM=256, TH=128` | 0.1547ms | 0.0904ms | **0.0428ms** | **+262%** | **+112%** |
| 04 | Backward (fwd) | `BN=512, BM=256, TH=128` | 0.5130ms | 0.5366ms | **0.3073ms** | **+67%** | **+75%** |
| 04 | Backward (bwd) | `BN=BM=128` | 1.1053ms | 0.8229ms | **0.5942ms** | **+86%** | **+38%** |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=64` | 0.4929ms | 0.4951ms | **0.4898ms** | **+1%** | **+1%** |
| 06 | Softmax (online) | `BM=4096, TH=512` | 1.0681ms | 1.6793ms | **1.0596ms** | **+1%** | **+58%** |
| 07 | Scalar Flash Attn | `TH=128, BS=1024` | 0.2044ms | 0.1217ms | **0.0815ms** | **+151%** | **+49%** |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | 1.1284ms<br>**121.8 TFLOPS** | 2.6103ms<br>52.7 TFLOPS | **1.1209ms**<br>**122.6 TFLOPS** | **+1%** | **+133%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=32, TH=256` | 0.0111ms | 0.0092ms | **0.0047ms** | **+138%** | **+97%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=16` | 0.0319ms | 0.0141ms | **0.0048ms** | **+563%** | **+193%** |
| 10 | Dequant MM (W4A16) | `BM=256, BN=128, BK=32, TH=256` | 1.7045ms<br>80.6 TFLOPS | 2.4969ms<br>55.0 TFLOPS | **1.4543ms**<br>**94.5 TFLOPS** | **+17%** | **+72%** |

**gfx1201 vs gfx1100 key differences:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp size = 32 — `WMMAIntrinEmitter` works unchanged
- **08_GEMM**: TileLang WMMA **122.6 TFLOPS surpasses rocBLAS** (119.9 TFLOPS) on gfx1201 — RDNA4 WMMA fully enabled by PR #2313 (warpSize fix + gfx12 registration)
- `01_copy`: `BLOCK_N=2048, TH=128` — 0.0049ms, **+16% vs PyTorch, +53% vs Triton**
- `02_vector_add`: `tl_add_gfx1201, BN=2048, TH=128` — **0.0166ms, +27% vs PyTorch, +19% vs Triton** (1.01 TB/s); previously −3% vs PyTorch. Fix: `T.alloc_fragment` + `T.copy` replaces `T.Parallel` direct indexing → 128-bit vectorized loads (`global_load_dwordx4`)
- `03_outer`: `BN=512, BM=256, TH=128` — **+262% vs PyTorch, +112% vs Triton**
- `07_flash_attn`: `TH=128, BS=1024` — **+151% vs PyTorch, +49% vs Triton** (3-pass scalar attn)
- `09_conv`: single **+138%**, multi **+563%** vs PyTorch (unfold+matmul ref for multi-ch)
- `10_dequant`: **94.5 TFLOPS** (+17% vs PyTorch, +72% vs Triton) — gfx1201 highest compute throughput

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
