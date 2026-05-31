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

**Software**: TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|--------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=1024` | 0.0063ms | 0.0071ms | **0.0053ms** | **+20%** | **+34%** |
| 02 | Vector Add | `BLOCK_N=1024` | 0.0199ms | 0.0163ms | **0.0158ms** | **+26%** | **+3%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=1024` | 0.0231ms | **0.0121ms** | **0.0119ms** | **+94%** | ≈ |
| 03 | Outer Vector Add | `BN=256, BM=64` | 0.1235ms | 0.0749ms | **0.0731ms** | **+69%** | **+2%** |
| 04 | Backward | `BN=BM=128` | 0.4499ms | 0.3556ms | **0.2890ms** | **+56%** | **+23%** |
| 05 | Reduce Sum | `BN=2, BM=128, TH=128` | 0.3011ms | **0.2959ms** | 0.3039ms | ≈ | ≈ |
| 06 | Softmax (online) | `BM=4096, TH=512` | 0.7194ms | 0.8587ms | **0.6477ms** | **+11%** | **+32%** |
| 07 | Scalar Flash Attn | `BB=1, TH=64, BS=1024` | 2.1001ms | 0.0463ms | **0.1155ms** | **+18.2×** | − |
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | 1.5590ms<br>87.9 TFLOPS | 1.9526ms<br>70.4 TFLOPS | **1.5634ms**<br>**87.9 TFLOPS** | ≈ | **+25%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0305ms | 0.0091ms | **0.0061ms** | **+400%** | **+49%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0397ms | 0.0093ms | **0.0073ms** | **+444%** | **+27%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 0.2852ms | 2.8004ms<br>49.1 TFLOPS | **0.1291ms** | **+121%** | **+2068%** |


† TileLang rdna4_ci branch (commit 4ec25e0b): PR #2210 (warpSize fix in reduce.h, gfx12 WMMA support in rdna.py, k_pack indexing fix in wmma_macro_generator.py) + HIPMath vector-dtype fallback fix for tirx.exp2 in intrin_rule_hip.cc.
  07_attn: PyTorch uses experimental scaled_dot_product_attention which falls back to a slow reference path on AMD; TileLang's scalar attention is 18× faster.
  10_dequant: PyTorch reference uses float matmul after nibble unpack (0.29ms); TileLang fuses nibble unpack + GEMM in shared memory (0.13ms, 2.2×).


## Benchmark Results (Radeon 8060S / gfx1151)

All kernels were benchmarked on an **AMD Radeon 8060S** (RDNA3.5 iGPU, gfx1151, 40 CU, 128 GB unified memory).

**Software**: TileLang 0.1.10+rocm · PyTorch 2.12.0+rocm7.2 · ROCm 7.2 · Triton 3.7.0

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).

| # | Kernel | Config (gfx1151) | PyTorch | Triton | TileLang | vs PyTorch | vs Triton |
|---|--------|-----------------|:-------:|:------:|:--------:|:----------:|:---------:|
| 01 | Copy (multi-block) | `BLOCK_N=8192` | 0.0068ms | 0.0084ms | **0.0068ms** | ≈ | **+24%** |
| 02 | Vector Add | `BLOCK_N=2048` | 0.0336ms | 0.0375ms | **0.0337ms** | ≈ | **+11%** |
| 02 | Mul + ReLU (fused) | `BLOCK_N=2048` | 0.0361ms ★ | 0.0373ms | 0.0377ms | ≈ | ≈ |
| 03 | Outer Vector Add | `BN=1, BM=4096` (1-row) | 0.3000ms | 0.2916ms | **0.2802ms** | **+7%** | **+4%** |
| 04 | Backward | `BN=1, BM=2048` (1-row) | 1.7408ms | **0.6110ms** | **0.9579ms** | **+82%** | − |
| 05 | Reduce Sum | `BN=1, BM=1024, TH=256` | **1.1170ms** | 1.1177ms | **1.1139ms** | ≈ | ≈ |
| 06 | Softmax (online) | `BM=1024, TH=256` | **2.5439ms** | 4.4869ms | **2.5505ms** | ≈ | **+76%** |
| 07 | Scalar Flash Attn | `BB=1, BS=256` (2-pass) | 13.6383ms | 0.5076ms | **0.2391ms** | **+57×** | **+112%** |
| 08 | GEMM (WMMA) | `wrt=wct=32, panel=8` | **3.9849ms**<br>**34.2 TFLOPS** | 9.7002ms<br>14.2 TFLOPS | 4.8509ms<br>28.3 TFLOPS | −22% | **+100%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64` | 0.0409ms ★ | 0.0106ms | **0.0046ms** | **+789%** | **+130%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0222ms | 0.0107ms | **0.0058ms** | **+283%** | **+84%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 1.5549ms | 19.2121ms<br>7.2 TFLOPS | **0.1311ms** | **+1086%** | **+14556%** |

★★ PyTorch Mul+ReLU on gfx1151 uses `torch.compile` to fuse mul+relu into a single Inductor kernel (vs 2 separate kernels in eager mode → 2 DRAM round-trips → 0.142 ms). Compile call in `code-pytorch`: `torch.compile(lambda a,b: torch.relu(a*b))`.
★ PyTorch single-ch on gfx1151 uses `unfold+matmul` instead of MIOpen conv1d (which has high launch overhead for N=64, L=1024) — 34% faster than MIOpen.

**gfx1151 (RDNA3.5 iGPU) characteristics:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp_size=32 as gfx1100/gfx1201 — `WMMAIntrinEmitter` works unchanged
- Unified memory (~0.21 TB/s effective): CPU and GPU share one memory bus, so cache-miss patterns are much more costly than on discrete GPUs
- **Key iGPU optimisation — BN=1 row-parallel**: `T.Parallel(BN, BM)` / Triton 2D tiles with large `BN` cause stride-M (8 KB) writes → cache miss. Fix: `BN=1` (one row per block/program), all threads write the same row’s consecutive columns → coalesced. Applied to **03_outer** and **04_backward** for both TileLang and Triton
- `T.reduce_sum`: no BM limit after PR #2210 warpSize fix. Config `BN=1, BM=1024, TH=256` saturates iGPU bandwidth at ~0.24 TB/s, matching PyTorch/Triton
- **06_softmax**: BM=1024 (16 tiles) now works after HIPMath vector-dtype fix; matches PyTorch (≈0%). Old BM=256 (64 tiles) gave −10%.
- **07_flash_attn**: TileLang **57× vs PyTorch** — PyTorch's `scaled_dot_product_attention` on gfx1151 falls back to a slow reference path (13.6ms); TileLang custom scalar attention runs in 0.24ms
- **10_dequant_mm**: TileLang **11.9×** vs PyTorch's pure-float reference. Fuses W4→FP16 unpack with GEMM in shared memory
- TileLang excels on compute-bound kernels: **04_bwd** (+82%), **07_attn** (+57×), **09_conv** (+789%), **10_dequant** (+1086%)

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

> Latency in milliseconds (lower is better). TFLOPS shown for compute-bound kernels (GEMM, Dequant MM).
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
| 08 | GEMM (WMMA) | `wrt=wct=64, panel=10` | 1.4262ms<br>96.4 TFLOPS | 1.6032ms<br>85.7 TFLOPS | **1.4174ms**<br>**97.0 TFLOPS** | ≈ | **+13%** |
| 09 | Conv 1D (single-ch) | `BN=4, BL=64, TH=128` | 0.0226ms | 0.0114ms | **0.0047ms** | **+381%** | **+143%** |
| 09 | Conv 1D (multi-ch) | `BN=4, BL=32, BF=32` | 0.0683ms | 0.0117ms | **0.0072ms** | **+849%** | **+62%** |
| 10 | Dequant MM (W4A16) | `BM=BN=128, BK=32` | 2.1519ms<br>63.9 TFLOPS | 2.5908ms<br>53.0 TFLOPS | **2.0895ms**<br>**65.8 TFLOPS** | ≈ | **+24%** |

† Config differs from gfx1100 optimal.  

**gfx1201 vs gfx1100 key differences:**
- Same WMMA ISA (`v_wmma_f32_16x16x16_f16`) and warp size = 32 — `WMMAIntrinEmitter` works unchanged
- `01_copy`: optimal `BLOCK_N=2048` (vs 1024) — 64 CU fills better with larger per-block tiles
- `07_attn`: TH=128 (4 wavefronts/block) with BS=1024 achieves 0.081ms vs Triton 0.128ms (+58%). PR #2210 WMMA fix enabled gfx1201 kernel path
- `05_reduce`: `BN=1, BM=64, TH=128` — this was chosen before PR #2210; larger BM also works correctly now
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

