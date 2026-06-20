---
type: kernel-doc
title: "W4A16 Fused Dequant+GEMM — Design"
tags: [fused-iq, design, w4a16]
timestamp: 2026-06-17
---

# W4A16 Fused Dequant+GEMM — Design

## 1. Problem statement

LLM inference with quantized weights (W4A16: 4-bit weights, 16-bit activations) is memory-bandwidth bound.  The naive two-step approach allocates a full fp16 weight matrix `[N, K]` in HBM, writes it from a dequant pass, then reads it back for GEMM.  This wastes 2× the peak bandwidth.  The fused kernel unpacks and dequantizes int4 nibbles *in registers* during the GEMM tile computation, cutting HBM weight traffic from `N*K*2 bytes` (fp16) to `N*K/2 bytes` (int4) — a 4× reduction on weight reads.

---

## 2. Quantization math

### Symmetric per-group

For output row `n` and input column `k`, the dequantized weight is:

$$w_{\text{fp16}}[n, k] = w_{\text{int4}}[n, k] \times s[n, \lfloor k / g \rfloor]$$

where $g$ is `group_size` (default 128) and $s$ is an fp16 scale tensor of shape $[N, K/g]$.

**Symmetric** means zero-point = 0 after subtracting 8 from the stored unsigned nibble.  The int4 range is $[-8, 7]$.  There is no explicit zero-point tensor.

**Per-group** means scales are shared within each group of $g$ consecutive input columns.  A smaller $g$ recovers more accuracy at the cost of more scale storage; $g = 128$ matches the default GPTQ/bitsandbytes convention.

---

## 3. Packing layout

Two int4 values are packed into one int8 byte:

```
byte = (hi_nibble << 4) | lo_nibble
      bits[7:4] = w[2i+1] + 8   (odd  column index, unsigned [0,15])
      bits[3:0] = w[2i]   + 8   (even column index, unsigned [0,15])
```

Offset of +8 converts signed int4 $[-8, 7]$ to unsigned nibble $[0, 15]$.  On dequant, subtract 8 to recover signed values.

Stored tensor: `W_packed` shape `[N, K//2]` dtype `int8`.

---

## 4. Fused GEMM tiling

The Triton kernel tiles the `[M, N]` output into `BLOCK_M × BLOCK_N` blocks.  Each program iterates over K in chunks of `BLOCK_K`.  For each K chunk:

1. Load `x` tile `[BLOCK_M, BLOCK_K]` in fp16.
2. Load `W_packed` tile `[BLOCK_N, BLOCK_K//2]` in int8 (half the K width because of packing).
3. Unpack int4 nibbles in registers → `[BLOCK_N, BLOCK_K]` int32.
4. Load `scales` tile `[BLOCK_N, BLOCK_K//group_size]` in fp16.
5. Broadcast scales to `[BLOCK_N, BLOCK_K]` and multiply → `W_fp16` in registers.
6. Accumulate `x_tile @ W_fp16.T` into `[BLOCK_M, BLOCK_N]` fp32 accumulator via `tl.dot`.

Final accumulator is cast to fp16 and stored.

**Key saving:** `W_fp16` never reaches HBM — it lives in registers for the duration of step 6 and is discarded.  HBM traffic for weights = `N * K / 2` bytes (int8), not `N * K * 2` bytes (fp16).

---

## 5. Parity tolerance

| Comparison | Tolerance | Rationale |
|---|---|---|
| Reference fp16 vs fused fp16 | atol=1e-3, rtol=0 | fp16 accumulation noise; tl.dot uses fp32 accum internally but cast back to fp16 introduces rounding |

The reference (`kernel_w4a16/reference.py`) uses the same packing/dequant math and computes `x @ W_fp16.T` via PyTorch ops.  The kernel matches it within fp16 rounding bounds.

---

## 6. Baselines to beat

| Baseline | Description | Typical bandwidth efficiency |
|---|---|---|
| `bitsandbytes` `LinearFP4` | Row-wise fp4 dequant; no fused GEMM on all GPUs | ~60–80 % of roofline on A100 |
| GPTQ (`exllamav2` / `AutoGPTQ`) | Fused group-dequant GEMM in CUDA; tuned for A100/H100 | 85–95 % on target GPU |
| This kernel (Triton, group_size=128) | Fused int4 dequant+GEMM; portable across architectures | GPU_STEP: to be measured |

The target is to be within 2× of a tuned CUDA GPTQ kernel on the same GPU, demonstrating the approach works before further tuning.

---

## 7. Autotune space

`BLOCK_M` × `BLOCK_N` × `BLOCK_K` with `num_warps`, keyed on `(M, N, K, group_size)`.  Configs require `BLOCK_K >= group_size` and `BLOCK_K % group_size == 0` so that the scale broadcast stride is an integer constexpr.  The autotuner selects the best config on first invocation per shape and caches it.

---

## 8. Namespace and registration

| Attribute | Value |
|---|---|
| Namespace | `w4a16` (distinct from `fused_iq` and `fused_iq_cuda`) |
| Callable | `torch.ops.w4a16.fused_gemm(x, W_packed, scales, group_size)` |
| DEF owner | `torch.library.Library("w4a16", "DEF")` in `kernel_w4a16/register.py` |
| CUDA impl | lazy import of `w4a16_triton.w4a16_fused_gemm` inside `_fused_gemm_cuda` |
| Meta/fake | `@torch.library.register_fake("w4a16::fused_gemm")` — shape propagation only |

`@torch.library.custom_op` decorator is explicitly rejected per repo convention (direct Library API only).

---

## 9. Out of scope (this kernel)

- Zero-point (asymmetric) quantization — symmetric only here.
- Mixed-precision scales (e.g. fp8 scales) — fp16 only.
- Shared-memory weight staging — the int4 tile is small enough for register storage at the tested group sizes.
- CUDA C++ binding — Triton path only in this scaffold; CUDA ext would follow the `kernel_cuda/` pattern from P3.
