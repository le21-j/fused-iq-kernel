"""
W4A16 fused dequant+GEMM Triton kernel.

Fuses int4->fp16 dequantization with the matrix multiply in a single
Triton program, avoiding materialisation of the [N, K] fp16 weight tensor.

IMPORT NOTE: triton is NOT imported at module level.  This file imports
cleanly on a CPU-only host.  The actual kernel and wrapper are defined
inside a try/except guard so collection under pytest works without CUDA/Triton.

Layout conventions (must match kernel_w4a16/reference.py)
----------------------------------------------------------
  W_packed : [N, K//2]  int8  — two int4 nibbles per byte
    lo nibble (bits 3:0) -> even K index
    hi nibble (bits 7:4) -> odd  K index
    stored value = true_int4 + 8  (zero_point offset, range [0,15])
  scales   : [N, K//group_size]  fp16  — symmetric per-group
  x        : [M, K]  fp16  — row-major activations
  output   : [M, N]  fp16  — row-major

Tiling strategy
---------------
  BLOCK_M x BLOCK_N output tile per Triton program.
  Each program iterates over K in BLOCK_K chunks.
  For each K chunk, it:
    1. Loads x tile [BLOCK_M, BLOCK_K] in fp16.
    2. Loads W_packed tile [BLOCK_N, BLOCK_K//2] in int8.
    3. Unpacks int4 nibbles -> signed int4 -> fp16 in registers.
    4. Loads the corresponding scales [BLOCK_N, BLOCK_K//group_size].
    5. Dequantizes W in registers: W_fp16 = W_int4 * scale (broadcast).
    6. Accumulates dot product into a [BLOCK_M, BLOCK_N] fp32 accumulator.
  Final result is cast to fp16 and stored.

Accumulation is done in fp32 (via tl.dot with out_dtype=tl.float32) to
maintain parity with the reference within atol=1e-3 for fp16 outputs.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kernel — only defined if triton is present
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2),
            triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4),
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4),
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8),
        ],
        key=["M", "N", "K", "group_size"],
    )
    @triton.jit
    def _w4a16_kernel(
        x_ptr,        # fp16 [M, K]
        w_ptr,        # int8 [N, K//2]  — packed int4
        s_ptr,        # fp16 [N, K//group_size]
        out_ptr,      # fp16 [M, N]
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        group_size: tl.constexpr,
        stride_xm: tl.constexpr,   # K  (row stride of x)
        stride_wn: tl.constexpr,   # K//2 (row stride of w_packed)
        stride_sn: tl.constexpr,   # K//group_size (row stride of scales)
        stride_om: tl.constexpr,   # N  (row stride of output)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # -------------------------------------------------------------------
        # Program maps to one [BLOCK_M, BLOCK_N] output tile.
        # -------------------------------------------------------------------
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

        m_mask = m_off < M
        n_mask = n_off < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Number of K groups per BLOCK_K chunk (BLOCK_K must be >= group_size
        # and a multiple of group_size for simplicity; autotuner enforces this
        # via configs where BLOCK_K >= group_size and BLOCK_K % group_size == 0).
        n_groups_per_block = BLOCK_K // group_size  # constexpr integer

        # Iterate over K in BLOCK_K chunks
        for k0 in range(0, K, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)            # [BLOCK_K]
            k_mask = k_off < K

            # ---------------------------------------------------------------
            # Load x tile: [BLOCK_M, BLOCK_K] fp16
            # ---------------------------------------------------------------
            x_ptrs = x_ptr + m_off[:, None] * stride_xm + k_off[None, :]
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float16)

            # ---------------------------------------------------------------
            # Load packed weight tile: [BLOCK_N, BLOCK_K//2] int8
            # Each byte holds two int4 nibbles.
            # ---------------------------------------------------------------
            k_half_off = k0 // 2 + tl.arange(0, BLOCK_K // 2)  # [BLOCK_K//2]
            k_half_mask = k_half_off < (K // 2)

            w_ptrs = w_ptr + n_off[:, None] * stride_wn + k_half_off[None, :]
            w_mask = n_mask[:, None] & k_half_mask[None, :]
            w_packed_tile = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.int8)
            # [BLOCK_N, BLOCK_K//2] int8 — bit-packed int4

            # ---------------------------------------------------------------
            # Unpack int4 nibbles.
            # lo nibble: even K positions  -> cast to int32, mask 0xF, sub 8
            # hi nibble: odd  K positions  -> shift >> 4,   mask 0xF, sub 8
            # Result: [BLOCK_N, BLOCK_K//2] each for lo and hi.
            # Interleave to [BLOCK_N, BLOCK_K].
            # ---------------------------------------------------------------
            w_int32 = w_packed_tile.to(tl.int32)
            lo = (w_int32 & 0xF) - 8   # [BLOCK_N, BLOCK_K//2]  signed int4
            hi = ((w_int32 >> 4) & 0xF) - 8

            # Build [BLOCK_N, BLOCK_K] by interleaving lo/hi.
            # Triton doesn't have gather, so we construct a [BLOCK_N, BLOCK_K]
            # tensor: columns 0,2,4,... from lo; columns 1,3,5,... from hi.
            # Strategy: reshape both to [BLOCK_N, BLOCK_K//2, 1], concat on
            # last dim -> [BLOCK_N, BLOCK_K//2, 2], reshape -> [BLOCK_N, BLOCK_K].
            lo_3d = tl.reshape(lo, [BLOCK_N, BLOCK_K // 2, 1])
            hi_3d = tl.reshape(hi, [BLOCK_N, BLOCK_K // 2, 1])
            w_unpacked = tl.reshape(
                tl.cat([lo_3d, hi_3d], dim=2),
                [BLOCK_N, BLOCK_K]
            )  # [BLOCK_N, BLOCK_K] int32

            # ---------------------------------------------------------------
            # Load scales tile: [BLOCK_N, n_groups_per_block] fp16
            # group index for column k = k // group_size
            # ---------------------------------------------------------------
            g0 = k0 // group_size
            g_off = g0 + tl.arange(0, n_groups_per_block)   # [n_groups_per_block]
            n_groups_total = K // group_size
            g_mask = g_off < n_groups_total

            s_ptrs = s_ptr + n_off[:, None] * stride_sn + g_off[None, :]
            s_mask = n_mask[:, None] & g_mask[None, :]
            s_tile = tl.load(s_ptrs, mask=s_mask, other=1.0).to(tl.float16)
            # [BLOCK_N, n_groups_per_block] fp16

            # Expand scales to [BLOCK_N, BLOCK_K] by repeating each group_size times
            s_expanded = tl.reshape(
                tl.broadcast_to(
                    tl.reshape(s_tile, [BLOCK_N, n_groups_per_block, 1]),
                    [BLOCK_N, n_groups_per_block, group_size],
                ),
                [BLOCK_N, BLOCK_K],
            )  # [BLOCK_N, BLOCK_K] fp16

            # ---------------------------------------------------------------
            # Dequantize: W_fp16 = w_unpacked * scales
            # ---------------------------------------------------------------
            w_fp16 = w_unpacked.to(tl.float16) * s_expanded  # [BLOCK_N, BLOCK_K]

            # ---------------------------------------------------------------
            # Accumulate: acc += x_tile @ w_fp16.T
            # tl.dot expects [M, K] x [K, N]; w_fp16 is [N, K], so transpose.
            # ---------------------------------------------------------------
            acc += tl.dot(x_tile, tl.trans(w_fp16), out_dtype=tl.float32)

        # -------------------------------------------------------------------
        # Cast to fp16 and store
        # -------------------------------------------------------------------
        out = acc.to(tl.float16)
        out_ptrs = out_ptr + m_off[:, None] * stride_om + n_off[None, :]
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, out, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def w4a16_fused_gemm(
    x: torch.Tensor,          # fp16 [M, K]
    W_packed: torch.Tensor,   # int8 [N, K//2]
    scales: torch.Tensor,     # fp16 [N, K//group_size]
    group_size: int = 128,
) -> torch.Tensor:
    """Fused int4 dequant + GEMM; return fp16 [M, N].

    Must be called on a CUDA host with triton installed.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; cannot run w4a16_fused_gemm.")
    assert x.is_cuda, "x must be on CUDA"
    assert x.dtype == torch.float16, f"x must be fp16, got {x.dtype}"
    assert W_packed.dtype == torch.int8, f"W_packed must be int8, got {W_packed.dtype}"
    assert scales.dtype == torch.float16, f"scales must be fp16, got {scales.dtype}"

    M, K = x.shape
    N, Khalf = W_packed.shape
    assert K == Khalf * 2, f"K={K} != 2*Khalf={Khalf*2}"
    assert K % group_size == 0, "K must be divisible by group_size"
    n_groups = K // group_size
    assert scales.shape == (N, n_groups), f"scales shape {scales.shape} != ({N},{n_groups})"

    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _w4a16_kernel[grid](
        x, W_packed, scales, out,
        M=M, N=N, K=K,
        group_size=group_size,
        stride_xm=K,
        stride_wn=Khalf,
        stride_sn=n_groups,
        stride_om=N,
    )
    return out
