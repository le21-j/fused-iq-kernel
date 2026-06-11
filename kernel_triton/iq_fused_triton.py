"""
Triton kernel: fused complex conv1d + complex bias + |z|^2.

IMPORT NOTE: this module imports triton at module level.  Import it ONLY on a
CUDA host with triton installed.  The registration shim (register.py) uses a
lazy import so the op can be registered — and the op object printed — without
ever importing this file on a CPU-only / macOS machine.

IQ layout: interleaved complex64 — PyTorch stores complex64 as contiguous
(re, im) float32 pairs.  The kernel receives x pre-cast to a float32 view of
shape [B, 1, 2*L], so adjacent threads read adjacent float2 pairs, giving
fully coalesced 8-byte loads.
"""

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 64},  num_warps=2),
        triton.Config({"BLOCK_L": 64},  num_warps=4),
        triton.Config({"BLOCK_L": 128}, num_warps=2),
        triton.Config({"BLOCK_L": 128}, num_warps=4),
        triton.Config({"BLOCK_L": 128}, num_warps=8),
        triton.Config({"BLOCK_L": 256}, num_warps=4),
        triton.Config({"BLOCK_L": 256}, num_warps=8),
        triton.Config({"BLOCK_L": 512}, num_warps=4),
        triton.Config({"BLOCK_L": 512}, num_warps=8),
    ],
    key=["L_out"],
)
@triton.jit
def _iq_fused_kernel(
    x_ptr,    # float32, [B, 1, 2*L]  — interleaved re/im
    Wr_ptr,   # float32, [C_out, 1, K]
    Wi_ptr,   # float32, [C_out, 1, K]
    br_ptr,   # float32, [C_out]
    bi_ptr,   # float32, [C_out]
    out_ptr,  # float32, [B, C_out, L_out]
    B: tl.constexpr,
    C_out: tl.constexpr,
    L: tl.constexpr,       # original complex length
    L_out: tl.constexpr,   # L - K + 1
    K: tl.constexpr,       # filter length (15)
    BLOCK_L: tl.constexpr,
):
    # -----------------------------------------------------------------------
    # Grid decomposition: grid = (B * C_out, cdiv(L_out, BLOCK_L))
    # pid0 encodes both the batch index and output channel.
    # -----------------------------------------------------------------------
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b  = pid0 // C_out   # batch index
    co = pid0 % C_out    # output channel index

    l0 = pid1 * BLOCK_L                          # first output position for this tile
    l_offsets = l0 + tl.arange(0, BLOCK_L)      # [BLOCK_L] output positions
    mask = l_offsets < L_out

    # -----------------------------------------------------------------------
    # Load bias scalars (one scalar per program — cheap broadcast)
    # -----------------------------------------------------------------------
    br = tl.load(br_ptr + co)
    bi = tl.load(bi_ptr + co)

    # -----------------------------------------------------------------------
    # Accumulate over filter taps k = 0 .. K-1.
    # Input base: x[b, 0, 2*(l+k)]  and  x[b, 0, 2*(l+k)+1]
    # x is stored as float32 interleaved, so:
    #   re = x_ptr + b*(2*L) + 2*(l+k)
    #   im = x_ptr + b*(2*L) + 2*(l+k)+1
    # Adjacent l_offsets are adjacent in memory => coalesced 32-bit loads.
    # We read re and im as two separate BLOCK_L-wide vectors per k, which the
    # hardware issues as two sequential coalesced transactions.
    # -----------------------------------------------------------------------
    x_base = b * (2 * L)   # byte offset in float elements to start of batch b

    acc_r = tl.zeros([BLOCK_L], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Weight pointers for this output channel: Wr[co, 0, :], Wi[co, 0, :]
    w_base = co * K  # Wr/Wi have shape [C_out, 1, K] stored C-contiguous

    # Sequential accumulation over k — mirrors conv1d left-to-right order.
    # Using a Python-level for-loop with tl.constexpr bounds forces unrolling.
    for k in range(K):
        # Interleaved addresses: even index = real, odd = imag
        re_offsets = x_base + 2 * (l_offsets + k)
        im_offsets = x_base + 2 * (l_offsets + k) + 1

        xr = tl.load(x_ptr + re_offsets, mask=mask, other=0.0)
        xi = tl.load(x_ptr + im_offsets, mask=mask, other=0.0)

        wr = tl.load(Wr_ptr + w_base + k)   # scalar broadcast
        wi = tl.load(Wi_ptr + w_base + k)

        # Complex multiply-accumulate: (xr + j*xi) * (wr + j*wi)
        acc_r += xr * wr - xi * wi
        acc_i += xr * wi + xi * wr

    # -----------------------------------------------------------------------
    # Complex bias add then squared-magnitude
    # -----------------------------------------------------------------------
    yr = acc_r + br
    yi = acc_i + bi
    result = yr * yr + yi * yi   # |z|^2, real float32

    # -----------------------------------------------------------------------
    # Store to out[b, co, l0:l0+BLOCK_L]  — contiguous fp32 output
    # -----------------------------------------------------------------------
    out_base = (b * C_out + co) * L_out
    tl.store(out_ptr + out_base + l_offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def iq_fused_triton(
    x: torch.Tensor,   # complex64 [B, 1, L]  on CUDA
    Wr: torch.Tensor,  # float32  [C_out, 1, K]
    Wi: torch.Tensor,  # float32  [C_out, 1, K]
    br: torch.Tensor,  # float32  [C_out]
    bi: torch.Tensor,  # float32  [C_out]
) -> torch.Tensor:
    """Launch the fused Triton kernel; return real float32 [B, C_out, L_out]."""
    # --- dtype / shape validation -------------------------------------------
    assert x.dtype == torch.complex64,  f"x must be complex64, got {x.dtype}"
    assert Wr.dtype == torch.float32,   f"Wr must be float32, got {Wr.dtype}"
    assert Wi.dtype == torch.float32,   f"Wi must be float32, got {Wi.dtype}"
    assert br.dtype == torch.float32,   f"br must be float32, got {br.dtype}"
    assert bi.dtype == torch.float32,   f"bi must be float32, got {bi.dtype}"

    assert x.ndim == 3 and x.shape[1] == 1, \
        f"x must be [B, 1, L], got {list(x.shape)}"
    assert Wr.ndim == 3 and Wi.ndim == 3, \
        f"Wr/Wi must be [C_out, 1, K], got {list(Wr.shape)} / {list(Wi.shape)}"
    assert Wr.shape == Wi.shape, \
        f"Wr and Wi must have identical shape, got {list(Wr.shape)} vs {list(Wi.shape)}"
    assert br.ndim == 1 and bi.ndim == 1 and br.shape == bi.shape, \
        f"br/bi must be 1-D same shape, got {list(br.shape)} / {list(bi.shape)}"
    assert Wr.shape[0] == br.shape[0], \
        f"C_out mismatch: Wr={Wr.shape[0]}, br={br.shape[0]}"

    # --- contiguity ---------------------------------------------------------
    assert x.is_contiguous(),  "x must be contiguous"
    assert Wr.is_contiguous(), "Wr must be contiguous"
    assert Wi.is_contiguous(), "Wi must be contiguous"
    assert br.is_contiguous(), "br must be contiguous"
    assert bi.is_contiguous(), "bi must be contiguous"

    B, _, L = x.shape
    C_out, _, K = Wr.shape
    L_out = L - K + 1
    assert L_out > 0, f"L={L} too short for K={K}"

    # --- reinterpret complex64 as float32 view [B, 1, 2*L] ------------------
    # torch.view_as_real returns [B, 1, L, 2]; reshape to [B, 1, 2*L]
    x_f = torch.view_as_real(x).reshape(B, 1, 2 * L).contiguous()

    # Flatten weight second dim (C_out, 1, K) -> (C_out, K) — the kernel
    # only needs C_out * K scalars, indexed as co*K + k.
    Wr_flat = Wr.squeeze(1).contiguous()  # [C_out, K]
    Wi_flat = Wi.squeeze(1).contiguous()

    # --- allocate output ----------------------------------------------------
    out = torch.empty((B, C_out, L_out), dtype=torch.float32, device=x.device)

    # --- launch -------------------------------------------------------------
    grid = lambda meta: (B * C_out, triton.cdiv(L_out, meta["BLOCK_L"]))
    _iq_fused_kernel[grid](
        x_f, Wr_flat, Wi_flat, br, bi, out,
        B=B, C_out=C_out, L=L, L_out=L_out, K=K,
    )
    return out
