"""
W4A16 reference: dequantize int4 weight then matmul against fp16 activation.

This is the correctness oracle.  All custom ops in this sub-project are
parity-tested against it at atol=1e-3 (fp16 accumulation is noisier than fp32).

Layout conventions
------------------
- Weights are stored as packed int8: two int4 nibbles per byte.
  low nibble  = element at even column index
  high nibble = element at odd column index
  Packing: byte[i] = (w[2i+1] << 4) | (w[2i] & 0xF)
- Quantization: symmetric per-group with fp16 scales.
  group_size (default 128) columns share one scale per output row.
  w_dequant[out, in] = int4_to_signed(packed[out, in//2]) * scale[out, in//group_size]
- int4 signed range: [-8, 7].  Stored as unsigned nibble with zero_point=8
  (i.e. stored value = true_value + 8, range [0, 15]).  Dequant subtracts 8.

Shapes
------
  W_packed : [N, K//2]     int8  (N=out_features, K=in_features, K even)
  scales   : [N, K//group_size]  fp16
  x        : [M, K]        fp16  (M=batch tokens, K=in_features)
  output   : [M, N]        fp16
"""

import torch


def unpack_int4(W_packed: torch.Tensor) -> torch.Tensor:
    """Unpack [N, K//2] int8 -> [N, K] int8 with values in [-8, 7].

    Packing convention: byte[i] = hi_nibble | lo_nibble
      lo_nibble -> even column, hi_nibble -> odd column.
    """
    N, Khalf = W_packed.shape
    K = Khalf * 2
    # Work in int32 to avoid sign-extension surprises during shifts
    packed = W_packed.to(torch.int32)
    lo = (packed & 0xF) - 8          # even columns: bits [3:0], zero_point=8
    hi = ((packed >> 4) & 0xF) - 8   # odd  columns: bits [7:4], zero_point=8
    # Interleave: col 0=lo, col 1=hi, col 2=lo, ...
    out = torch.empty(N, K, dtype=torch.int8, device=W_packed.device)
    out[:, 0::2] = lo.to(torch.int8)
    out[:, 1::2] = hi.to(torch.int8)
    return out  # [N, K] int8 in [-8, 7]


def dequantize(
    W_packed: torch.Tensor,   # [N, K//2] int8
    scales: torch.Tensor,     # [N, K//group_size] fp16
    group_size: int = 128,
) -> torch.Tensor:
    """Return dequantized weight matrix [N, K] as fp16."""
    N, Khalf = W_packed.shape
    K = Khalf * 2
    n_groups = K // group_size
    assert scales.shape == (N, n_groups), (
        f"scales shape mismatch: got {tuple(scales.shape)}, "
        f"expected ({N}, {n_groups})"
    )

    W_int = unpack_int4(W_packed)                       # [N, K] int8
    W_fp = W_int.to(torch.float16)                     # [N, K] fp16
    # Broadcast scales: [N, n_groups] -> [N, K]
    # Repeat each scale group_size times along the K dimension
    scales_expanded = scales.repeat_interleave(group_size, dim=1)  # [N, K]
    return W_fp * scales_expanded                       # [N, K] fp16


def w4a16_reference(
    x: torch.Tensor,          # [M, K] fp16
    W_packed: torch.Tensor,   # [N, K//2] int8
    scales: torch.Tensor,     # [N, K//group_size] fp16
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize W then compute x @ W.T; return [M, N] fp16.

    This is the naive 2-step reference used for parity checks.
    A fused kernel avoids materialising the [N, K] dequantized weight.
    """
    assert x.dtype == torch.float16, f"x must be fp16, got {x.dtype}"
    assert W_packed.dtype == torch.int8, f"W_packed must be int8, got {W_packed.dtype}"
    assert scales.dtype == torch.float16, f"scales must be fp16, got {scales.dtype}"

    M, K = x.shape
    N, Khalf = W_packed.shape
    assert K == Khalf * 2, f"K={K} must equal 2*Khalf={2*Khalf}"

    W_fp16 = dequantize(W_packed, scales, group_size)  # [N, K] fp16
    return x @ W_fp16.t()                             # [M, N] fp16


def make_quantized_weight(
    N: int,
    K: int,
    group_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random (W_packed, scales) pair for testing.

    Returns
    -------
    W_packed : [N, K//2] int8   packed int4 weights
    scales   : [N, K//group_size] fp16  per-group scales
    """
    assert K % 2 == 0, "K must be even (two int4 per byte)"
    assert K % group_size == 0, "K must be divisible by group_size"
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Random int4 in [0, 15] then convert to signed [-8, 7]
    W_int4 = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int32)
    W_signed = (W_int4 - 8).to(torch.int8)              # [-8, 7]
    # Pack: lo nibble = even col, hi nibble = odd col
    lo = (W_signed[:, 0::2].to(torch.int32) + 8) & 0xF  # back to unsigned [0,15]
    hi = (W_signed[:, 1::2].to(torch.int32) + 8) & 0xF
    packed = (hi << 4) | lo                              # [N, K//2] int32
    W_packed = packed.to(torch.int8)                     # reinterpret bits
    # Random fp16 scales (small positive values)
    n_groups = K // group_size
    scales_fp32 = torch.rand(N, n_groups, generator=g) * 0.1 + 0.01
    scales = scales_fp32.to(torch.float16)
    if device != "cpu":
        W_packed = W_packed.to(device)
        scales = scales.to(device)
    return W_packed, scales
