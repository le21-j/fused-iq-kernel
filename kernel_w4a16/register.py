"""
Register torch.ops.w4a16.fused_gemm via the direct torch.library.Library API.

Design constraints (CLAUDE.md):
  - @torch.library.custom_op decorator API is out of scope.
  - Use torch.library.Library + lib.define + lib.impl + register_fake exclusively.
  - Triton is NOT imported at module level; lazy import in the CUDA impl path.
  - This file imports cleanly on a CPU-only host (no CUDA, no Triton).

Op signature
------------
  w4a16::fused_gemm(Tensor x, Tensor W_packed, Tensor scales, int group_size) -> Tensor
    x        : fp16 [M, K]
    W_packed : int8 [N, K//2]   — packed int4
    scales   : fp16 [N, K//group_size]
    group_size: int (default 128; must be passed explicitly here since torch.library
                requires a concrete schema — default values not supported in schema DSL)
  returns fp16 [M, N]
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 1. DEF: own the "w4a16" namespace and declare the op schema
# ---------------------------------------------------------------------------
lib = torch.library.Library("w4a16", "DEF")
lib.define(
    "fused_gemm(Tensor x, Tensor W_packed, Tensor scales, int group_size) -> Tensor"
)


# ---------------------------------------------------------------------------
# 2. CUDA implementation — lazy-imports the Triton kernel
# ---------------------------------------------------------------------------

def _fused_gemm_cuda(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    from kernel_w4a16.w4a16_triton import w4a16_fused_gemm  # lazy import
    return w4a16_fused_gemm(x, W_packed, scales, group_size)


lib.impl("fused_gemm", _fused_gemm_cuda, "CUDA")


# ---------------------------------------------------------------------------
# 3. register_fake meta-kernel (shape propagation for torch.compile / export)
# ---------------------------------------------------------------------------

@torch.library.register_fake("w4a16::fused_gemm")
def _fused_gemm_fake(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    # x: [M, K], W_packed: [N, K//2] -> output: [M, N] fp16
    M = x.shape[0]
    N = W_packed.shape[0]
    return x.new_empty((M, N), dtype=torch.float16)
