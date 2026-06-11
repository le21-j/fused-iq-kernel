"""
Register torch.ops.fused_iq.fused_stage via the direct torch.library.Library API.

Design constraint (CLAUDE.md): @torch.library.custom_op decorator API is out of
scope.  We use torch.library.Library + lib.impl + torch.library.register_fake
exclusively.

Lazy-import guarantee: triton is NOT imported at module load time.  The CUDA
implementation function imports iq_fused_triton only when the op is actually
called.  This means the module can be imported — and the op object inspected —
on a CPU-only / macOS machine that does not have triton installed.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 1. Define the op schema in the DEF library fragment
# ---------------------------------------------------------------------------
lib = torch.library.Library("fused_iq", "DEF")
lib.define(
    "fused_stage(Tensor x, Tensor Wr, Tensor Wi, Tensor br, Tensor bi) -> Tensor"
)


# ---------------------------------------------------------------------------
# 2. CUDA implementation — lazy-imports the Triton kernel
# ---------------------------------------------------------------------------

def _fused_stage_cuda(
    x: torch.Tensor,
    Wr: torch.Tensor,
    Wi: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
) -> torch.Tensor:
    from kernel_triton.iq_fused_triton import iq_fused_triton  # lazy import
    return iq_fused_triton(x, Wr, Wi, br, bi)


lib.impl("fused_stage", _fused_stage_cuda, "CUDA")


# ---------------------------------------------------------------------------
# 3. register_fake meta-kernel (abstract impl for torch.compile / export)
# ---------------------------------------------------------------------------

@torch.library.register_fake("fused_iq::fused_stage")
def _fused_stage_fake(
    x: torch.Tensor,
    Wr: torch.Tensor,
    Wi: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
) -> torch.Tensor:
    # Shapes derived from inputs — nothing hard-coded except dtype.
    B = x.shape[0]
    C_out = Wr.shape[0]
    L = x.shape[2]
    K = Wr.shape[2]
    L_out = L - K + 1
    return x.new_empty((B, C_out, L_out), dtype=torch.float32)
