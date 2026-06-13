"""
Parity tests: w4a16 fused GEMM op vs kernel_w4a16.reference.w4a16_reference at
atol=1e-3 fp16, on CUDA.

Kernel imports are deferred (inside fixtures) so this file collects cleanly
on a CPU-only host without Triton installed.

Parity tolerance rationale
--------------------------
fp16 accumulation introduces ~0.5–1 ULP per addition relative to fp32.
For group_size=128 and typical K values used here (128..1024), the max
absolute error between the fused kernel (fp32 tl.dot accumulator, fp16 store)
and the reference (fp16 @ fp16 via PyTorch) is bounded empirically below 1e-3.
A tolerance of atol=1e-3 is therefore the parity contract for W4A16, matching
the fp16 output dtype (compare: the IQ kernel uses atol=1e-4 with fp32 outputs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# -- sys.path bootstrap so `from kernel_w4a16.reference import ...` works --
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# -- Skip entire module if no CUDA --
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ref_fn():
    """Return the reference dequant+GEMM callable (pure PyTorch)."""
    from kernel_w4a16.reference import w4a16_reference
    return w4a16_reference


@pytest.fixture(scope="session")
def fused_op():
    """Register torch.ops.w4a16.fused_gemm and return it."""
    import kernel_w4a16.register  # noqa: F401 — side-effect: registers op
    return torch.ops.w4a16.fused_gemm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    M: int, N: int, K: int, group_size: int = 128, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_fp16, W_packed_int8, scales_fp16) on CUDA."""
    from kernel_w4a16.reference import make_quantized_weight
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.float32, generator=g).to(torch.float16).cuda()
    W_packed, scales = make_quantized_weight(N, K, group_size=group_size, seed=seed + 1)
    return x, W_packed.cuda(), scales.cuda()


# ---------------------------------------------------------------------------
# Shape sweep — Triton fused op vs reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("K", [128, 256, 512])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("M", [1, 8, 32])
def test_fused_gemm_parity(ref_fn, fused_op, M: int, N: int, K: int, group_size: int):
    x, W_packed, scales = _make_inputs(M, N, K, group_size=group_size, seed=M * 1000 + N + K)
    ref = ref_fn(x, W_packed, scales, group_size)
    out = fused_op(x, W_packed, scales, group_size)
    assert out.shape == ref.shape == (M, N), f"shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# Single-element sanity check: one output row, one group
# ---------------------------------------------------------------------------

def test_fused_gemm_single_group(ref_fn, fused_op):
    """M=1, N=1, K=group_size — minimal sanity."""
    from kernel_w4a16.reference import make_quantized_weight
    group_size = 128
    K = 128
    W_packed, scales = make_quantized_weight(1, K, group_size=group_size, seed=99)
    x = torch.ones(1, K, dtype=torch.float16, device="cuda")
    W_packed = W_packed.cuda()
    scales = scales.cuda()
    ref = ref_fn(x, W_packed, scales, group_size)
    out = fused_op(x, W_packed, scales, group_size)
    assert out.shape == (1, 1)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# Output dtype check
# ---------------------------------------------------------------------------

def test_output_dtype(fused_op):
    x, W_packed, scales = _make_inputs(4, 32, 128)
    out = fused_op(x, W_packed, scales, 128)
    assert out.dtype == torch.float16, f"expected fp16, got {out.dtype}"


# ---------------------------------------------------------------------------
# register_fake / meta-kernel: shape propagation works without GPU execution
# ---------------------------------------------------------------------------

def test_register_fake_shape():
    """register_fake should propagate shapes under FakeTensorMode."""
    import kernel_w4a16.register  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode
    M, N, K, group_size = 8, 64, 128, 128
    with FakeTensorMode():
        x = torch.empty(M, K, dtype=torch.float16)
        W_packed = torch.empty(N, K // 2, dtype=torch.int8)
        scales = torch.empty(N, K // group_size, dtype=torch.float16)
        out = torch.ops.w4a16.fused_gemm(x, W_packed, scales, group_size)
    assert tuple(out.shape) == (M, N)
    assert out.dtype == torch.float16
