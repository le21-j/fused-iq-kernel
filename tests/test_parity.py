"""
Parity tests: both custom ops vs IQClassifier.fused_stage at atol=1e-4, FP32, on CUDA.

Kernel imports are deferred (inside fixtures/helpers) so this file collects cleanly
on macOS without CUDA or Triton installed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# -- sys.path bootstrap so `from baseline.reference import IQClassifier` works --
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
def model():
    from baseline.reference import IQClassifier
    return IQClassifier(seed=0).cuda()


@pytest.fixture(scope="session")
def triton_op():
    # Importing register.py side-effectfully registers torch.ops.fused_iq.fused_stage
    import kernel_triton.register  # noqa: F401
    return torch.ops.fused_iq.fused_stage


@pytest.fixture(scope="session")
def cuda_op():
    try:
        import fused_iq_cuda_ext  # noqa: F401
    except ImportError:
        pytest.skip("fused_iq_cuda_ext not found — run `make build` first")
    return torch.ops.fused_iq_cuda.fused_stage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_input(B: int, L: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(B, 1, L, dtype=torch.complex64, device="cuda", generator=gen)


def _single_tone(B: int, L: int) -> torch.Tensor:
    """x[b,0,n] = exp(j*2*pi*n/L) — catches I/Q swap and sign errors."""
    n = torch.arange(L, dtype=torch.float32, device="cuda")
    phase = 2 * math.pi * n / L
    xr = torch.cos(phase)
    xi = torch.sin(phase)
    x = torch.complex(xr, xi)           # [L]
    return x.unsqueeze(0).unsqueeze(0).expand(B, 1, L).contiguous()


def _params(model_fixture):
    return (
        model_fixture.Wr.detach(),
        model_fixture.Wi.detach(),
        model_fixture.br.detach(),
        model_fixture.bi.detach(),
    )


# ---------------------------------------------------------------------------
# Parametrized sweep — Triton op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 8, 32, 128])
@pytest.mark.parametrize("L", [1024, 4096, 16384])
def test_triton_parity_random(model, triton_op, B: int, L: int):
    x = _make_random_input(B, L, seed=B * 100000 + L)
    ref = model.fused_stage(x)
    Wr, Wi, br, bi = _params(model)
    out = triton_op(x, Wr, Wi, br, bi)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_triton_parity_single_tone(model, triton_op):
    B, L = 2, 128
    x = _single_tone(B, L)
    ref = model.fused_stage(x)
    Wr, Wi, br, bi = _params(model)
    out = triton_op(x, Wr, Wi, br, bi)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# Parametrized sweep — CUDA C++ op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 8, 32, 128])
@pytest.mark.parametrize("L", [1024, 4096, 16384])
def test_cuda_parity_random(model, cuda_op, B: int, L: int):
    x = _make_random_input(B, L, seed=B * 100000 + L)
    ref = model.fused_stage(x)
    Wr, Wi, br, bi = _params(model)
    out = cuda_op(x, Wr, Wi, br, bi)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_cuda_parity_single_tone(model, cuda_op):
    B, L = 2, 128
    x = _single_tone(B, L)
    ref = model.fused_stage(x)
    Wr, Wi, br, bi = _params(model)
    out = cuda_op(x, Wr, Wi, br, bi)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)
