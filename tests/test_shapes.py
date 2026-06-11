"""
Shape and error-case tests: odd/awkward lengths, edge batches, both kernels.

Kernel imports deferred (fixtures) — file collects cleanly without CUDA/Triton.
K=15, C_out=16, so L_out = L - 14.  L=15 -> L_out=1 (minimum valid length).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# -- sys.path bootstrap --
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# -- Skip entire module if no CUDA --
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K = 15
C_OUT = 16

# ---------------------------------------------------------------------------
# Fixtures (self-contained — no conftest.py dependency)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    from baseline.reference import IQClassifier
    return IQClassifier(seed=0).cuda()


@pytest.fixture(scope="session")
def triton_op():
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

def _make_input(B: int, L: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(B * 9999 + L)
    return torch.randn(B, 1, L, dtype=torch.complex64, device="cuda", generator=gen)


def _params(model_fixture):
    return (
        model_fixture.Wr.detach(),
        model_fixture.Wi.detach(),
        model_fixture.br.detach(),
        model_fixture.bi.detach(),
    )


# ---------------------------------------------------------------------------
# Output-shape + parity sweep
# Odd/awkward L values; L=15 -> L_out=1 (minimum); edge batch sizes
# ---------------------------------------------------------------------------

_ODD_LENGTHS = [15, 16, 17, 31, 127, 1023, 4097]
_EDGE_BATCHES = [1, 2, 3, 5]


@pytest.mark.parametrize("B", _EDGE_BATCHES)
@pytest.mark.parametrize("L", _ODD_LENGTHS)
def test_triton_output_shape_and_parity(model, triton_op, B: int, L: int):
    x = _make_input(B, L)
    Wr, Wi, br, bi = _params(model)
    out = triton_op(x, Wr, Wi, br, bi)
    L_out = L - K + 1
    assert out.shape == (B, C_OUT, L_out), (
        f"triton shape mismatch: got {tuple(out.shape)}, expected ({B},{C_OUT},{L_out})"
    )
    ref = model.fused_stage(x)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("B", _EDGE_BATCHES)
@pytest.mark.parametrize("L", _ODD_LENGTHS)
def test_cuda_output_shape_and_parity(model, cuda_op, B: int, L: int):
    x = _make_input(B, L)
    Wr, Wi, br, bi = _params(model)
    out = cuda_op(x, Wr, Wi, br, bi)
    L_out = L - K + 1
    assert out.shape == (B, C_OUT, L_out), (
        f"cuda shape mismatch: got {tuple(out.shape)}, expected ({B},{C_OUT},{L_out})"
    )
    ref = model.fused_stage(x)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# Error cases: L < K must raise; complex128 input must raise
# ---------------------------------------------------------------------------

def test_triton_raises_L_too_short(model, triton_op):
    x = _make_input(1, 14)  # L=14 < K=15
    Wr, Wi, br, bi = _params(model)
    with pytest.raises(Exception):
        triton_op(x, Wr, Wi, br, bi)


def test_triton_raises_wrong_dtype(model, triton_op):
    x = _make_input(1, 128).to(torch.complex128)
    Wr, Wi, br, bi = _params(model)
    with pytest.raises(Exception):
        triton_op(x, Wr, Wi, br, bi)


def test_cuda_raises_L_too_short(model, cuda_op):
    x = _make_input(1, 14)  # L=14 < K=15
    Wr, Wi, br, bi = _params(model)
    with pytest.raises(Exception):
        cuda_op(x, Wr, Wi, br, bi)


def test_cuda_raises_wrong_dtype(model, cuda_op):
    x = _make_input(1, 128).to(torch.complex128)
    Wr, Wi, br, bi = _params(model)
    with pytest.raises(Exception):
        cuda_op(x, Wr, Wi, br, bi)
