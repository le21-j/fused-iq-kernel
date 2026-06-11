"""
Benchmark: eager / compiled / Triton-op / CUDA-ext IQClassifier across (B, L) sweep.
Device: CUDA if available, else CPU.

WARNING: on macOS/CPU torch.compile uses aot_eager (no Triton backend);
timings are PROVISIONAL and will NOT reflect GPU performance.
Triton and CUDA columns are only attempted when torch.cuda.is_available();
each column is guarded independently — a missing CUDA extension produces
"n/a" with a printed note, not a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.compiled import build_compiled
from baseline.reference import IQClassifier

# ---------------------------------------------------------------------------
# Sweep config (overridden by --quick)
# ---------------------------------------------------------------------------
BATCH_SIZES_FULL = [1, 8, 32, 128]
SEQ_LENS_FULL = [1024, 4096, 16384]
BATCH_SIZES_QUICK = [1, 8]
SEQ_LENS_QUICK = [1024]

WARMUP = 10
TIMED = 50
TIMED_LONG = 20  # reduced iters for L=16384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, warmup: int, n: int) -> float:
    # Returns mean latency in milliseconds over n iterations.
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3  # ms


def time_model(model: torch.nn.Module, x: torch.Tensor, n: int) -> float:
    return time_fn(lambda: model(x), WARMUP, n)


# ---------------------------------------------------------------------------
# Dispatch-overhead measurement (CLAUDE.md: measure, never quote fixed µs)
# ---------------------------------------------------------------------------

def measure_dispatch_overhead(
    eager_model: IQClassifier,
    device: torch.device,
) -> Optional[float]:
    """
    Measure per-call overhead of routing through torch.ops.fused_iq.fused_stage
    vs calling the equivalent Python method directly.

    Uses B=1, L=64 (tiny input) so kernel execution time is negligible relative
    to dispatch housekeeping.  Reports the delta in µs as MEASURED on this run.
    Returns None if CUDA unavailable (Triton op not callable on CPU).
    """
    if not torch.cuda.is_available():
        return None

    try:
        import kernel_triton.register  # registers torch.ops.fused_iq.fused_stage
    except Exception as e:
        print(f"  dispatch overhead: skipped (register import failed: {e})")
        return None

    B, L = 1, 64
    x = torch.randn(B, 1, L, dtype=torch.complex64, device=device)
    Wr = eager_model.Wr.data
    Wi = eager_model.Wi.data
    br = eager_model.br.data
    bi = eager_model.bi.data

    N_DISP = 500
    WARMUP_DISP = 50

    # Direct Python wrapper (same arithmetic path, no dispatcher routing)
    def direct():
        eager_model.fused_stage(x)

    # Through torch.ops dispatch
    def via_ops():
        torch.ops.fused_iq.fused_stage(x, Wr, Wi, br, bi)

    direct_ms = time_fn(direct, WARMUP_DISP, N_DISP)
    ops_ms = time_fn(via_ops, WARMUP_DISP, N_DISP)

    delta_us = (ops_ms - direct_ms) * 1e3  # convert ms -> µs
    return delta_us


# ---------------------------------------------------------------------------
# Try to build/load the Triton custom op and CUDA extension
# ---------------------------------------------------------------------------

def _load_triton_op() -> bool:
    """Import register.py (CPU-safe); returns True if op is callable on this device."""
    try:
        import kernel_triton.register  # noqa: F401 — side-effect: registers the op
        return torch.cuda.is_available()
    except Exception as e:
        print(f"  NOTE: Triton op import failed: {e}")
        return False


def _load_cuda_ext() -> bool:
    """Try to import the pre-built CUDA extension; returns True on success."""
    try:
        import fused_iq_cuda_ext  # noqa: F401
        return True
    except ImportError:
        print("  NOTE: fused_iq_cuda_ext not found (run `make build` on a CUDA host); cuda column = n/a")
        return False
    except Exception as e:
        print(f"  NOTE: fused_iq_cuda_ext load error: {e}; cuda column = n/a")
        return False


# ---------------------------------------------------------------------------
# Per-row timing helpers for the two custom ops
# ---------------------------------------------------------------------------

def time_triton_op(model: IQClassifier, x: torch.Tensor, n: int) -> float:
    Wr, Wi = model.Wr.data, model.Wi.data
    br, bi = model.br.data, model.bi.data
    return time_fn(
        lambda: torch.ops.fused_iq.fused_stage(x, Wr, Wi, br, bi),
        WARMUP, n,
    )


def time_cuda_op(model: IQClassifier, x: torch.Tensor, n: int) -> float:
    import fused_iq_cuda_ext  # noqa: F401 — import only registers the op
    Wr, Wi = model.Wr.data, model.Wi.data
    br, bi = model.br.data, model.bi.data
    return time_fn(
        lambda: torch.ops.fused_iq_cuda.fused_stage(x, Wr, Wi, br, bi),
        WARMUP, n,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(quick: bool, json_out: Optional[Path]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = device.type == "cuda"

    print(f"Device: {device}")
    if not cuda_available:
        print(
            "PROVISIONAL: macOS/CPU timings — torch.compile has no Triton backend here "
            "(aot_eager); remote GPU results will differ."
        )
        print("triton/cuda columns: n/a (no CUDA)")
    print()

    batch_sizes = BATCH_SIZES_QUICK if quick else BATCH_SIZES_FULL
    seq_lens = SEQ_LENS_QUICK if quick else SEQ_LENS_FULL

    eager_model = IQClassifier().to(device)
    compiled_model = build_compiled().to(device)

    triton_ok = _load_triton_op() if cuda_available else False
    cuda_ok = _load_cuda_ext() if cuda_available else False

    # Dispatch overhead (MEASURED, not quoted)
    if cuda_available:
        print("Measuring dispatch overhead (B=1, L=64, 500 calls)...")
        delta_us = measure_dispatch_overhead(eager_model, device)
        if delta_us is not None:
            print(f"  torch.ops.fused_iq dispatch overhead vs direct call: {delta_us:+.2f} µs (MEASURED)\n")
        else:
            print("  dispatch overhead: n/a\n")

    # Table header
    hdr = (
        f"{'batch':>6}  {'seqlen':>7}  {'eager_ms':>10}  {'compiled_ms':>12}"
        f"  {'triton_ms':>10}  {'cuda_ms':>9}  {'best_vs_compiled':>17}"
    )
    print(hdr)
    print("-" * len(hdr))

    records = []
    for B in batch_sizes:
        for L in seq_lens:
            x = torch.randn(B, 1, L, dtype=torch.complex64, device=device)
            n_iters = TIMED_LONG if L == 16384 else TIMED

            eager_ms = time_model(eager_model, x, n_iters)
            compiled_ms = time_model(compiled_model, x, n_iters)

            triton_ms: Optional[float] = None
            cuda_ms: Optional[float] = None

            if triton_ok:
                triton_ms = time_triton_op(eager_model, x, n_iters)
            if cuda_ok:
                cuda_ms = time_cuda_op(eager_model, x, n_iters)

            # best of the two hand-written kernels vs compiled
            candidates = [v for v in (triton_ms, cuda_ms) if v is not None]
            best_custom = min(candidates) if candidates else None
            ratio = compiled_ms / best_custom if best_custom and best_custom > 0 else float("nan")

            def fmt(v: Optional[float], w: int) -> str:
                return f"{v:>{w}.3f}" if v is not None else f"{'n/a':>{w}}"

            ratio_str = f"{ratio:>17.2f}x" if ratio == ratio else f"{'n/a':>17}"  # nan check
            print(
                f"{B:>6}  {L:>7}  {eager_ms:>10.3f}  {compiled_ms:>12.3f}"
                f"  {fmt(triton_ms, 10)}  {fmt(cuda_ms, 9)}  {ratio_str}"
            )

            records.append({
                "batch": B,
                "seqlen": L,
                "eager_ms": round(eager_ms, 4),
                "compiled_ms": round(compiled_ms, 4),
                "triton_ms": round(triton_ms, 4) if triton_ms is not None else None,
                "cuda_ms": round(cuda_ms, 4) if cuda_ms is not None else None,
            })

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nResults written to {json_out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IQClassifier 4-way benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sweep: B in {1,8}, L=1024 (for local CI)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Dump results as JSON to this path (e.g. results/bench.json)",
    )
    args = parser.parse_args()

    json_path = Path(args.json) if args.json else None
    run_sweep(quick=args.quick, json_out=json_path)


if __name__ == "__main__":
    main()
