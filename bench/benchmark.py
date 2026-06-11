"""
Benchmark: eager vs compiled IQClassifier across (B, L) sweep.
Device: CUDA if available, else CPU.
WARNING: on macOS/CPU torch.compile uses aot_eager (no Triton backend);
timings are PROVISIONAL and will NOT reflect GPU performance.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.compiled import build_compiled
from baseline.reference import IQClassifier

BATCH_SIZES = [1, 8, 32, 128]
SEQ_LENS = [1024, 4096, 16384]
WARMUP = 10
TIMED = 50
TIMED_LONG = 20  # reduced iters for L=16384 to keep CPU runtime reasonable


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_model(model: torch.nn.Module, x: torch.Tensor, n: int) -> float:
    # Returns mean latency in milliseconds over n iterations.
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            _ = model(x)
        sync()
        t1 = time.perf_counter()
    return (t1 - t0) / n * 1e3  # ms


def run_sweep() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print(
            "PROVISIONAL: macOS/CPU timings — torch.compile has no Triton backend here "
            "(aot_eager); remote GPU results will differ."
        )
    print()

    eager_model = IQClassifier().to(device)
    compiled_model = build_compiled().to(device)

    # Table header
    print(f"{'batch':>6}  {'seqlen':>7}  {'eager_ms':>10}  {'compiled_ms':>12}  {'speedup':>8}")
    print("-" * 55)

    for B in BATCH_SIZES:
        for L in SEQ_LENS:
            x = torch.randn(B, 1, L, dtype=torch.complex64, device=device)
            n_iters = TIMED_LONG if L == 16384 else TIMED
            eager_ms = time_model(eager_model, x, n_iters)
            compiled_ms = time_model(compiled_model, x, n_iters)
            speedup = eager_ms / compiled_ms if compiled_ms > 0 else float("nan")
            print(f"{B:>6}  {L:>7}  {eager_ms:>10.3f}  {compiled_ms:>12.3f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    run_sweep()
