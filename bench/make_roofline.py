"""
make_roofline.py — Roofline plot for the two hand-written IQ kernels.

Reads bench.json (produced by benchmark.py --json), accepts peak hardware
numbers via CLI args (NOT hard-coded arch tables — the operator supplies
datasheet values for whatever nvidia-smi reports on the target host).

Usage (on the GPU host after running the benchmark):
    python bench/make_roofline.py \\
        --json results/bench.json \\
        --gpu "NVIDIA A100-SXM4-80GB" \\
        --peak-bw-gbs 2039 \\
        --peak-tflops 312 \\
        --out results/roofline.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _arithmetic_intensity(record: dict) -> float:
    """
    Estimate arithmetic intensity (FLOP/byte) for the fused stage on one row.

    Fused stage: complex conv1d (C_in=1, C_out=16, K=15) + bias + |z|^2.
    Per output element:
      - Complex multiply-accumulate: 2 real MACs per tap => 8 FLOP/tap (add+mul, real+imag)
        More precisely: yr += xr*wr - xi*wi  (2 mul + 1 sub)
                        yi += xr*wi + xi*wr  (2 mul + 1 add)
        = 6 FLOP per tap, times 2 real input channels (Wr, Wi separately):
        actually K taps x 6 FLOP = 6K FLOP per output channel per output sample.
        C_out=16, K=15 => 16 * 15 * 6 = 1440 FLOP per output time-step.
      - Bias add: 2 FLOP (real, imag) per output element: +2*16 = 32 FLOP per step.
      - |z|^2: 2 mul + 1 add per output element: +3*16 = 48 FLOP per step.
    Total per output time-step: 1440 + 32 + 48 = 1520 FLOP.
    L_out = L - K + 1 = L - 14.
    Total FLOP = B * L_out * 1520.

    Memory traffic (bytes, assuming no cache reuse):
      Reads: x complex64 [B,1,L] = B*L*8 bytes
             Wr float32 [16,1,15] = 16*15*4 = 960 bytes (small, likely cached)
             Wi float32 [16,1,15] = 960 bytes
             br, bi [16] = 64+64 = 128 bytes
      Writes: output float32 [B,16,L_out] = B*16*L_out*4 bytes
    Dominant terms for large L: x reads + output writes.
    """
    B = record["batch"]
    L = record["seqlen"]
    C_out, K = 16, 15
    L_out = L - K + 1

    flop = B * L_out * 1520.0
    # bytes: x read + output write (weight tensors small — counted but minor)
    bytes_moved = (
        B * 1 * L * 8           # x complex64
        + 2 * C_out * 1 * K * 4 # Wr + Wi float32
        + 2 * C_out * 4          # br + bi float32
        + B * C_out * L_out * 4  # output float32
    )
    return flop / bytes_moved


def _achieved_gflops(record: dict, key: str) -> float | None:
    """Convert millisecond timing + FLOP estimate to achieved GFLOP/s."""
    ms = record.get(key)
    if ms is None:
        return None
    B = record["batch"]
    L = record["seqlen"]
    L_out = L - 15 + 1
    flop = B * L_out * 1520.0
    seconds = ms / 1e3
    return flop / seconds / 1e9


def plot(
    records: list[dict],
    gpu_name: str,
    peak_bw_gbs: float,
    peak_tflops: float,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — cannot generate roofline plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Roofline ceiling
    # Ridge point: I* = peak_tflops*1e12 / (peak_bw_gbs*1e9) FLOP/byte
    ridge = (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)
    intensities_range = [1e-2, 1e3]

    import numpy as np
    x_roof = np.logspace(
        np.log10(intensities_range[0]), np.log10(intensities_range[1]), 500
    )
    y_roof = np.minimum(
        peak_bw_gbs * x_roof,       # memory-bandwidth ceiling (GFLOP/s = GB/s * FLOP/byte)
        peak_tflops * 1e3,           # compute ceiling (TFLOP/s -> GFLOP/s)
    )
    ax.loglog(x_roof, y_roof, "k-", linewidth=2, label="Roofline ceiling")
    ax.axvline(ridge, color="gray", linestyle="--", linewidth=1, label=f"Ridge I*={ridge:.1f} FLOP/B")

    markers = {"triton_ms": ("o", "tab:blue", "Triton kernel"), "cuda_ms": ("s", "tab:orange", "CUDA kernel")}
    for key, (marker, color, label) in markers.items():
        xs, ys = [], []
        for rec in records:
            ai = _arithmetic_intensity(rec)
            perf = _achieved_gflops(rec, key)
            if perf is not None:
                xs.append(ai)
                ys.append(perf)
        if xs:
            ax.scatter(xs, ys, marker=marker, color=color, s=60, zorder=5, label=label)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (GFLOP/s)", fontsize=12)
    ax.set_title(f"Roofline — fused IQ stage\nGPU: {gpu_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Roofline plot written to {out_path}  (GPU: {gpu_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate roofline plot from bench.json")
    parser.add_argument("--json", required=True, metavar="PATH", help="Path to bench.json")
    parser.add_argument("--gpu", required=True, metavar="STRING",
                        help="Exact GPU model string from nvidia-smi (embedded in plot title)")
    parser.add_argument("--peak-bw-gbs", required=True, type=float, metavar="GB/s",
                        help="Peak memory bandwidth in GB/s (from datasheet)")
    parser.add_argument("--peak-tflops", required=True, type=float, metavar="TFLOP/s",
                        help="Peak FP32 compute in TFLOP/s (from datasheet)")
    parser.add_argument("--out", default="results/roofline.png", metavar="PATH",
                        help="Output PNG path (default: results/roofline.png)")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run benchmark.py --json first.", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        records = json.load(f)

    plot(
        records=records,
        gpu_name=args.gpu,
        peak_bw_gbs=args.peak_bw_gbs,
        peak_tflops=args.peak_tflops,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
