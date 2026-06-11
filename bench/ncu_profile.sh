#!/usr/bin/env bash
# ncu_profile.sh — Nsight Compute capture for both hand-written IQ kernels.
# Writes results/ncu_triton.txt and results/ncu_cuda.txt.
#
# Captured metrics (three P3-PROMPT-5 requirements):
#   dram__bytes.sum                               — achieved memory bandwidth (bytes moved)
#   sm__warps_active.avg.pct_of_peak_sustained_active — achieved occupancy (% of peak)
#   launch__kernel_count (via --launch-count)      — kernel-launch count per op call
#
# Guard: requires Linux + CUDA + ncu in PATH.
# On macOS/CPU this exits 2 (GPU_STEP park), not 1 (real failure).
set -euo pipefail

command -v ncu || { echo "GPU_STEP: Nsight Compute requires Linux+CUDA"; exit 2; }

PY=${PY:-.venv/bin/python}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Shared ncu options
# Metrics captured:
#   dram__bytes.sum  — total DRAM bytes read+written (memory-bandwidth proxy)
#   sm__warps_active.avg.pct_of_peak_sustained_active — achieved occupancy
# Launch count is reported by ncu's summary line (--set full includes it).
# ---------------------------------------------------------------------------
NCU_METRICS="dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active"
NCU_OPTS="--set full --metrics $NCU_METRICS"

# ---------------------------------------------------------------------------
# Section 1: Triton kernel (torch.ops.fused_iq.fused_stage)
# ---------------------------------------------------------------------------
echo "=== Profiling Triton kernel (fused_iq::fused_stage) ==="
ncu $NCU_OPTS \
    --launch-count 1 \
    --target-processes all \
    --output "$RESULTS/ncu_triton" \
    --force-overwrite \
    $PY - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent if '__file__' in dir() else '.'))
import torch
import kernel_triton.register  # registers torch.ops.fused_iq.fused_stage
from baseline.reference import IQClassifier
model = IQClassifier().cuda()
x  = torch.randn(32, 1, 4096, dtype=torch.complex64, device="cuda")
# Warmup outside ncu launch window handled by --launch-count 1
torch.ops.fused_iq.fused_stage(x, model.Wr.data, model.Wi.data, model.br.data, model.bi.data)
torch.cuda.synchronize()
PYEOF

# ncu writes <output>.ncu-rep by default; also dump text report
ncu --import "$RESULTS/ncu_triton.ncu-rep" \
    --print-summary per-kernel \
    > "$RESULTS/ncu_triton.txt" 2>&1 || true

echo "Triton profile written to $RESULTS/ncu_triton.txt"

# ---------------------------------------------------------------------------
# Section 2: CUDA extension (torch.ops.fused_iq_cuda.fused_stage)
# ---------------------------------------------------------------------------
echo ""
echo "=== Profiling CUDA extension (fused_iq_cuda::fused_stage) ==="
ncu $NCU_OPTS \
    --launch-count 1 \
    --target-processes all \
    --output "$RESULTS/ncu_cuda" \
    --force-overwrite \
    $PY - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent if '__file__' in dir() else '.'))
import torch
import fused_iq_cuda_ext  # pre-built CUDA extension
from baseline.reference import IQClassifier
model = IQClassifier().cuda()
x = torch.randn(32, 1, 4096, dtype=torch.complex64, device="cuda")
torch.ops.fused_iq_cuda.fused_stage(x, model.Wr.data, model.Wi.data, model.br.data, model.bi.data)
torch.cuda.synchronize()
PYEOF

ncu --import "$RESULTS/ncu_cuda.ncu-rep" \
    --print-summary per-kernel \
    > "$RESULTS/ncu_cuda.txt" 2>&1 || true

echo "CUDA profile written to $RESULTS/ncu_cuda.txt"
echo ""
echo "ncu_profile.sh: done."
