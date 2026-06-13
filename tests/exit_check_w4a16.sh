#!/usr/bin/env bash
# EXIT check for W4A16 fused dequant+GEMM (GPU-gated):
# Parity tests pass for the fused Triton op vs the reference at atol=1e-3 fp16.
# On a non-CUDA host this check PARKS (exit 2). Never fabricate a run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if ! $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU_STEP: EXIT parked (no CUDA on this host). Tests written, not runnable here."
  # Collection must still succeed locally — catches syntax/import rot early
  $PY -m pytest tests/test_w4a16_parity.py --collect-only -q
  exit 2
fi

$PY -m pytest tests/test_w4a16_parity.py -v
echo "EXIT CHECK W4A16: PASS"
