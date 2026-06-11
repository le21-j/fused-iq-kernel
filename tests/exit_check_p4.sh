#!/usr/bin/env bash
# EXIT check for P3-PROMPT-4 (GPU-gated):
# ALL parity (atol=1e-4 FP32) and shape tests pass for BOTH kernels on CUDA.
# On a non-CUDA host this check PARKS (exit 2) — tests require real kernels.
# No benchmarking before parity holds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if ! $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU_STEP: EXIT parked (no CUDA on this host). Tests written, not runnable here."
  # collection must still succeed locally — catches syntax/import rot early
  $PY -m pytest tests/test_parity.py tests/test_shapes.py --collect-only -q
  exit 2
fi

$PY -m pytest tests/test_parity.py tests/test_shapes.py -v
echo "EXIT CHECK P3-PROMPT-4: PASS"
