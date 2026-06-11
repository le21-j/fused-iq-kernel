#!/usr/bin/env bash
# EXIT check for P3-PROMPT-5 (GPU-gated):
# report_card.md populated with REAL 4-way numbers; roofline.png exists and the
# report names the EXACT GPU read from nvidia-smi. PARKS (exit 2) off-CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if ! $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU_STEP: EXIT parked (no CUDA on this host). report_card/roofline remain PLACEHOLDER."
  exit 2
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ -n "$GPU" ] || { echo "FAIL: nvidia-smi returned no GPU name"; exit 1; }

grep -q "PLACEHOLDER" results/report_card.md && { echo "FAIL: report_card.md still PLACEHOLDER"; exit 1; }
grep -q "eager"  results/report_card.md || { echo "FAIL: no eager column"; exit 1; }
grep -q "triton" results/report_card.md || { echo "FAIL: no triton column"; exit 1; }
grep -q "cuda"   results/report_card.md || { echo "FAIL: no cuda column"; exit 1; }
grep -qF "$GPU"  results/report_card.md || { echo "FAIL: report_card.md does not name profiled GPU: $GPU"; exit 1; }
[ -s results/roofline.png ] || { echo "FAIL: results/roofline.png missing/empty"; exit 1; }

echo "EXIT CHECK P3-PROMPT-5: PASS (GPU: $GPU)"
