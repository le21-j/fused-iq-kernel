#!/usr/bin/env bash
# EXIT check for P3-PROMPT-2 (GPU-gated):
# Triton op imports, runs on CUDA, callable as torch.ops.fused_iq.fused_stage.
# On a non-CUDA host this check PARKS (exit 2) — it must only PASS on a CUDA box.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if ! $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU_STEP: ready for remote build. EXIT parked (no CUDA on this host)."
  exit 2
fi

$PY - <<'EOF'
import torch
from kernel_triton import register  # registers torch.ops.fused_iq.fused_stage
from baseline.reference import IQClassifier

m = IQClassifier().cuda()
x = torch.randn(2, 1, 256, dtype=torch.complex64, device="cuda")
out = torch.ops.fused_iq.fused_stage(x, m.Wr, m.Wi, m.br, m.bi)
ref = m.fused_stage(x)
assert out.shape == ref.shape == (2, 16, 242), out.shape
torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)
print("EXIT CHECK P3-PROMPT-2: PASS")
EOF
