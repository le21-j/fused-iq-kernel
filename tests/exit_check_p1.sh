#!/usr/bin/env bash
# EXIT check for P3-PROMPT-1:
# benchmark.py must print a latency table for eager vs compiled across a sweep
# of (batch, seqlen) sizes. On macOS the numbers are PROVISIONAL (aot_eager).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=$(.venv/bin/python bench/benchmark.py 2>&1)
echo "$OUT"

echo "$OUT" | grep -q "eager"            || { echo "FAIL: no eager column"; exit 1; }
echo "$OUT" | grep -q "compiled"         || { echo "FAIL: no compiled column"; exit 1; }
echo "$OUT" | grep -q "PROVISIONAL"      || { echo "FAIL: missing PROVISIONAL label on non-CUDA host"; exit 1; }
# at least 4 sweep rows: lines starting with a batch size number
NROWS=$(echo "$OUT" | grep -cE '^\s*[0-9]+\s+[0-9]+' || true)
[ "$NROWS" -ge 4 ] || { echo "FAIL: expected >=4 sweep rows, got $NROWS"; exit 1; }

echo "EXIT CHECK P3-PROMPT-1: PASS"
