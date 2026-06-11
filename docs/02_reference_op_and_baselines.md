# Chapter 2: Reference Op and Baselines

**Punchline:** `IQClassifier` defines the correctness oracle for all later kernels;
`fused_stage()` is the parity boundary; the `torch.compile` baseline uses
`aot_eager` on macOS (no Triton backend), so all timings from Prompt 1 are
**PROVISIONAL CPU numbers** — not predictive of GPU performance.

## The op pipeline

```
x: complex64 [B, 1, L]
  → complex conv1d  (C_in=1, C_out=16, K=15, stride 1, no pad)
  → complex bias add  (br, bi each [16])
  → squared-magnitude activation: yr² + yi²  →  real [B, 16, L_out]
                                                   where L_out = L − 14
  → global mean pool over time dim  →  [B, 16]
  → Linear(16 → 2)                  →  logits [B, 2]
```

The first three steps form `fused_stage()`. The last two (`mean pool + Linear`)
are left as plain torch ops and are **not** kernel targets.

## Complex conv via real decomposition

PyTorch's native complex `conv1d` support is patchy and not compile-friendly.
The reference op decomposes it into four real `conv1d` calls:

```python
yr = F.conv1d(xr, Wr) - F.conv1d(xi, Wi)   # real part of output
yi = F.conv1d(xr, Wi) + F.conv1d(xi, Wr)   # imaginary part of output
```

This is the **Karatsuba-like** real decomposition: `(xr + j·xi)(Wr + j·Wi)`.
It maps cleanly to four real memory accesses in both Triton and CUDA and avoids
any complex-number special casing in the kernels.

## `fused_stage()` — the parity boundary

`baseline/reference.py::IQClassifier.fused_stage` is the exact function that
hand-written kernels in Prompts 2 and 3 must match:

```python
def fused_stage(self, x: torch.Tensor) -> torch.Tensor:
    xr, xi = x.real, x.imag                         # zero-copy views
    yr = F.conv1d(xr, self.Wr) - F.conv1d(xi, self.Wi)
    yi = F.conv1d(xr, self.Wi) + F.conv1d(xi, self.Wr)
    yr = yr + self.br.view(1, -1, 1)                 # complex bias
    yi = yi + self.bi.view(1, -1, 1)
    return yr * yr + yi * yi                         # |z|^2, real output
```

**Input:** `complex64 [B, 1, L]` — interleaved layout (Chapter 1).
**Output:** `float32 [B, 16, L_out]` — real squared magnitudes, where `L_out = L − 14`.

Parity test tolerance: `atol=1e-4` (FP32 accumulation errors at this kernel size).

## Why fuse conv + bias + |z|²?

Without fusion, the naive op writes two intermediate complex tensors
`[B, 16, L_out]` (each complex64 = 8 bytes/element) to HBM after conv, reads them
back for bias, writes again, reads again for the magnitude. Fusion keeps those
intermediates in registers and writes only the final real output.

At `B=128, L=16384`: `L_out = 16370`, complex tensor size ≈ 128·16·16370·8 ≈ 270 MB
round-trip per intermediate — 2× intermediates = ~540 MB saved per forward pass.

## `torch.compile` baseline

`baseline/compiled.py::build_compiled` wraps `IQClassifier` in `torch.compile`:

```python
backend = "inductor" if torch.cuda.is_available() else "aot_eager"
return torch.compile(model, backend=backend)
```

- **On CUDA (remote host):** `inductor` lowers to Triton-generated kernels — the
  real bar that the hand-written Triton/CUDA kernels in Prompts 2/3 must beat.
- **On macOS arm64 (this machine):** no Triton backend; `inductor` is unavailable.
  `aot_eager` is used instead, which captures the graph but does not generate
  optimized code. **Timings are PROVISIONAL.**

## Benchmark methodology

`bench/benchmark.py` sweeps `(B, L)` ∈ `{1,8,32,128} × {1024,4096,16384}`:

- **Warmup:** 10 forward passes (discarded).
- **Timed:** 50 iterations for `L ∈ {1024, 4096}`; 20 for `L=16384` (CPU budget).
- **Timing:** `time.perf_counter()` wall-clock; `torch.cuda.synchronize()` called
  before start and after end (no-op on CPU).
- **Metric:** mean latency in milliseconds over the timed window.

> [!warning]
> `cuda.synchronize()` is a no-op when there is no CUDA device. Wall-clock timing
> on CPU includes Python overhead (loop, dispatch, autograd bookkeeping) that
> `torch.compile` should eliminate — but without Triton lowering, `aot_eager`
> provides graph capture with minimal kernel-level optimization.

## PROVISIONAL benchmark results (macOS arm64 CPU)

These numbers were produced on the local machine with no CUDA device.
**Do not interpret as GPU performance.** Real GPU numbers are populated in Prompt 5.

```
Device: cpu
PROVISIONAL: macOS/CPU timings — torch.compile has no Triton backend here
(aot_eager); remote GPU results will differ.

 batch   seqlen    eager_ms   compiled_ms   speedup
-------------------------------------------------------
     1     1024       0.147         0.204      0.72x
     1     4096       0.306         0.424      0.72x
     1    16384       0.676         0.716      0.94x
     8     1024       0.486         0.588      0.83x
     8     4096       1.132         1.148      0.99x
     8    16384       6.214         6.343      0.98x
    32     1024     147.721       148.402      1.00x
    32     4096     631.284       605.411      1.04x
    32    16384    2482.891      2414.229      1.03x
   128     1024     387.499       386.554      1.00x
   128     4096    1551.526      1534.133      1.01x
   128    16384    6384.113      6061.923      1.05x
```

**Reading the table:** `aot_eager` compile overhead dominates at small batch/seqlen
(0.72× at `B=1`). The overhead amortizes as work increases; at large batch sizes the
two paths are essentially tied (1.00–1.05×). Neither path benefits from kernel fusion
on CPU — the HBM savings are not realized until a CUDA device is present.

## Gotchas

**`aot_eager` ≠ `inductor`.** The CUDA inductor path fuses ops, reduces kernel
launches, and lowers to Triton. `aot_eager` does none of this. Seeing `compiled_ms ≈
eager_ms` on macOS is expected — it is not evidence that the hand-written kernels
won't beat compile on GPU.

**`time.perf_counter()` without sync is wrong on CUDA.** The benchmark calls
`sync()` correctly both before the timed window starts and after it ends. If you
copy-paste the timing loop, keep both sync calls or you'll measure launch latency
rather than execution time.

**`L_out = L − 14`, not `L`.** The conv has `K=15` with no padding; output length
shrinks by `K−1 = 14`. Parity tests that compare shapes before and after the fused
stage without accounting for this will fail with a shape mismatch, not a numerical
error.

## Related

- `baseline/reference.py` — `IQClassifier`, `fused_stage`, exact parity oracle.
- `baseline/compiled.py` — `build_compiled`, backend selection.
- `bench/benchmark.py` — sweep + timing harness.
- Chapter 1 (`docs/01_iq_layout_decision.md`) — interleaved layout constraint.
- `docs/design.md` §(b) — op definition and fusion boundary.
- Prompt 4 — parity test suite (`atol=1e-4`) against this exact oracle.
- Prompt 5 — 4-way benchmark with real GPU numbers; roofline from `nvidia-smi`.
