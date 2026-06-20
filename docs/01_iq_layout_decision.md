---
type: kernel-doc
title: "Chapter 1: IQ Layout Decision"
tags: [fused-iq, iq, layout, decision]
timestamp: 2026-06-17
---

# Chapter 1: IQ Layout Decision

**Punchline:** input is `torch.complex64` with shape `[B, 1, L]` — the interleaved
layout — meaning PyTorch's native memory order already gives us contiguous `(re, im)`
float pairs with no copy or view transformation needed.

## Example: what interleaved means in memory

For a single-sample sequence of length 4, the memory block looks like:

```
address:  0    4    8   12   16   20   24   28  (bytes)
value:   re0  im0  re1  im1  re2  im2  re3  im3
```

Each adjacent pair of floats is one complex sample. In CUDA C++, a thread at lane
`tid` loads the entire sample in **one 8-byte transaction**:

```c
float2 val = reinterpret_cast<float2*>(ptr)[tid];
float re = val.x, im = val.y;
```

That single load is coalesced: warp lanes 0–31 each hit 8 consecutive bytes in the
same 256-byte cache line. Planar would need two separate `float` loads from
non-adjacent base pointers, doubling the HBM traffic for inputs.

## The decision

**Chosen layout: Interleaved.**

`torch.complex64` *is* the interleaved layout — PyTorch stores it as contiguous
`(re, im)` float pairs by definition. No packing step is needed to go from PyTorch
tensor → kernel pointer.

| Property | Interleaved | Planar |
|---|---|---|
| Memory order | `re0 im0 re1 im1 ...` | `re0 re1 ... im0 im1 ...` |
| Coalescing (GPU warp) | One `float2` load per thread, 8-byte transaction | Two strided `float` streams, separate transactions |
| PyTorch interop | Zero-copy: `x.real`, `x.imag` are views into the same storage | Requires two separate real-valued tensors + a packing step |
| Kernel load idiom | `float2 val = reinterpret_cast<float2*>(ptr)[tid]` | `float re = rptr[tid]; float im = iptr[tid]` |

**Why coalescing matters here:** the fused stage is memory-bandwidth bound for the
input read at small batch sizes. A non-coalesced access pattern would show up
directly as lower achieved bandwidth in `ncu` profiles (Prompt 5).

## Zero-copy interop with the reference op

In `baseline/reference.py`, `fused_stage` extracts real and imaginary parts as:

```python
xr = x.real   # [B, 1, L]  — view into x storage, offset 0, stride 2
xi = x.imag   # [B, 1, L]  — view into x storage, offset 1, stride 2
```

These are PyTorch views (no allocation). The hand-written kernels in Prompts 2/3
receive the same `torch.complex64` tensor and interpret its underlying `float*`
as `float2*` — consistent with what `x.real` / `x.imag` expose.

## Constraint on Prompts 2 and 3

Both kernels **must**:

1. Accept `x: torch.complex64 [B, 1, L]` — no re-packing, no transpose.
2. Index input as `float2` (Triton: `tl.load` pairs; CUDA: `reinterpret_cast<float2*>`).
3. Be parity-tested against `IQClassifier.fused_stage()` at `atol=1e-4` (FP32).

Any kernel that accepts a planar `(xr, xi)` pair violates the spec and must be
rewritten.

## Gotchas

**Stride is 2 floats per complex sample.** `x.real` has stride `(L, L, 2)` in
bytes — not 1. If a kernel treats the storage as a flat `float*` array and
iterates with stride 1, it reads interleaved `re` and `im` values as if they were
consecutive real samples. This is the single most common layout bug.

**`view(torch.float32)` doubles the last dimension.** `x.view(torch.float32)` gives
shape `[B, 1, 2L]` with all floats interleaved — useful for raw pointer arithmetic
in custom ops, but easy to misconstrue as planar if you forget the factor of 2.

**`torch.complex64` vs `complex128`.** The kernel contract is `complex64`
(`float32` components). Passing `complex128` (`float64`) will silently produce
wrong results if the kernel casts unconditionally to `float2`.

## Related

- `docs/design.md` — authoritative one-paragraph rationale checked into the repo.
- `baseline/reference.py` — `fused_stage` shows the zero-copy `x.real` / `x.imag` pattern.
- Chapter 2 (`docs/02_reference_op_and_baselines.md`) — fused stage boundary and benchmarks.
- Prompts 2/3 (`kernel_triton/`, `kernel_cuda/`) — must consume this layout.
