# Design decisions

## (a) IQ layout: Interleaved

**Choice:** input tensor is `torch.complex64` with shape `[B, 1, L]`.

PyTorch stores `complex64` as contiguous `(re, im)` float pairs in memory — this *is* the interleaved layout. There is no copy or view transformation needed; the tensor's data pointer already holds `[re0, im0, re1, im1, ...]`.

**Rationale vs planar layout:**

| | Interleaved | Planar |
|---|---|---|
| Memory layout | `re0 im0 re1 im1 ...` | `re0 re1 ... im0 im1 ...` |
| Coalescing (GPU) | Adjacent threads load adjacent `float2` pairs — one 8-byte transaction per thread | Real and imag buffers are separate; each requires its own strided stream |
| PyTorch interop | Zero-copy: `x.real` / `x.imag` return views into the same storage | Would require two separate real-valued tensors and a packing step |
| Kernel load pattern | `float2 val = reinterpret_cast<float2*>(ptr)[tid]` | Two separate `float` loads from different base pointers |

Prompts 2 and 3 kernels MUST consume this layout.

## (b) Op definition and fusion boundary

**Full op pipeline:**

```
x: complex64 [B, 1, L]
  -> complex conv1d (C_in=1, C_out=16, K=15, stride 1, no padding)
  -> complex bias add (br, bi each shape [16])
  -> squared-magnitude activation: act = yr² + yi²   →  real [B, 16, L_out]  where L_out = L - 14
  -> global mean pool over time dim                  →  [B, 16]
  -> Linear(16 → 2)                                  →  logits [B, 2]
```

**Complex conv via real decomposition** (portable; torch.compile-friendly):

```python
yr = conv1d(xr, Wr) - conv1d(xi, Wi)
yi = conv1d(xr, Wi) + conv1d(xi, Wr)
```

This avoids relying on PyTorch's patchy native complex conv support and maps cleanly to four real conv1d calls in both Triton and CUDA.

**Fusion boundary:**

- **Fused stage** (hand-written kernels in Prompts 2/3): `conv + bias + |z|²` → real `[B, 16, L_out]`
- **Unfused stage** (plain torch ops, not kernel targets): `mean pool + Linear`

The fusion avoids two intermediate `[B, 16, L_out]` complex tensors being written to and read from HBM; instead a single real output is written.

## (c) Kernel contract (Prompts 2 and 3)

Both the Triton kernel (`kernel_triton/`) and CUDA kernel (`kernel_cuda/`) MUST:

1. Accept `x: torch.complex64 [B, 1, L]` — interleaved layout, no re-packing.
2. Implement `conv + bias + |z|²` and return `real [B, 16, L_out]`.
3. Be parity-tested against `IQClassifier.fused_stage()` with `atol=1e-4` (FP32).
4. Be registered as `torch.ops.<ns>.iq_fused_stage` via `torch.library.Library` with a `register_fake` meta-kernel.

## Triton kernel tiling

*(Prompt 2)*

## CUDA thread/block mapping

*(Prompt 3)*

## Registration API choice

*(Prompt 2/3 — direct `torch.library.Library` API only; `@torch.library.custom_op` decorator API is out of scope per CLAUDE.md)*
