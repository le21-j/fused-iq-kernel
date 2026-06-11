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

### Grid / block scheme

Grid: `(B * C_out, cdiv(L_out, BLOCK_L))`.

- `pid0` encodes both the batch index and the output channel: `b = pid0 // C_out`, `co = pid0 % C_out`.
- `pid1` selects a tile of `BLOCK_L` contiguous output positions: `l0 = pid1 * BLOCK_L`.
- Each program computes `out[b, co, l0 : l0 + BLOCK_L]` — a contiguous slice of the real output tensor.

### Interleaved float2 load pattern and coalescing

The kernel receives `x` pre-reinterpreted as a `float32` view of shape `[B, 1, 2*L]` (via `torch.view_as_real(...).reshape(B, 1, 2*L)`).  For a given tap `k`, the real and imaginary parts of output position `l` are at flat float offsets `2*(l+k)` and `2*(l+k)+1` respectively.

With `l_offsets = l0 + arange(0, BLOCK_L)`, adjacent lanes load:

```
re offsets: 2*(l0+k), 2*(l0+k+1), 2*(l0+k+2), ...
im offsets: 2*(l0+k)+1, 2*(l0+k+1)+1, ...
```

Both address sequences are contiguous strides of 2 float32s = 8 bytes apart per lane.  A warp of 32 threads spans `32 * 8 = 256` bytes — a single 256-byte coalesced transaction.  The real and imaginary loads are issued as two separate 256-byte transactions per tap (same cache line coverage), which is equivalent to one `float2` vectorised load per element in CUDA terms.  This is the interleaved layout's advantage: a planar layout would require two independent non-contiguous streams with twice the cache pressure.

### Autotune space

`BLOCK_L` ∈ {64, 128, 256, 512} × `num_warps` ∈ {2, 4, 8}, keyed on `L_out`.  Triton's autotuner selects the best config on first invocation for each distinct `L_out` and caches it; no manual tuning is required.

### Sequential-k accumulation for parity

The inner loop over `k = 0 .. K-1` is written as a sequential Python `for k in range(K)` with `tl.constexpr` bounds so Triton unrolls it.  Accumulation is left-to-right:

```
acc_r += xr*wr - xi*wi
acc_i += xr*wi + xi*wr
```

This exactly mirrors `torch.nn.functional.conv1d`'s accumulation order, which is also sequential over the filter dimension in fp32.  A tree-reduction would produce a different floating-point summation order and could push max absolute error above the `atol=1e-4` parity threshold.  Sequential accumulation is therefore a deliberate parity constraint, not a performance choice.

## CUDA thread/block mapping

*(Prompt 3)*

## Registration API choice

**Choice:** direct `torch.library.Library` API (`lib = torch.library.Library("fused_iq", "DEF")`, `lib.define(...)`, `lib.impl(...)`, `torch.library.register_fake(...)`).

**Rejected alternative:** `@torch.library.custom_op` decorator.

The decorator is syntactic sugar that wraps the direct API but adds Python-level overhead on every dispatch path: it installs input validation callbacks, wires autograd, and injects functionalization hooks during registration.  These extras are useful when authoring a general-purpose op for a library; for a benchmark kernel they add noise to the dispatch measurement.

More practically, the decorator API hides the registration objects (`lib`, fake registrations) inside a closure, making it harder to introspect or un-register during testing.

**Dispatch overhead measurement:** the actual per-call overhead of `torch.ops.fused_iq.fused_stage` vs direct Python invocation of `iq_fused_triton()` will be measured in the Prompt 5 benchmark (micro-benchmark loop subtracting kernel latency from total call latency).  No fixed µs number is quoted here.
