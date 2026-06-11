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
4. Be registered as `torch.ops.<ns>.fused_stage` via `torch.library.Library` with a `register_fake` meta-kernel (`<ns>` = `fused_iq` for Triton, `fused_iq_cuda` for the CUDA ext).

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

### Grid / block scheme

```
block = (256, 1, 1)          — 256 threads along the L_out dimension
grid  = (cdiv(L_out, 256), C_out, B)
          blockIdx.x  -> position tile  (l = blockIdx.x*256 + threadIdx.x)
          blockIdx.y  -> output channel co
          blockIdx.z  -> batch index b
```

Each thread owns exactly one output element `out[b, co, l]`.  The early-exit guard `if (l >= L_out) return;` handles the final partial tile.

### float2 reinterpret trick and coalesced loads

PyTorch stores `complex64` as contiguous `(re, im)` float32 pairs — the interleaved layout.  The kernel reinterprets the input data pointer:

```cpp
const float2* x2 = reinterpret_cast<const float2*>(x.data_ptr<c10::complex<float>>());
```

This is a safe zero-copy reinterpret: `c10::complex<float>` is guaranteed to be layout-compatible with `float2` (two consecutive floats, no padding).

Thread `l` reads tap `k` as `x2[b*L + l + k]`.  Adjacent threads `l`, `l+1`, ... access adjacent `float2` elements — every warp of 32 threads spans 32 × 8 = 256 contiguous bytes, which maps to a single 256-byte coalesced HBM transaction per tap.  A planar layout would require two independent streams (one for real, one for imaginary) with non-unit stride, halving effective bandwidth utilisation.

### __ldg for weights and bias

Weights `Wr[co*K+k]`, `Wi[co*K+k]` and bias `br[co]`, `bi[co]` are loaded via `__ldg`, which routes through the read-only L1 cache (backed by the texture unit on Volta+).  Since weights are shared identically across all B×L_out threads for a given `co`, they hit in cache after the first warp loads them — no explicit shared-memory staging needed.

`#pragma unroll` on the K loop lets nvcc maximise instruction-level parallelism: it issues multiple `__ldg` loads and FMA instructions in flight while waiting for memory.

### No shared-memory tiling

Shared-memory tiling is explicitly out of scope (CLAUDE.md).  The performance win for this op is fully coalesced HBM access (float2 loads) plus kernel fusion (conv + bias + |z|^2 in one pass, avoiding two intermediate complex tensors).  GEMM-style tile-and-reuse would conflate two distinct optimisation strategies and add complexity without a clear roofline justification for a K=15 filter.

### Namespace decision and collision rationale

The CUDA op is registered in the `fused_iq_cuda` namespace via `TORCH_LIBRARY(fused_iq_cuda, m)`.

The Triton path uses Python's `torch.library.Library("fused_iq", "DEF")` to own the `fused_iq` namespace.  PyTorch's dispatcher permits only one `DEF` owner per namespace.  A second `TORCH_LIBRARY("fused_iq", ...)` block in C++ would trigger a runtime collision error when both extensions are loaded.  Using `fused_iq_cuda` as a distinct namespace avoids the collision and makes it unambiguous in tests which backend is under evaluation:

```python
torch.ops.fused_iq.fused_stage(...)       # Triton path
torch.ops.fused_iq_cuda.fused_stage(...)  # CUDA path
```

### Meta impl as C++ register_fake

`TORCH_LIBRARY_IMPL(fused_iq_cuda, Meta, m)` provides a shape-propagation kernel that returns `at::empty({B, C_out, L_out}, dtype=float32, device=meta)` without executing any real computation.  This is the C++ analogue of Python's `torch.library.register_fake`: both tell the compiler (Dynamo, FX, symbolic shape analysis) how to compute the output shape from input shapes, enabling `torch.compile` to trace through the op without a GPU present.

## Registration API choice

**Choice:** direct `torch.library.Library` API (`lib = torch.library.Library("fused_iq", "DEF")`, `lib.define(...)`, `lib.impl(...)`, `torch.library.register_fake(...)`).

**Rejected alternative:** `@torch.library.custom_op` decorator.

The decorator is syntactic sugar that wraps the direct API but adds Python-level overhead on every dispatch path: it installs input validation callbacks, wires autograd, and injects functionalization hooks during registration.  These extras are useful when authoring a general-purpose op for a library; for a benchmark kernel they add noise to the dispatch measurement.

More practically, the decorator API hides the registration objects (`lib`, fake registrations) inside a closure, making it harder to introspect or un-register during testing.

**Dispatch overhead measurement:** the actual per-call overhead of `torch.ops.fused_iq.fused_stage` vs direct Python invocation of `iq_fused_triton()` will be measured in the Prompt 5 benchmark (micro-benchmark loop subtracting kernel latency from total call latency).  No fixed µs number is quoted here.
