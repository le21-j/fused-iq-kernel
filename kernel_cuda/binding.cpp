// kernel_cuda/binding.cpp
// Registers the fused_stage op in the "fused_iq_cuda" namespace using
// TORCH_LIBRARY / TORCH_LIBRARY_IMPL — the direct torch.library C++ API
// (mirrors the Python-side torch.library.Library("fused_iq","DEF") pattern
// used in kernel_triton/register.py).
//
// Namespace separation rationale:
//   The Triton path registers in the "fused_iq" namespace via Python's
//   torch.library.Library("fused_iq", "DEF").  A second TORCH_LIBRARY("fused_iq",...)
//   block in C++ would collide with that Python registration (the dispatcher allows
//   only ONE DEF owner per namespace).  Using "fused_iq_cuda" as a separate namespace
//   avoids the collision entirely and makes dispatch unambiguous in tests.
//
// Op schema:
//   fused_iq_cuda::fused_stage(Tensor x, Tensor Wr, Tensor Wi, Tensor br, Tensor bi) -> Tensor
//
// Dispatch keys registered:
//   CUDA  — calls iq_fused_forward_cuda() in iq_fused.cu
//   Meta  — returns an empty float32 tensor with the correct output shape [B, C_out, L_out]
//           This is the C++ analogue of torch.library.register_fake: it provides shape
//           propagation for torch.compile / FX tracing without executing any real computation.

#include <torch/library.h>
#include <ATen/core/TensorBody.h>
#include "iq_fused.h"

// ---------------------------------------------------------------------------
// Schema definition (equivalent to lib.define(...) in Python)
// ---------------------------------------------------------------------------

TORCH_LIBRARY(fused_iq_cuda, m) {
    m.def("fused_stage(Tensor x, Tensor Wr, Tensor Wi, Tensor br, Tensor bi) -> Tensor");
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(fused_iq_cuda, CUDA, m) {
    m.impl("fused_stage", &fused_iq_cuda::iq_fused_forward_cuda);
}

// ---------------------------------------------------------------------------
// Meta implementation (shape propagation — C++ analogue of register_fake)
//
// The Meta dispatch key is invoked by:
//   - torch.compile / Dynamo (to trace through the op)
//   - torch.fx symbolic shape analysis
//   - torch._dynamo.export
//
// It must return a tensor of the correct dtype, device=meta, and shape without
// executing any real computation.  This mirrors what register_fake does on the
// Python side: provide a fake-tensor kernel that the compiler can symbolically
// evaluate to propagate shapes through the graph.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(fused_iq_cuda, Meta, m) {
    m.impl("fused_stage",
        [](const at::Tensor& x,
           const at::Tensor& Wr,
           const at::Tensor& Wi,
           const at::Tensor& br,
           const at::Tensor& bi) -> at::Tensor {
            // Derive output shape from inputs — same arithmetic as the host launcher
            int64_t B     = x.size(0);
            int64_t C_out = Wr.size(0);
            int64_t L     = x.size(2);
            int64_t K     = Wr.size(2);
            int64_t L_out = L - K + 1;

            // Return empty meta tensor: correct shape + float32, no storage allocated
            return at::empty({B, C_out, L_out},
                             x.options().dtype(at::kFloat).device(at::kMeta));
        });
}
