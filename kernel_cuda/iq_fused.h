#pragma once
// kernel_cuda/iq_fused.h
// Declaration of the CUDA host launcher for the fused IQ classification stage.
//
// Op semantics: complex conv1d (cross-correlation, C_in=1, C_out generic, K generic,
// stride 1, no padding) + complex bias + |z|^2 -> real float32 output.
// Input layout: interleaved complex64 [B, 1, L] (PyTorch stores complex64 as
// contiguous (re, im) float32 pairs; we reinterpret_cast to float2* for coalesced loads).

#include <ATen/Tensor.h>

namespace fused_iq_cuda {

// Host launcher called from binding.cpp's TORCH_LIBRARY_IMPL(CUDA) handler.
// Validates tensors, allocates output, launches the CUDA kernel, returns result.
//
// Args:
//   x  : complex64 contiguous [B, 1, L]
//   Wr : float32  contiguous [C_out, 1, K]
//   Wi : float32  contiguous [C_out, 1, K]
//   br : float32  contiguous [C_out]
//   bi : float32  contiguous [C_out]
// Returns:
//   float32 contiguous [B, C_out, L_out]  where L_out = L - K + 1
at::Tensor iq_fused_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& Wr,
    const at::Tensor& Wi,
    const at::Tensor& br,
    const at::Tensor& bi
);

} // namespace fused_iq_cuda
