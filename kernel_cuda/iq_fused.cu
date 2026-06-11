// kernel_cuda/iq_fused.cu
// CUDA kernel: fused complex conv1d + complex bias + |z|^2.
//
// Thread/block mapping:
//   block = (256, 1, 1) — 256 threads along L_out.
//   grid  = (cdiv(L_out, 256), C_out, B)
//             blockIdx.x -> position tile
//             blockIdx.y -> output channel co
//             blockIdx.z -> batch b
//
// Each thread owns one output element: out[b, co, l].
//
// Coalesced load strategy:
//   Input is reinterpreted as const float2* (complex64 IS interleaved (re,im) in
//   PyTorch memory).  Thread l reads x2[b*L + l + k] for k in 0..K-1.
//   Adjacent threads (l, l+1, ...) access adjacent float2 elements ->
//   fully coalesced 8-byte transactions.
//
// Weights and bias are loaded via __ldg (read-only L1/texture cache path) with
// #pragma unroll on the k loop to maximise ILP.
// No shared-memory tiling: the win is coalescing + fusion, not GEMM-style reuse
// (explicitly out of scope per CLAUDE.md).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "iq_fused.h"

namespace fused_iq_cuda {

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------

__global__ void iq_fused_kernel(
    const float2* __restrict__ x2,  // [B, L] as float2 (C_in=1 collapsed)
    const float*  __restrict__ Wr,  // [C_out, K]  — squeezed from [C_out,1,K]
    const float*  __restrict__ Wi,  // [C_out, K]
    const float*  __restrict__ br,  // [C_out]
    const float*  __restrict__ bi,  // [C_out]
    float*        __restrict__ out, // [B, C_out, L_out]
    int B,
    int C_out,
    int L,
    int L_out,
    int K
) {
    // Grid: blockIdx.z=b, blockIdx.y=co, blockIdx.x=l-tile
    int b  = blockIdx.z;
    int co = blockIdx.y;
    int l  = blockIdx.x * blockDim.x + threadIdx.x;

    if (l >= L_out) return;

    // Input base for batch b (C_in=1, stride is L complex elements per batch)
    const float2* x_b = x2 + (long long)b * L;

    // Weight base for output channel co: Wr/Wi have shape [C_out, K] in memory
    int w_base = co * K;

    // Load bias scalars for this output channel via __ldg
    float vbr = __ldg(br + co);
    float vbi = __ldg(bi + co);

    float acc_r = 0.0f;
    float acc_i = 0.0f;

    // Sequential k-accumulation in fp32 — matches conv1d summation order exactly.
    // #pragma unroll hints to nvcc to unroll the loop for ILP; K is runtime but
    // the compiler unrolls up to a threshold (default 4, raised by ptxas settings).
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        // Coalesced 8-byte load: adjacent threads read adjacent float2 elements
        float2 xv = x_b[l + k];
        float xr = xv.x;
        float xi = xv.y;

        // Weights via __ldg -> read-only cache (avoids polluting L1 data cache)
        float wr = __ldg(Wr + w_base + k);
        float wi = __ldg(Wi + w_base + k);

        // Complex multiply-accumulate: (xr+j*xi)*(wr+j*wi) contributes to cross-correlation tap k
        acc_r += xr * wr - xi * wi;
        acc_i += xr * wi + xi * wr;
    }

    // Complex bias
    float yr = acc_r + vbr;
    float yi = acc_i + vbi;

    // Squared magnitude |z|^2 -> real output
    float mag2 = yr * yr + yi * yi;

    // Output index: out[b, co, l]
    long long out_idx = ((long long)b * C_out + co) * L_out + l;
    out[out_idx] = mag2;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

at::Tensor iq_fused_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& Wr,
    const at::Tensor& Wi,
    const at::Tensor& br,
    const at::Tensor& bi
) {
    // --- dtype checks -------------------------------------------------------
    TORCH_CHECK(x.dtype()  == at::kComplexFloat,
                "iq_fused: x must be complex64, got ", x.dtype());
    TORCH_CHECK(Wr.dtype() == at::kFloat,
                "iq_fused: Wr must be float32, got ", Wr.dtype());
    TORCH_CHECK(Wi.dtype() == at::kFloat,
                "iq_fused: Wi must be float32, got ", Wi.dtype());
    TORCH_CHECK(br.dtype() == at::kFloat,
                "iq_fused: br must be float32, got ", br.dtype());
    TORCH_CHECK(bi.dtype() == at::kFloat,
                "iq_fused: bi must be float32, got ", bi.dtype());

    // --- contiguity ---------------------------------------------------------
    TORCH_CHECK(x.is_contiguous(),  "iq_fused: x must be contiguous");
    TORCH_CHECK(Wr.is_contiguous(), "iq_fused: Wr must be contiguous");
    TORCH_CHECK(Wi.is_contiguous(), "iq_fused: Wi must be contiguous");
    TORCH_CHECK(br.is_contiguous(), "iq_fused: br must be contiguous");
    TORCH_CHECK(bi.is_contiguous(), "iq_fused: bi must be contiguous");

    // --- shape checks -------------------------------------------------------
    TORCH_CHECK(x.dim()  == 3 && x.size(1) == 1,
                "iq_fused: x must be [B, 1, L], got ", x.sizes());
    TORCH_CHECK(Wr.dim() == 3 && Wi.dim() == 3,
                "iq_fused: Wr/Wi must be [C_out, 1, K]");
    TORCH_CHECK(Wr.sizes() == Wi.sizes(),
                "iq_fused: Wr and Wi shape mismatch");
    TORCH_CHECK(br.dim() == 1 && bi.dim() == 1 && br.sizes() == bi.sizes(),
                "iq_fused: br/bi must be 1-D same shape");
    TORCH_CHECK(Wr.size(0) == br.size(0),
                "iq_fused: C_out mismatch between Wr and br");

    int B     = x.size(0);
    int L     = x.size(2);
    int C_out = Wr.size(0);
    int K     = Wr.size(2);
    int L_out = L - K + 1;

    TORCH_CHECK(L_out > 0,
                "iq_fused: L=", L, " too short for K=", K, " (L_out=", L_out, ")");

    // --- device / CUDA checks -----------------------------------------------
    TORCH_CHECK(x.is_cuda(),  "iq_fused: x must be a CUDA tensor");
    TORCH_CHECK(Wr.is_cuda(), "iq_fused: Wr must be a CUDA tensor");
    TORCH_CHECK(Wi.is_cuda(), "iq_fused: Wi must be a CUDA tensor");
    TORCH_CHECK(br.is_cuda(), "iq_fused: br must be a CUDA tensor");
    TORCH_CHECK(bi.is_cuda(), "iq_fused: bi must be a CUDA tensor");

    // --- allocate output ----------------------------------------------------
    at::Tensor out = at::empty({B, C_out, L_out},
                               x.options().dtype(at::kFloat));

    // --- reinterpret input as float2* ---------------------------------------
    // complex64 IS interleaved (re,im) pairs; reinterpret_cast is safe and zero-copy.
    const float2* x2_ptr =
        reinterpret_cast<const float2*>(x.data_ptr<c10::complex<float>>());

    // Squeeze Wr/Wi from [C_out, 1, K] to [C_out, K] — data_ptr is same
    const float* Wr_ptr = Wr.data_ptr<float>();
    const float* Wi_ptr = Wi.data_ptr<float>();
    const float* br_ptr = br.data_ptr<float>();
    const float* bi_ptr = bi.data_ptr<float>();
    float*       out_ptr = out.data_ptr<float>();

    // --- grid / block -------------------------------------------------------
    const int THREADS = 256;
    dim3 block(THREADS, 1, 1);
    dim3 grid(
        (L_out + THREADS - 1) / THREADS,  // tiles over L_out
        C_out,                              // blockIdx.y -> output channel
        B                                   // blockIdx.z -> batch
    );

    // Launch on the current stream so it interoperates correctly with
    // PyTorch's CUDA stream management (essential for correctness under
    // torch.compile and multi-stream scenarios).
    iq_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x2_ptr, Wr_ptr, Wi_ptr, br_ptr, bi_ptr, out_ptr,
        B, C_out, L, L_out, K
    );

    // Check for async kernel errors immediately after launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace fused_iq_cuda
