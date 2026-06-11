"""
setup.py — builds the fused_iq_cuda CUDA extension.

Usage (on a Linux host with nvcc available):
    python setup.py build_ext --inplace
or via Makefile:
    make build

The extension name "fused_iq_cuda_ext" matches what the test harness
(tests/test_parity.py and tests/exit_check_p3.sh) will import.  After
building, `import fused_iq_cuda_ext` triggers TORCH_LIBRARY registration
for the "fused_iq_cuda" namespace, making
    torch.ops.fused_iq_cuda.fused_stage(...)
callable.

Architecture flags: deliberately omitted here.  torch.utils.cpp_extension
reads TORCH_CUDA_ARCH_LIST from the environment and selects appropriate SM
targets at build time.  Hard-coding sm_XX values would break portability
across GPU generations and violates the CLAUDE.md "no hard-coded arch flags"
constraint.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_iq_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="fused_iq_cuda_ext",
            sources=[
                "kernel_cuda/binding.cpp",
                "kernel_cuda/iq_fused.cu",
            ],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
