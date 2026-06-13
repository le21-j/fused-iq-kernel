# kernel_w4a16 package
# Import register to activate torch.ops.w4a16.fused_gemm (lazy Triton import).
from kernel_w4a16 import register  # noqa: F401
