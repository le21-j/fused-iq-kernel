"""
build_compiled: returns a torch.compile'd IQClassifier.

On CUDA, inductor lowers to Triton-generated kernels — the real bar to beat
with the hand-written Triton / CUDA extensions in Prompts 2/3.
On macOS arm64 there is NO Triton backend; inductor falls back automatically,
so we switch to backend="aot_eager" to avoid a noisy warning and get a clean
graph capture. LOCAL TIMINGS ARE PROVISIONAL — do not quote them as GPU numbers.
"""

import torch

from baseline.reference import IQClassifier


def build_compiled(seed: int = 0) -> torch.nn.Module:
    model = IQClassifier(seed)
    # inductor on CUDA; aot_eager on macOS/CPU (no Triton backend — timings PROVISIONAL)
    backend = "inductor" if torch.cuda.is_available() else "aot_eager"
    return torch.compile(model, backend=backend)
