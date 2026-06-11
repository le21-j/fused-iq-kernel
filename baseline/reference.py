"""
IQClassifier: correctness oracle for the fused IQ classification layer.

IQ layout: INTERLEAVED — input is torch.complex64 [B, 1, L].
PyTorch stores complex64 as contiguous (re, im) float pairs in memory.
Later hand-written kernels exploit this with float2 coalesced loads
(each thread loads one complex sample in a single 8-byte transaction).
Planar layout would split real/imag into separate buffers, requiring two
strided memory streams — worse coalescing and no zero-copy interop with
torch complex tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQClassifier(nn.Module):
    # Fused stage: complex conv1d (C_in=1, C_out=16, K=15, stride 1, no pad)
    # -> complex bias add -> squared-magnitude |z|^2 -> [B, 16, L_out]
    # Pool + linear stage: global mean pool -> Linear(16, 2) logits.

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        # Real decomposition weights: yr = conv(xr,Wr) - conv(xi,Wi); yi = conv(xr,Wi) + conv(xi,Wr)
        self.Wr = nn.Parameter(torch.empty(16, 1, 15).normal_(generator=g))
        self.Wi = nn.Parameter(torch.empty(16, 1, 15).normal_(generator=g))
        # Complex bias
        self.br = nn.Parameter(torch.zeros(16))
        self.bi = nn.Parameter(torch.zeros(16))
        self.linear = nn.Linear(16, 2)
        # Re-init linear with the same generator for determinism
        nn.init.normal_(self.linear.weight, generator=g)
        nn.init.zeros_(self.linear.bias)

    def fused_stage(self, x: torch.Tensor) -> torch.Tensor:
        # x: complex64 [B, 1, L] -> real [B, 16, L_out]  (fused conv+bias+|z|^2)
        xr = x.real  # [B, 1, L]
        xi = x.imag
        yr = F.conv1d(xr, self.Wr) - F.conv1d(xi, self.Wi)  # [B, 16, L_out]
        yi = F.conv1d(xr, self.Wi) + F.conv1d(xi, self.Wr)
        # Complex bias add
        yr = yr + self.br.view(1, -1, 1)
        yi = yi + self.bi.view(1, -1, 1)
        # Squared-magnitude activation
        return yr * yr + yi * yi  # real [B, 16, L_out]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: complex64 [B, 1, L] -> logits [B, 2]
        act = self.fused_stage(x)              # [B, 16, L_out]
        pooled = act.mean(dim=-1)              # [B, 16]
        return self.linear(pooled)             # [B, 2]
