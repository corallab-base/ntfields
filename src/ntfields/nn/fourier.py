import math

import torch
from torch import Tensor, nn


class RandomFourierFeatures(nn.Module):
    """[sin(2π q Bᵀ), cos(2π q Bᵀ)] with a fixed B ~ scale·N(0, 1) of shape (n, dim), saved as a buffer."""

    b: Tensor

    def __init__(self, dim: int, n: int, scale: float, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.register_buffer("b", scale * torch.randn(n, dim, generator=generator))

    @property
    def out_features(self) -> int:
        return 2 * self.b.shape[0]

    def forward(self, q: Tensor) -> Tensor:
        proj = 2 * math.pi * (q @ self.b.T)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)
