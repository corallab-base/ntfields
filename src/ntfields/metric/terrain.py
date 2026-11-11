import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass
class Terrain:
    """A point mass on a height field h over [-1, 1]²: G(q) = I + ∇h∇hᵀ (the pullback metric) and a downhill drift
    f(q) = c · (-∇h/‖∇h‖) · ‖∇h‖/(‖∇h‖ + `taper`).

    The global scale c makes max_q fᵀGf = `max_norm_g_sq` (without the taper), found on a 200 x 200 grid over
    [-1, 1]² as in the reference. Heights:
    - "single_hill": h = height · exp(-‖p - ½‖² / (2·spread²)), p = (q + 1)/2
    - "smooth_ridge": h = height · sin(π(x + 1)/2)
    """

    kind: Literal["single_hill", "smooth_ridge"]
    height: float = 0.5
    spread: float = 0.3
    max_norm_g_sq: float = 0.9
    taper: float = 0.05
    _scale: float | None = field(default=None, init=False, repr=False)

    def height_grad(self, q: Tensor) -> Tensor:
        """∇h (B, 2)."""
        if self.kind == "single_hill":
            p = (q + 1) / 2 - 0.5
            h = self.height * torch.exp(-(p**2).sum(-1, keepdim=True) / (2 * self.spread**2))
            return -h * p / self.spread**2 * 0.5
        zx = self.height * torch.cos(math.pi * (q[:, :1] + 1) / 2) * (math.pi / 2)
        return torch.cat([zx, torch.zeros_like(zx)], dim=-1)

    def G(self, q: Tensor) -> Tensor:
        g = self.height_grad(q)
        return torch.eye(2, dtype=q.dtype, device=q.device) + g[:, :, None] * g[:, None, :]

    def drift(self, q: Tensor) -> Tensor:
        g = self.height_grad(q)
        norm = g.norm(dim=-1, keepdim=True)
        unit = -g / norm.clamp(min=1e-8)
        f = unit * self.scale() * norm / (norm + self.taper)
        return torch.where(norm < 1e-8, torch.zeros_like(f), f)

    def scale(self) -> float:
        if self._scale is None:
            axis = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
            q = torch.cartesian_prod(axis, axis)
            g = self.height_grad(q)
            norm = g.norm(dim=-1, keepdim=True)
            u = (-g / norm.clamp(min=1e-30))[norm[:, 0] >= 1e-8]
            g = g[norm[:, 0] >= 1e-8]
            raw = (u**2).sum(-1) + (u * g).sum(-1) ** 2  # uᵀ(I + ggᵀ)u
            self._scale = math.sqrt(self.max_norm_g_sq / max(float(raw.max()) if len(raw) else 0.0, 1e-8))
        return self._scale
