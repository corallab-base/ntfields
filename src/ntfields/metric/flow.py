from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class FlowField:
    """A drift field f(q) with an isotropic metric G = I, in the first two coordinates (the rest get no drift).

    - "constant": f = `vector`
    - "circular": counterclockwise about `center` with constant speed `magnitude` (0 within 1e-8 of it)
    - "saddle": f = `growth_rate` · (x, -y) about `center`, with |f| clipped to `magnitude`
    """

    kind: Literal["constant", "circular", "saddle"]
    vector: tuple[float, ...] = (0.0, 0.0)
    magnitude: float = 1.0
    growth_rate: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)

    def G(self, q: Tensor) -> Tensor:
        return torch.eye(q.shape[-1], dtype=q.dtype, device=q.device).expand(len(q), -1, -1)

    def drift(self, q: Tensor) -> Tensor:
        f = torch.zeros_like(q)
        if self.kind == "constant":
            return f + torch.tensor(self.vector, dtype=q.dtype, device=q.device)
        x = q[:, 0] - self.center[0]
        y = q[:, 1] - self.center[1]
        if self.kind == "circular":
            r = torch.hypot(x, y)
            safe = torch.where(r > 1e-8, r, torch.ones_like(r))
            f[:, 0] = torch.where(r > 1e-8, -self.magnitude * y / safe, 0.0)
            f[:, 1] = torch.where(r > 1e-8, self.magnitude * x / safe, 0.0)
            return f
        fx, fy = self.growth_rate * x, -self.growth_rate * y
        scale = (self.magnitude / torch.hypot(fx, fy).clamp(min=1e-30)).clamp(max=1.0)
        f[:, 0], f[:, 1] = fx * scale, fy * scale
        return f
