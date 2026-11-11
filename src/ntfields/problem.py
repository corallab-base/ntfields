from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.autodiff import grad
from ntfields.base import Environment, Metric, SpeedModel


@dataclass
class Problem:
    env: Environment
    speed: SpeedModel
    metric: Metric | None = None  # anisotropy G(q) and drift f(q); None: isotropic, no drift

    @property
    def dim(self) -> int:
        return self.env.dim

    def speed_at(self, q: Tensor) -> Tensor:
        return self.speed(self.env.distance(q))

    def speed_and_normal(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """S*(q) and the unit clearance gradient ∇d/‖∇d‖ (0 where undefined), both detached."""
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            d = self.env.distance(q)
            s = self.speed(d).detach()
            if not d.requires_grad:  # e.g. no obstacles
                return s, torch.zeros_like(q)
            g = grad(d, q, create_graph=False)
        return s, torch.nn.functional.normalize(g.nan_to_num(), dim=-1)

    def metric_at(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """G(q) (B, dim, dim) and f(q) (B, dim), detached; identity and zero without a metric."""
        if self.metric is None:
            return torch.eye(q.shape[-1], dtype=q.dtype, device=q.device).expand(len(q), -1, -1), torch.zeros_like(q)
        with torch.no_grad():
            return self.metric.G(q), self.metric.drift(q)
