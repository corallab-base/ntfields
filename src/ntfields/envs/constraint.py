from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from ntfields.base import Environment, SceneCondition


class Constraint(Protocol):
    def violation(self, q: Tensor) -> Tensor: ...  # (B, dim) -> (B,) d_M, the distance to the manifold
    def sample(self, n: int, generator: torch.Generator, device: str = "cpu") -> Tensor: ...  # (n, dim) near-manifold
    def tangent(self, q: Tensor, v: Tensor) -> Tensor: ...  # (B, dim) v without its component normal to the manifold


@dataclass
class ConstrainedEnv:
    """`base` restricted to a band around a constraint manifold.

    `distance` is ε - d(q) with CNTFields' d(q) = max(d_M, ε - d_c) (Eq. 4), which is min(ε - d_M, d_c): as for any
    environment, larger is better. With no obstacles d_c is inf, leaving the reference's d = d_M. `is_valid` is
    stricter than distance > 0: `tolerance` is the δ a path may stray from the manifold.

    `clearance` (None: Eq. 4 as written) is the d_c at which obstacles stop slowing the field, for when d_M and d_c need
    different widths: d_c is scaled by ε / clearance, so that contact still gives the same d as leaving the band.
    """

    base: Environment
    constraint: Constraint
    epsilon: float = 0.7
    tolerance: float = 0.1
    clearance: float | None = None
    bounds: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.bounds = self.base.bounds

    @property
    def dim(self) -> int:
        return self.base.dim

    def distance(self, q: Tensor) -> Tensor:
        d_c = self.base.distance(q)
        if self.clearance is not None:
            d_c = d_c * self.epsilon / self.clearance
        return torch.minimum(self.epsilon - self.constraint.violation(q), d_c)

    def is_valid(self, q: Tensor) -> Tensor:
        return self.base.is_valid(q) & (self.constraint.violation(q) < self.tolerance)

    def condition(self) -> SceneCondition:
        return self.base.condition()


@dataclass
class SphereConstraint:
    """A point on the sphere ‖q‖ = radius, with samples up to `offset` off it radially. Stands in for the paper's
    geometric manifolds, and its great circles give exact path lengths."""

    radius: float = 0.4
    offset: float = 0.1

    def violation(self, q: Tensor) -> Tensor:
        return (q.norm(dim=-1) - self.radius).abs()

    def sample(self, n: int, generator: torch.Generator, device: str = "cpu") -> Tensor:
        direction = torch.nn.functional.normalize(torch.randn(n, 3, generator=generator), dim=-1)
        return direction * (self.radius + self.offset * (2 * torch.rand(n, 1, generator=generator) - 1))

    def tangent(self, q: Tensor, v: Tensor) -> Tensor:
        n = torch.nn.functional.normalize(q, dim=-1)
        return v - (v * n).sum(-1, keepdim=True) * n
