from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from ntfields.base import SceneCondition

Point = Sequence[float]


@dataclass
class SphereBoxEnv:
    """Point mass in an axis-aligned box with sphere and box obstacles."""

    bounds: Tensor  # (dim, 2)
    sphere_centers: Tensor  # (S, dim)
    sphere_radii: Tensor  # (S,)
    box_centers: Tensor  # (K, dim)
    box_sizes: Tensor  # (K, dim), full side lengths

    @classmethod
    def build(
        cls,
        bounds: Sequence[Point],
        spheres: Sequence[tuple[Point, float]] = (),
        boxes: Sequence[tuple[Point, Point]] = (),
    ) -> Self:
        dim = len(bounds)
        return cls(
            bounds=torch.tensor(bounds, dtype=torch.float32),
            sphere_centers=torch.tensor([c for c, _ in spheres], dtype=torch.float32).reshape(-1, dim),
            sphere_radii=torch.tensor([r for _, r in spheres], dtype=torch.float32),
            box_centers=torch.tensor([c for c, _ in boxes], dtype=torch.float32).reshape(-1, dim),
            box_sizes=torch.tensor([s for _, s in boxes], dtype=torch.float32).reshape(-1, dim),
        )

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    def distance(self, q: Tensor) -> Tensor:
        """Signed distance to the nearest obstacle (inf if there are none)."""
        sphere_d = (q[:, None] - self.sphere_centers.to(q)).norm(dim=-1) - self.sphere_radii.to(q)
        diff = (q[:, None] - self.box_centers.to(q)).abs() - self.box_sizes.to(q) / 2
        box_d = diff.clamp(min=0).norm(dim=-1) + diff.amax(dim=-1).clamp(max=0)
        d = torch.cat([sphere_d, box_d], dim=1)
        if d.shape[1] == 0:
            return torch.full(q.shape[:1], torch.inf, dtype=q.dtype, device=q.device)
        return d.amin(dim=1)

    def in_bounds(self, q: Tensor) -> Tensor:
        b = self.bounds.to(q)
        return ((q >= b[:, 0]) & (q <= b[:, 1])).all(dim=-1)

    def is_valid(self, q: Tensor) -> Tensor:
        return self.in_bounds(q) & (self.distance(q) > 0)

    def condition(self) -> SceneCondition:
        return SceneCondition()
