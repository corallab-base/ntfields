from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor


def point_triangle_distance(p: Tensor, triangles: Tensor, chunk: int = 2048) -> Tensor:
    """Exact unsigned distance from points p (N, 3) to the nearest of `triangles` (T, 3, 3)."""
    a, b, c = triangles.to(p).unbind(1)  # (T, 3) each
    n = torch.linalg.cross(b - a, c - a)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    edges = ((a, b), (b, c), (c, a))
    out = []
    for q in p.split(chunk):
        q = q[:, None]  # (n, 1, 3)
        ap = q - a
        h = (ap * n).sum(-1)  # signed height above the plane
        foot = q - h[..., None] * n
        # the foot of the perpendicular is inside iff it is on the inner side of all three edges
        inside = torch.ones_like(h, dtype=torch.bool)
        for u, v in edges:
            inside &= (torch.linalg.cross((v - u).expand_as(foot), foot - u) * n).sum(-1) >= 0
        d2 = torch.where(inside, h * h, torch.full_like(h, torch.inf))
        for u, v in edges:
            e = v - u
            t = ((q - u) * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-24)
            d2 = torch.minimum(d2, (q - u - t.clamp(0, 1)[..., None] * e).pow(2).sum(-1))
        out.append(d2.amin(dim=1))
    return torch.cat(out).sqrt()


@dataclass
class MeshDistance:
    """Unsigned distance to a triangle mesh: trilinear lookup in a precomputed grid (capped at `cap` and beyond the
    grid), recomputed exactly where the grid reads below `exact_below`."""

    triangles: Tensor  # (T, 3, 3)
    grid: Tensor  # (X, Y, Z): the distance at origin + spacing · index
    origin: Tensor  # (3,)
    spacing: float
    cap: float
    exact_below: float

    @classmethod
    def build(
        cls, triangles: Tensor, spacing: float = 0.005, cap: float = 0.1, exact_below: float = 0.01, device: str = "cpu"
    ) -> Self:
        """Grid over the mesh's bounding box padded by `cap`; takes minutes for ~10⁷ nodes on a GPU."""
        tri = triangles.to(device)
        lo = tri.reshape(-1, 3).amin(0) - cap
        shape = (((tri.reshape(-1, 3).amax(0) + cap - lo) / spacing).ceil().long() + 1).tolist()
        axes = [lo[i] + spacing * torch.arange(shape[i], device=device, dtype=tri.dtype) for i in range(3)]
        grid = torch.empty(shape, device=device, dtype=tri.dtype)
        yz = torch.stack(torch.meshgrid(axes[1], axes[2], indexing="ij"), dim=-1).reshape(-1, 2)
        for i, x in enumerate(axes[0]):
            p = torch.cat([x.expand(len(yz), 1), yz], dim=1)
            grid[i] = point_triangle_distance(p, tri).clamp_max(cap).view(shape[1], shape[2])
        return cls(triangles.cpu(), grid.cpu(), lo.cpu(), spacing, cap, exact_below)

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            self.triangles.to(device), self.grid.to(device), self.origin.to(device), self.spacing, self.cap,
            self.exact_below,
        )  # fmt: skip

    def __call__(self, p: Tensor) -> Tensor:
        """Distances (N,) for points p (N, 3)."""
        size = torch.tensor(self.grid.shape, device=p.device, dtype=p.dtype)
        u = 2 * (p - self.origin.to(p)) / (self.spacing * (size - 1)) - 1  # grid_sample coordinates in [-1, 1]
        sample = F.grid_sample(
            self.grid.to(p).permute(2, 1, 0)[None, None],  # (1, 1, Z, Y, X), indexed by (x, y, z) coordinates
            u.view(1, -1, 1, 1, 3),
            align_corners=True,
            padding_mode="border",
        ).view(-1)
        outside = ((u < -1) | (u > 1)).any(dim=-1)
        d = torch.where(outside, torch.full_like(sample, self.cap), sample)
        near = d < self.exact_below
        if near.any():
            d = d.clone()
            d[near] = point_triangle_distance(p[near], self.triangles)
        return d
