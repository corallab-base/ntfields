from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ntfields.problem import Problem

SEED_RADIUS_DX = 1.5  # skfmm needs a zero contour that encloses at least one node


@dataclass
class GridField:
    """Travel time to `goal` on a regular grid (array axis i = coordinate i), multilinearly interpolated."""

    values: Tensor  # (R,) * dim, float64
    bounds: Tensor  # (dim, 2)
    goal: Tensor  # (dim,)

    def value(self, q: Tensor) -> Tensor:
        lo, hi = self.bounds.to(q).unbind(-1)
        u = (2 * (q - lo) / (hi - lo) - 1).flip(-1)  # grid_sample wants (x, y[, z]) = reversed array axes
        dim = q.shape[-1]
        grid = u.reshape(1, -1, *[1] * (dim - 1), dim).double()
        out = F.grid_sample(self.values[None, None].to(q.device), grid, align_corners=True, padding_mode="border")
        return out.reshape(-1).to(q.dtype)


def fmm_travel_time(problem: Problem, goal: Tensor, resolution: int = 201) -> GridField:
    """Second-order fast marching of S* from `goal` (2D/3D); needs the `gt` extra (scikit-fmm)."""
    import skfmm

    bounds = problem.env.bounds.double()
    axes = [torch.linspace(float(lo), float(hi), resolution, dtype=torch.float64) for lo, hi in bounds]
    pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    speed = problem.speed_at(pts.reshape(-1, problem.dim)).reshape(pts.shape[:-1])
    s_goal = float(problem.speed_at(goal[None].double()))

    dx = ((bounds[:, 1] - bounds[:, 0]) / (resolution - 1)).tolist()
    r0 = SEED_RADIUS_DX * max(dx)
    dist = (pts - goal.double()).norm(dim=-1).numpy()
    t = np.asarray(skfmm.travel_time(dist - r0, speed.numpy(), dx=dx, order=2))  # pyright: ignore[reportArgumentType]  (per-axis dx is supported)
    t = np.where(dist <= r0, dist / s_goal, t + r0 / s_goal)
    return GridField(torch.from_numpy(t), problem.env.bounds.clone(), goal.clone())
