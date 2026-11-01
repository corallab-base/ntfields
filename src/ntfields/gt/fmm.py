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

    def descend(self, start: Tensor, step: float, directions: int = 256, max_steps: int = 100_000) -> Tensor | None:
        """Path from `start` to the goal that moves `step` per iteration to whichever of `directions` evenly spread
        neighbours has the lowest T (robust at the kinks where two routes tie); None if T stops decreasing."""
        dirs = _sphere_directions(self.goal.shape[0], directions)
        q, goal = start.double(), self.goal.double()
        t = self.value(q[None]).double()[0]
        path = [q]
        for _ in range(max_steps):
            if (q - goal).norm() <= step:
                return torch.stack([*path, goal]).to(start.dtype)
            cand = q + step * dirs
            t_cand = self.value(cand).double().nan_to_num(torch.inf)
            k = int(t_cand.argmin())
            if not t_cand[k] < t:
                return None
            q, t = cand[k], t_cand[k]
            path.append(q)
        return None


def _sphere_directions(dim: int, n: int) -> Tensor:
    """n roughly uniform unit vectors: evenly spaced angles in 2D, a Fibonacci lattice in 3D."""
    i = torch.arange(n, dtype=torch.float64)
    if dim == 2:
        a = 2 * torch.pi * i / n
        return torch.stack([a.cos(), a.sin()], dim=-1)
    if dim == 3:
        z = 1 - 2 * (i + 0.5) / n
        a = torch.pi * (3 - 5**0.5) * i
        r = (1 - z * z).sqrt()
        return torch.stack([r * a.cos(), r * a.sin(), z], dim=-1)
    raise ValueError(f"descend supports 2D and 3D, not {dim}D")


def fmm_travel_time(problem: Problem, goal: Tensor, resolution: int = 201, margin: float | None = None) -> GridField:
    """Second-order fast marching of S* from `goal` (2D/3D); needs the `gt` extra (scikit-fmm).

    With `margin`, nodes with clearance <= margin are impassable (T = inf there and behind them); without it, S*
    alone sets the cost, so paths may cross obstacles at the speed S* gives them."""
    import skfmm

    bounds = problem.env.bounds.double()
    axes = [torch.linspace(float(lo), float(hi), resolution, dtype=torch.float64) for lo, hi in bounds]
    pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    speed = problem.speed_at(pts.reshape(-1, problem.dim)).reshape(pts.shape[:-1])
    s_goal = float(problem.speed_at(goal[None].double()))

    dx = ((bounds[:, 1] - bounds[:, 0]) / (resolution - 1)).tolist()
    r0 = SEED_RADIUS_DX * max(dx)
    dist = (pts - goal.double()).norm(dim=-1).numpy()
    phi: np.ndarray = dist - r0
    if margin is not None:
        blocked = (problem.env.distance(pts.reshape(-1, problem.dim)) <= margin).reshape(pts.shape[:-1]).numpy()
        phi = np.ma.MaskedArray(phi, blocked & (dist > r0))
    t = skfmm.travel_time(phi, speed.numpy(), dx=dx, order=2)  # pyright: ignore[reportArgumentType]  (per-axis dx is supported)
    t = np.ma.filled(np.ma.MaskedArray(t, np.ma.getmask(t)).astype(np.float64), np.inf)
    t = np.where(dist <= r0, dist / s_goal, t + r0 / s_goal)
    return GridField(torch.from_numpy(t), problem.env.bounds.clone(), goal.clone())
