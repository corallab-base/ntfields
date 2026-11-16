from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem

SEED_RADIUS_DX = 1.5  # skfmm needs a zero contour that encloses at least one node


Interpolation = Literal["linear", "cubic"]


@dataclass
class GridField:
    """Travel time to `goal` on a regular grid (array axis i = coordinate i), interpolated between the nodes.

    `interp`:
    - "cubic" (default): separable Catmull-Rom, which passes through the nodes and has a continuous gradient, so
      descent doesn't zig-zag at cell edges; it can overshoot slightly next to kinks (the seed, ridges);
    - "linear": multilinear, whose gradient jumps at every cell edge.

    A `GoalField`: queries on another device than `values` copy the grid each time, so move it once with `to`.
    """

    values: Tensor  # (R,) * dim, float64
    bounds: Tensor  # (dim, 2)
    goal: Tensor  # (dim,)
    interp: Interpolation = "cubic"

    @property
    def device(self) -> torch.device:
        return self.values.device

    def to(self, device: torch.device | str) -> "GridField":
        return GridField(self.values.to(device), self.bounds.to(device), self.goal.to(device), self.interp)

    def travel_time(self, q: Tensor) -> Tensor:
        """T(q, goal) (B,) for q (B, dim), clamped to the grid outside `bounds`."""
        with torch.no_grad():
            return self._interpolate(q)

    def gradient(self, q: Tensor) -> Tensor:
        """∇_q T(q, goal) (B, dim), by autograd through the interpolation (0 outside `bounds`)."""
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self._interpolate(q).sum(), q)
        return g

    def _interpolate(self, q: Tensor) -> Tensor:
        values = self.values.to(q.device)
        lo, hi = self.bounds.to(q.device, torch.float64).unbind(-1)
        if self.interp == "linear":
            u = (2 * (q.double() - lo) / (hi - lo) - 1).flip(-1)  # grid_sample wants (x, y[, z]) = reversed axes
            dim = q.shape[-1]
            grid = u.reshape(1, -1, *[1] * (dim - 1), dim)
            out = F.grid_sample(values[None, None], grid, align_corners=True, padding_mode="border").reshape(-1)
        else:
            out = _catmull_rom(values, (q.double() - lo) / (hi - lo))
        return out.to(q.dtype)


def _catmull_rom(values: Tensor, u: Tensor) -> Tensor:
    """Separable Catmull-Rom interpolation of `values` (R_0, ..., R_{d-1}) at u (B, d) in [0, 1]^d (clamped), with
    the edge nodes repeated past the border: 4^d taps per query, gathered at once."""
    values = values.contiguous()
    device, dim = u.device, values.dim()
    size = torch.tensor(values.shape, device=device)
    x = u.clamp(0, 1) * (size - 1)
    i0 = x.detach().floor().clamp(max=size - 2).long()
    t = x - i0
    t2, t3 = t * t, t * t * t
    # per axis: the 4 nodes i0-1..i0+2 and their weights, (B, d, 4)
    nodes = (i0[..., None] + torch.arange(-1, 3, device=device)).clamp(min=0).minimum(size[:, None] - 1)
    weights = torch.stack(
        ((-t + 2 * t2 - t3) / 2, (2 - 5 * t2 + 3 * t3) / 2, (t + 4 * t2 - 3 * t3) / 2, (t3 - t2) / 2), -1
    )
    # every combination of one node per axis, (B, 4^d, d)
    taps = torch.cartesian_prod(*[torch.arange(4, device=device)] * dim).reshape(-1, dim)
    axes = torch.arange(dim, device=device)
    flat_index = (nodes[:, axes, taps] * torch.tensor(values.stride(), device=device)).sum(-1)
    return (weights[:, axes, taps].prod(-1) * values.view(-1)[flat_index]).sum(-1)


def fmm_travel_time(
    problem: Problem, goal: Tensor, resolution: int = 201, interp: Interpolation = "cubic"
) -> GridField:
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
    return GridField(torch.from_numpy(t), problem.env.bounds.clone(), goal.clone(), interp)


@dataclass
class FMMField:
    """The FMM ground truth as a `Field`: one `fmm_travel_time` solve per distinct goal, kept in an LRU cache.

    T(q_s, q_g) is symmetric (isotropic eikonal), so ∇_{q_g} T comes from the solve seeded at q_s; both directions
    share the cache. Every new seed is a full solve: `gradient_bidirectional` moves both ends, so it solves twice per
    step (milliseconds on a 2D grid, far more in 3D) and fills the cache with seeds it never reuses. The default
    planner only moves the start.
    """

    problem: Problem
    resolution: int = 201
    interp: Interpolation = "cubic"
    cache_size: int = 64
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    default_planner: str = "mppi"
    _cache: OrderedDict[bytes, GridField] = field(default_factory=OrderedDict, init=False, repr=False)

    def towards(self, goal: Tensor) -> GridField:
        """T(·, goal) for a goal (dim,), solved on first use."""
        goal = goal.detach().to("cpu", torch.float64)
        key = goal.numpy().tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        gt = fmm_travel_time(self.problem, goal, self.resolution, self.interp).to(self.device)
        self._cache[key] = gt
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return gt

    def travel_time(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        return self._per_seed(q_g, q_s, lambda gt, q: gt.travel_time(q), q_s.new_empty(len(q_s)))

    def gradient(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor]:
        """(∇_{q_s} T, ∇_{q_g} T)."""
        g_s = self._per_seed(q_g, q_s, lambda gt, q: gt.gradient(q), torch.empty_like(q_s))
        g_g = self._per_seed(q_s, q_g, lambda gt, q: gt.gradient(q), torch.empty_like(q_g))
        return g_s, g_g

    def plan(self, start: Tensor, goal: Tensor, planner: str | None = None) -> PlanResult:
        """Single query with `planner` (default: `default_planner`)."""
        return PLANNERS[planner or self.default_planner](self, start, goal)

    def _per_seed(self, seeds: Tensor, q: Tensor, query: Callable[[GridField, Tensor], Tensor], out: Tensor) -> Tensor:
        """`out[i] = query(towards(seeds[i]), q[i])`, one solve per distinct seed."""
        unique, inverse = torch.unique(seeds, dim=0, return_inverse=True)
        for k, seed in enumerate(unique):
            rows = inverse == k
            out[rows] = query(self.towards(seed), q[rows])
        return out
