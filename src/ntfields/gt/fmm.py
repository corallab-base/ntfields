from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ntfields.grid import Interpolation, interpolate
from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem

SEED_RADIUS_DX = 1.5  # skfmm needs a zero contour that encloses at least one node


@dataclass
class GridField:
    """Travel time to `goal` on a regular grid (array axis i = coordinate i), interpolated between the nodes (see
    `ntfields.grid.interpolate`). Cubic, the default, has a continuous gradient, so descent doesn't zig-zag at cell
    edges; it can overshoot slightly next to the seed and ridges.

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
        return self.travel_time_and_gradient(q)[1]

    def travel_time_and_gradient(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """(T(q, goal), ∇_q T(q, goal)) from one interpolation and its backward pass."""
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            t = self._interpolate(q)
            (g,) = torch.autograd.grad(t.sum(), q)
        return t.detach(), g

    def _interpolate(self, q: Tensor) -> Tensor:
        return interpolate(self.values, self.bounds, q, self.interp)


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
