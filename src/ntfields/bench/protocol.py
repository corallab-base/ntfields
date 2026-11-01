import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.problem import Problem


@dataclass(frozen=True)
class Protocol:
    """How a suite version judges a path. Every field is part of the frozen suite."""

    margin: float  # a configuration is valid iff it is inside the bounds with clearance > margin
    resolution: float  # paths are checked and costed on segments at most this long
    endpoint_tol: float = 1e-4  # the path must start at the start and end at the goal within this distance
    reference_tol: float = 0.02  # an entry beating the reference cost c* by more than this fraction flags the reference


@dataclass
class Outcome:
    success: bool
    cost: float | None  # travel time ∫ds/S* along the path; None unless success
    length: float | None


def is_valid(problem: Problem, q: Tensor, margin: float) -> Tensor:
    b = problem.env.bounds.to(q)
    inside = ((q >= b[:, 0]) & (q <= b[:, 1])).all(dim=-1)
    return inside & (problem.env.distance(q) > margin)


def densify(path: Tensor, resolution: float) -> Tensor:
    """The piecewise-linear path resampled so that no segment is longer than `resolution`."""
    seg = path.diff(dim=0)
    n = (seg.norm(dim=-1) / resolution).ceil().clamp(min=1).long()
    idx = torch.repeat_interleave(torch.arange(len(seg)), n)  # the segment each new point lies on
    first = torch.cumsum(n, 0) - n  # index of each segment's first point
    frac = (torch.arange(len(idx)) - first[idx]).to(path.dtype) / n[idx]
    return torch.cat([path[idx] + frac[:, None] * seg[idx], path[-1:]])


def evaluate_path(problem: Problem, protocol: Protocol, start: Tensor, goal: Tensor, path: Tensor) -> Outcome:
    """Success iff the path joins start to goal and every point of its densified version is valid. The planner's own
    convergence flag plays no part: a path is judged only as geometry."""
    path = path.detach().cpu().double()
    ends_ok = (path[0] - start.cpu().double()).norm() <= protocol.endpoint_tol and (
        path[-1] - goal.cpu().double()
    ).norm() <= protocol.endpoint_tol
    if not ends_ok or not torch.isfinite(path).all():
        return Outcome(False, None, None)
    dense = densify(path, protocol.resolution)
    if not bool(is_valid(problem, dense, protocol.margin).all()):
        return Outcome(False, None, None)
    seg = dense.diff(dim=0)
    speed = problem.speed_at((dense[:-1] + seg / 2).float()).double()
    cost = float((seg.norm(dim=-1) / speed).sum())
    return Outcome(True, cost, float(path.diff(dim=0).norm(dim=-1).sum()))


def spl(success: bool, cost: float | None, c_star: float) -> float:
    """Success weighted by path cost: c*/max(c, c*) for a success, 0 for a failure."""
    if not success or cost is None:
        return 0.0
    return c_star / max(cost, c_star)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(torch.tensor(values, dtype=torch.float64).quantile(q / 100))
