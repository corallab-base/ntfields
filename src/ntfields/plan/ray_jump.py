import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult

if TYPE_CHECKING:
    from ntfields.field import Field
    from ntfields.problem import Problem


def ray_jump(
    field: "Field",
    start: Tensor,
    goal: Tensor,
    eps: float = 0.02,
    candidates: int = 32,
    min_fraction: float = 1e-3,
    resolution: float = 0.005,
    max_speed: float = 1.0,
    goal_tol: float = 1e-3,
    max_passes: int = 100,
    fan: int = 16,
    fan_retries: int = 3,
    generator: torch.Generator | None = None,
) -> PlanResult:
    """Long jumps down the field, each certified by its slack; one batched field evaluation per jump.

    From q, `candidates` points lie along the descent ray -∇T, spaced geometrically from the longest useful jump
    (T(q)·`max_speed`, an upper bound on the remaining geodesic length) down to `min_fraction` of it, plus the goal
    itself. One `travel_time_and_gradient` pass evaluates them all. Each straight segment q -> c is checked against
    the environment every `resolution` and costs c_seg = ∫ ds / S* (midpoint rule). Its slack δ = c_seg + T(c) - T(q)
    is >= 0 for an exact field, and the path costs exactly T(start) + Σδ, so accepting only jumps with
    δ <= `eps`·c_seg bounds the path's cost by T(start) / (1 - `eps`), with T the field's own estimate (its error
    moves T(start), not the bound). The planner takes the accepted candidate with the most progress (the goal ends
    the plan). The next ray's direction comes from the same pass.

    If no candidate is accepted, the next passes add `fan` fresh random directions (`candidates` points each) to the
    ray, up to `fan_retries` times; after that, the rays shrink by `min_fraction`. The fan is for ridges of T (e.g. a
    start and goal mirrored through an obstacle), where the smoothed -∇T points between the two optimal paths and no
    jump along it has a small slack, while a long jump along either one does.

    `eps` can't go below the field's own error: near obstacles, where it is largest, short jumps stall. (An absolute
    allowance for it would let the planner crawl along ridges in short accepted steps instead of fanning out.)
    """
    problem = field.problem
    target = field.towards(goal)
    q = start[None]
    t_q, g_q = target.travel_time_and_gradient(q)
    path, passes = [start], 1
    longest = float(t_q[0]) * max_speed
    fractions = min_fraction ** (torch.arange(candidates, dtype=start.dtype, device=start.device) / (candidates - 1))
    stalls = 0  # passes since the last accepted jump

    while passes < max_passes:
        if (q[0] - goal).norm() < goal_tol:
            return PlanResult(torch.stack([*path, goal]), True)
        if longest < 1e-9:
            break

        direction = -g_q[0]
        if direction.norm() == 0:  # e.g. outside a grid field's bounds: head for the goal
            direction = goal - q[0]
        directions = (direction / direction.norm())[None]
        if stalls:
            extra = torch.randn(fan, len(direction), generator=generator, dtype=start.dtype, device=start.device)
            directions = torch.cat([directions, torch.nn.functional.normalize(extra, dim=-1)])
        steps = (longest * fractions)[None, :, None] * directions[:, None]  # (D, K, dim)
        ends = torch.cat([q + steps.reshape(-1, len(direction)), goal[None]])

        t_c, g_c = target.travel_time_and_gradient(ends)
        passes += 1
        t_c = t_c.clone()
        t_c[-1] = 0.0

        free, cost = _segments(problem, q[0], ends, resolution)
        slack = cost + t_c - t_q
        accept = free & (slack <= eps * cost)

        if accept[-1]:
            return PlanResult(torch.stack([*path, goal]), True)
        if not accept.any():
            stalls += 1
            if stalls > fan_retries:
                longest *= min_fraction
                stalls = 1
            continue

        progress = torch.where(accept, t_q - t_c, -math.inf)
        k = int(progress.argmax())
        q, t_q, g_q = ends[k : k + 1], t_c[k : k + 1], g_c[k : k + 1]
        path.append(q[0])
        longest = float(t_q[0]) * max_speed
        stalls = 0

    return PlanResult(torch.stack(path), False)


def _segments(problem: "Problem", q: Tensor, ends: Tensor, resolution: float) -> tuple[Tensor, Tensor]:
    """For the straight segments q -> ends (C, dim): collision-free (sampled every `resolution` or finer, the start
    excluded) and travel time ∫ ds / S* (midpoint rule on the same samples)."""
    seg = ends - q
    length = seg.norm(dim=-1)
    n = max(1, math.ceil(float(length.max()) / resolution))
    f = torch.arange(1, 2 * n + 1, dtype=q.dtype, device=q.device) / (2 * n)  # odd numerators: subsegment midpoints
    pts = q + f[None, :, None] * seg[:, None]  # (C, 2n, dim)
    flat = pts.reshape(-1, q.shape[-1])
    free = problem.env.is_valid(flat).reshape(pts.shape[:2]).all(dim=-1)
    speed = problem.speed_at(pts[:, 0::2].reshape(-1, q.shape[-1])).reshape(len(ends), n)
    cost = length / n * (1 / speed).sum(dim=-1)
    return free, cost.to(ends.dtype)
