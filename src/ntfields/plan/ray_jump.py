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
    min_step: float = 1e-3,
    max_passes: int = 100,
    fan: int = 16,
    fan_retries: int = 3,
    max_eps: float = 0.5,
    batch: int | None = None,
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
    ray. The fan is for ridges of T (e.g. a start and goal mirrored through an obstacle), where the smoothed -∇T
    points between the two optimal paths and no jump along it has a small slack, while a long jump along either one
    does. After `fan_retries` failed fans, each further failed pass doubles the tolerance for this jump, up to
    `max_eps` (then the plan fails): that is for regions where the field's own error exceeds `eps` (a trained field's
    is largest next to obstacles), where the slack ratio doesn't shrink with the jump. The tolerance resets after
    each jump, and the bound holds with the largest tolerance used in place of `eps`.

    Jumps shorter than `min_step` are never taken: there, T's rounding (float32) swamps the slack, and accepting them
    would crawl in place instead of fanning out.

    Every field evaluation has the same batch size, `batch` (default: the largest pass, (1 + `fan`)·`candidates` + 1),
    padded with copies of q, so a compiled or CUDA-graph field (`accelerate`) is specialized once, never during a
    plan.

    `eps` can't go below the field's own error, relative to the jump: a trained field's needs `eps` ~ 0.1, the FMM's
    ~ 0.02. (An absolute allowance for it would let the planner crawl along ridges in short accepted steps instead of
    fanning out.)
    """
    problem = field.problem
    target = field.towards(goal)
    size = (1 + fan) * candidates + 1 if batch is None else batch

    def evaluate(points: Tensor) -> tuple[Tensor, Tensor]:
        """T and ∇T at `points`, evaluated as a batch of exactly `size` (padded with copies of the first point)."""
        n = len(points)
        if n > size:
            raise ValueError(f"a pass of {n} points exceeds batch={size}")
        padded = torch.cat([points, points[:1].expand(size - n, -1)])
        t, g = target.travel_time_and_gradient(padded)
        return t[:n], g[:n]

    q = start[None]
    t_q, g_q = evaluate(q)
    path, passes = [start], 1
    longest = float(t_q[0]) * max_speed
    fractions = min_fraction ** (torch.arange(candidates, dtype=start.dtype, device=start.device) / (candidates - 1))
    stalls, tol = 0, eps  # passes since the last accepted jump; this jump's slack tolerance

    while passes < max_passes:
        if (q[0] - goal).norm() < goal_tol:
            return PlanResult(torch.stack([*path, goal]), True)
        if longest < min_step:
            break

        direction = -g_q[0]
        if direction.norm() == 0:  # e.g. outside a grid field's bounds: head for the goal
            direction = goal - q[0]
        directions = (direction / direction.norm())[None]
        if stalls:
            extra = torch.randn(fan, len(direction), generator=generator, dtype=start.dtype, device=start.device)
            directions = torch.cat([directions, torch.nn.functional.normalize(extra, dim=-1)])
        to_goal = goal - q[0]
        gap = to_goal.norm()
        rays = torch.cat([directions, (to_goal / gap)[None]])  # the goal is the one candidate on the last ray
        dist = torch.cat([(longest * fractions).expand(len(directions), -1), gap.expand(1, candidates)])  # (R, K)
        ends = q + dist[..., None] * rays[:, None]  # (R, K, dim)
        ends = torch.cat([ends[:-1].reshape(-1, len(direction)), goal[None]])

        t_c, g_c = evaluate(ends)
        passes += 1
        t_c = t_c.clone()
        t_c[-1] = 0.0

        free, cost = _rays(problem, q[0], rays, dist, resolution)
        free, cost = torch.cat([free[:-1].reshape(-1), free[-1:, 0]]), torch.cat([cost[:-1].reshape(-1), cost[-1:, 0]])
        slack = cost + t_c - t_q
        long_enough = (ends - q).norm(dim=-1) >= min_step
        long_enough[-1] = True  # the goal
        accept = free & long_enough & (slack <= tol * cost)

        if accept[-1]:
            return PlanResult(torch.stack([*path, goal]), True)
        if not accept.any():
            stalls += 1
            if stalls > fan_retries:
                if tol >= max_eps:
                    break
                tol = min(2 * tol, max_eps)
            continue

        progress = torch.where(accept, t_q - t_c, -math.inf)
        k = int(progress.argmax())
        q, t_q, g_q = ends[k : k + 1], t_c[k : k + 1], g_c[k : k + 1]
        path.append(q[0])
        longest = float(t_q[0]) * max_speed
        stalls, tol = 0, eps

    return PlanResult(torch.stack(path), False)


def _rays(problem: "Problem", q: Tensor, rays: Tensor, dist: Tensor, resolution: float) -> tuple[Tensor, Tensor]:
    """For the candidates q + dist[r, k]·rays[r] (rays (R, dim) unit, dist (R, K)): whether the segment from q is
    collision-free, and its travel time ∫ ds / S*. Each ray is sampled once, up to its longest candidate, every
    `resolution` or finer (validity at the samples, the start excluded, and at each candidate; speed at the midpoints
    between them); every candidate on it reads its own prefix (cumulative cost, interpolated)."""
    r = len(dist)
    dim = q.shape[-1]
    length = dist.amax(dim=-1)  # (R,)
    n = max(1, math.ceil(float(length.max()) / resolution))
    h = length / n  # (R,) sample spacing per ray
    s = h[:, None] * torch.arange(1, n + 1, dtype=q.dtype, device=q.device)  # (R, n) distances of the samples
    along = torch.cat([s, s - h[:, None] / 2, dist], dim=-1)  # samples, midpoints, candidates
    pts = (q + along[..., None] * rays[:, None]).reshape(-1, dim)
    valid = problem.env.is_valid(pts).reshape(r, -1)
    speed = problem.speed_at(pts.reshape(r, -1, dim)[:, n : 2 * n].reshape(-1, dim)).reshape(r, n)

    cum = torch.cat([torch.zeros_like(h)[:, None], (h[:, None] / speed.to(h)).cumsum(dim=-1)], dim=-1)  # (R, n + 1)
    x = dist / h[:, None].clamp(min=torch.finfo(h.dtype).tiny)
    i0 = x.floor().clamp(max=n - 1).long()
    w = x - i0
    cost = cum.gather(1, i0) * (1 - w) + cum.gather(1, i0 + 1) * w
    first_invalid = torch.where(valid[:, :n], torch.inf, s).amin(dim=-1, keepdim=True)  # (R, 1)
    free = (dist < first_invalid) & valid[:, 2 * n :]
    return free, cost.to(dist.dtype)
