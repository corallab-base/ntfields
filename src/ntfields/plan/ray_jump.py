import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult

if TYPE_CHECKING:
    from ntfields.field import Field


def ray_jump(
    field: "Field",
    start: Tensor,
    goal: Tensor,
    eps: float = 0.1,
    samples: int = 64,
    max_speed: float = 1.0,
    goal_tol: float = 1e-3,
    min_step: float = 1e-3,
    max_passes: int = 100,
    fan: int = 16,
    fan_retries: int = 3,
    shrink: float = 0.125,
    max_spacing: float | None = 0.01,
    generator: torch.Generator | None = None,
) -> PlanResult:
    """Long straight jumps down the goal field, each certified by the field alone; one batched evaluation per jump.

    From q, `samples` points lie evenly along the descent ray -∇T(·, goal), out to the longest useful jump
    (T(q)·`max_speed`, an upper bound on the remaining geodesic length, capped at `samples`·`max_spacing`), plus as
    many on the segment to the goal. One `travel_time_and_gradient` pass evaluates them all. Every sample p is a
    candidate jump, costed in the field's own metric, c = ∫ ‖∇T‖ ds from q to p (trapezoid over the samples). Since
    T(q) - T(p) = ∫ ∇T·ds along the segment, the slack δ = c - (T(q) - T(p)) is >= 0 for any field: it measures how
    far ∇T turns away from the segment, weighted by ‖∇T‖ = 1/S. A segment through an obstacle, where ∇T turns and
    ‖∇T‖ is large, has a large slack, which is how the field alone keeps jumps clear. Accepting only jumps with
    δ <= `eps`·c bounds the path's cost in the field's metric by T(start) / (1 - `eps`). The planner takes the
    accepted jump with the most progress (the goal ends the plan), and the next ray's direction comes from the same
    pass. The environment is never queried: check the path.

    The samples set the resolution: an obstacle much thinner than their spacing can slip between them, so the spacing
    is at most `max_spacing` (None: uncapped), on the rays and on the goal segment (the goal is only taken directly
    from within `samples`·`max_spacing`). Short jumps along -∇T always pass (their slack ratio vanishes with their
    length), so if no jump is accepted, the next passes add `fan` random directions (for ridges of T, where the
    smoothed -∇T points between two optimal paths), and after `fan_retries` of those, each failed pass shortens the
    rays by `shrink`, down to `min_step` (then the plan fails). Jumps shorter than `min_step` are never taken: there,
    T's rounding swamps the slack.

    Field evaluations come in two fixed batch sizes (`batch_sizes`): 2·`samples` for a plain pass (and the start)
    and (2 + `fan`)·`samples` for a fan pass, padded with copies of q, so a compiled or CUDA-graph field
    (`accelerate`) is specialized twice per field, never more; `warm_up` does both before the first plan.
    """
    target = field.towards(goal)
    dim = len(start)
    plain, fanned = batch_sizes(samples, fan)
    cap = math.inf if max_spacing is None else samples * max_spacing

    def evaluate(points: Tensor) -> tuple[Tensor, Tensor]:
        """T and ∇T at `points`, padded (with copies of the first point) to the smallest batch size that fits."""
        n = len(points)
        size = plain if n <= plain else fanned
        t, g = target.travel_time_and_gradient(torch.cat([points, points[:1].expand(size - n, -1)]))
        return t[:n], g[:n]

    q = start
    t_q, g_q = (v[0] for v in evaluate(start[None]))
    path, passes = [start], 1
    longest = min(float(t_q) * max_speed, cap)
    steps = torch.arange(1, samples + 1, dtype=start.dtype, device=start.device) / samples  # (n,) fractions of a ray
    stalls = 0  # passes since the last accepted jump

    while passes < max_passes:
        if (q - goal).norm() < goal_tol:
            return PlanResult(torch.stack([*path, goal]), True)
        if longest < min_step:
            break

        descent = -g_q if g_q.norm() > 0 else goal - q  # e.g. outside a grid field's bounds: head for the goal
        rays = (descent / descent.norm())[None]
        if stalls:
            extra = torch.randn(fan, dim, generator=generator, dtype=start.dtype, device=start.device)
            rays = torch.cat([rays, torch.nn.functional.normalize(extra, dim=-1)])
        gap = (goal - q).norm()
        rays = torch.cat([rays, ((goal - q) / gap)[None]])  # the last ray ends at the goal
        length = torch.cat([torch.full((len(rays) - 1,), longest, dtype=start.dtype, device=start.device), gap[None]])
        dist = length[:, None] * steps  # (R, n)
        points = q + dist[..., None] * rays[:, None]  # (R, n, dim)

        t_p, g_p = evaluate(points.reshape(-1, dim))
        passes += 1
        t_p, g_p = t_p.reshape(dist.shape).clone(), g_p.reshape(*dist.shape, dim)
        t_p[-1, -1] = 0.0  # T(goal, goal)

        speed_inv = torch.cat([g_q.norm().expand(len(rays), 1), g_p.norm(dim=-1)], dim=-1).to(dist)  # ‖∇T‖ = 1/S
        cost = ((speed_inv[:, :-1] + speed_inv[:, 1:]) / 2 * (length / samples)[:, None]).cumsum(dim=-1)  # (R, n)
        progress = (t_q - t_p).to(dist)
        accept = (cost - progress <= eps * cost) & (dist >= min_step)
        accept[-1, :-1] = False  # on the goal segment, only the goal itself
        accept[-1, -1] = (accept[-1, -1] & (gap <= cap)) | (gap < min_step)

        if accept[-1, -1]:
            return PlanResult(torch.stack([*path, goal]), True)
        if not accept.any():
            stalls += 1
            if stalls > fan_retries:
                longest *= shrink
            continue

        k = int(torch.where(accept, progress, -math.inf).argmax())
        r, i = divmod(k, samples)
        q, t_q, g_q = points[r, i], t_p[r, i], g_p[r, i]
        path.append(q)
        longest = min(float(t_q) * max_speed, cap)
        stalls = 0

    return PlanResult(torch.stack(path), False)


def batch_sizes(samples: int = 64, fan: int = 16) -> tuple[int, int]:
    """The two batch sizes `ray_jump` evaluates its field at: a plain pass and a fan pass."""
    return 2 * samples, (2 + fan) * samples


def warm_up(field: "Field", goal: Tensor, samples: int = 64, fan: int = 16) -> None:
    """Evaluates `field.towards(goal)` at both of `ray_jump`'s batch sizes, so that a compiled or CUDA-graph field is
    specialized before planning rather than during the first plan that needs a fan."""
    target = field.towards(goal)
    for size in batch_sizes(samples, fan):
        target.travel_time_and_gradient(goal.expand(size, -1).contiguous())
