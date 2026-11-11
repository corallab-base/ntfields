from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult

if TYPE_CHECKING:
    from ntfields.field import TrainedField


def riemannian_direction(
    field: "TrainedField", q: Tensor, goal: Tensor, beta: float = 0.1, lam: float | None = None
) -> tuple[Tensor, float]:
    """Unit descent direction at q (dim,) toward `goal`, and T(q → goal).

    The optimal control for q̇ = f(q) + u, uᵀG(q)u ≤ 1 is u* = -G⁻¹∇T / √(∇ᵀG⁻¹∇T); the direction is that of f + u*,
    blended with the straight line to the goal as λ·dir + (1 - λ)·(goal - q)/‖goal - q‖, λ = βT/(1 + βT) (or the
    fixed `lam`): far from the goal the field steers, near it the goal attracts, where a small error in ∇T would
    otherwise stall. Zero if ∇T vanishes.
    """
    with torch.enable_grad():
        x = q[None].detach().requires_grad_(True)
        tau = field.model(x, goal[None], field.problem.env.condition())
        (grad,) = torch.autograd.grad(tau.sum(), x)
    grad, t = grad[0], float(tau.detach()[0])
    if grad.norm() <= 1e-8:
        return torch.zeros_like(q), t
    G, f = field.problem.metric_at(q[None])
    g_inv_grad = torch.linalg.solve(G[0], grad)
    u_star = -g_inv_grad / torch.sqrt(torch.dot(grad, g_inv_grad).clamp(min=1e-12))
    velocity = f[0] + u_star
    direction = velocity / velocity.norm() if velocity.norm() > 1e-8 else -grad / grad.norm()
    lam = beta * t / (1 + beta * t) if lam is None else lam
    to_goal = goal - q
    to_goal = to_goal / to_goal.norm() if to_goal.norm() > 1e-8 else torch.zeros_like(to_goal)
    blended = lam * direction + (1 - lam) * to_goal
    return (blended / blended.norm() if blended.norm() > 1e-8 else direction), t


def riemannian_gradient(
    field: "TrainedField",
    start: Tensor,
    goal: Tensor,
    step: float = 0.01,
    goal_tol: float = 0.02,
    max_steps: int = 2000,
    beta: float = 0.1,
    lam: float | None = None,
    reversal_cos: float = -0.5,
    backoff: float = 0.5,
    growth: float = 1.15,
    min_backoff: float = 0.05,
) -> PlanResult:
    """Integrate `riemannian_direction` from start to goal with step `step` (at most a tenth of the distance left),
    clamped to the bounds. When consecutive directions reverse (cosine below `reversal_cos`) the step shrinks by
    `backoff` (down to `min_backoff`) and regrows by `growth`, to escape two-cycles across narrow features."""
    bounds = field.problem.env.bounds.to(start)
    q, path = start.clone(), [start.clone()]
    step_size, multiplier, prev = step, 1.0, None
    for _ in range(max_steps):
        dist = float((q - goal).norm())
        if dist < goal_tol:
            path.append(goal.clone())
            return PlanResult(torch.stack(path), True)
        direction, _ = riemannian_direction(field, q, goal, beta, lam)
        if direction.norm() == 0:
            break
        if prev is not None:
            if float(torch.dot(direction, prev)) < reversal_cos:
                multiplier = max(multiplier * backoff, min_backoff)
            else:
                multiplier = min(1.0, multiplier * growth)
        q = torch.minimum(torch.maximum(q + step_size * multiplier * direction, bounds[:, 0]), bounds[:, 1])
        path.append(q.clone())
        step_size = min(step, 0.1 * dist)
        prev = direction
    return PlanResult(torch.stack(path), False)
