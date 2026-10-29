from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult

if TYPE_CHECKING:
    from ntfields.field import TrainedField


def mppi(
    field: "TrainedField",
    start: Tensor,
    goal: Tensor,
    samples: int = 50,
    horizon: int = 5,
    step: float = 0.015,
    prior_gain: float = 2.0,
    first_weight: float = 10.0,
    temperature: float = 50.0,
    goal_tol: float = 0.01,
    max_steps: int = 200,
    generator: torch.Generator | None = None,
) -> PlanResult:
    """TD-NTFields' MPPI (ntrl-demo `arm_plan_stat.py::MPPI`): only the start moves.

    Each iteration samples `samples` rollouts of `horizon` steps (noise of std `step` per rollout plus per step,
    around `prior_gain`·previous step, each step clipped to length `step`), scores them by
    `first_weight`·T(first state, goal) + T(last state, goal), and takes the softmax(-temperature·cost)-weighted
    first step.
    """
    dim = start.shape[-1]
    q, prior = start.clone(), torch.zeros_like(start)
    path, reached = [start], False
    for _ in range(max_steps):
        noise = step * _randn(samples, 1, dim, like=start, generator=generator)
        noise = noise + step * _randn(samples, horizon, dim, like=start, generator=generator)
        steps = noise + prior_gain * prior
        steps = steps / (steps.norm(dim=2, keepdim=True).clamp(min=step) / step)
        ends = (q + steps.cumsum(dim=1))[:, [0, -1]].reshape(-1, dim)
        cost = field.travel_time(ends, goal.expand_as(ends)).reshape(samples, 2)
        weight = torch.softmax(-temperature * (first_weight * cost[:, 0] + cost[:, 1]), dim=0)
        prior = weight @ steps[:, 0]
        q = q + prior
        path.append(q)
        if (goal - q).norm() < goal_tol:
            reached = True
            break
    path.append(goal)
    return PlanResult(torch.stack(path), reached)


def _randn(*shape: int, like: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=like.dtype, device=like.device)
