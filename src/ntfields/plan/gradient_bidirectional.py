from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ntfields.field import TrainedField


@dataclass
class PlanResult:
    path: Tensor  # (N, dim), start to goal
    success: bool  # the two fronts met; collisions are checked by ntfields.eval


def gradient_bidirectional(
    field: "TrainedField",
    start: Tensor,
    goal: Tensor,
    beta: float = 0.03,
    goal_tol: float = 0.06,
    max_steps: int = 1000,
    tangent: bool = False,
) -> PlanResult:
    """P-NTFields Eq. 12: march both ends down T with step β·S²·∇T, S = 1/‖∇T‖ (the model's own speed).

    `tangent` is CNTFields' step β·S²·∇⊥T: the gradient keeps only its component along the manifold of a
    `ConstrainedEnv`, with S still from the full gradient.
    """
    constraint = None
    if tangent:
        constraint = getattr(field.problem.env, "constraint", None)
        if constraint is None:
            raise ValueError("tangent=True needs a ConstrainedEnv")
    q_s, q_g = start[None], goal[None]
    fwd, bwd = [q_s], [q_g]
    met = False
    for _ in range(max_steps):
        if (q_s - q_g).norm() < goal_tol:
            met = True
            break
        g_s, g_g = field.gradient(q_s, q_g)
        step_s, step_g = g_s, g_g
        if constraint is not None:
            step_s, step_g = constraint.tangent(q_s, g_s), constraint.tangent(q_g, g_g)
        q_s = q_s - beta * step_s / g_s.pow(2).sum(dim=-1, keepdim=True)
        q_g = q_g - beta * step_g / g_g.pow(2).sum(dim=-1, keepdim=True)
        fwd.append(q_s)
        bwd.append(q_g)
    return PlanResult(torch.cat([*fwd, *reversed(bwd)]), met)
