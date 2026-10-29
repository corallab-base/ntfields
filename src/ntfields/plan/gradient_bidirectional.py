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
) -> PlanResult:
    """P-NTFields Eq. 12: march both ends down T with step β·S²·∇T, S = 1/‖∇T‖ (the model's own speed)."""
    q_s, q_g = start[None], goal[None]
    fwd, bwd = [q_s], [q_g]
    met = False
    for _ in range(max_steps):
        if (q_s - q_g).norm() < goal_tol:
            met = True
            break
        g_s, g_g = field.gradient(q_s, q_g)
        q_s = q_s - beta * g_s / g_s.pow(2).sum(dim=-1, keepdim=True)
        q_g = q_g - beta * g_g / g_g.pow(2).sum(dim=-1, keepdim=True)
        fwd.append(q_s)
        bwd.append(q_g)
    return PlanResult(torch.cat([*fwd, *reversed(bwd)]), met)
