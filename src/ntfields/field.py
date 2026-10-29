from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ntfields.base import Method, TravelTimeModel
from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem


@dataclass
class TrainedField:
    method: Method[Any]
    problem: Problem
    model: TravelTimeModel
    ckpt_path: Path | None = None

    def travel_time(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(q_s, q_g, self.problem.env.condition())

    def gradient(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor]:
        """(∇_{q_s} T, ∇_{q_g} T)."""
        with torch.enable_grad():
            q_s = q_s.detach().requires_grad_(True)
            q_g = q_g.detach().requires_grad_(True)
            t = self.model(q_s, q_g, self.problem.env.condition())
            g_s, g_g = torch.autograd.grad(t.sum(), (q_s, q_g))
        return g_s, g_g

    def plan(self, start: Tensor, goal: Tensor, planner: str | None = None) -> PlanResult:
        """Single query with `planner` (default: the method's default planner)."""
        return PLANNERS[planner or self.method.default_planner](self, start, goal)
