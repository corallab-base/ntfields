from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ntfields.base import Method, SceneCondition, TravelTimeModel
from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem


@dataclass
class GoalField:
    """T(·, q_g) for one goal, encoded once (see `TrainedField.towards`)."""

    model: TravelTimeModel
    scene: SceneCondition
    e_g: Tensor  # the model's encode_goal(q_g)

    def travel_time(self, q: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.travel_time_from_encoded_goal(q, self.e_g, self.scene)

    def gradient(self, q: Tensor) -> Tensor:
        """∇_q T(q, q_g)."""
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self.model.travel_time_from_encoded_goal(q, self.e_g, self.scene).sum(), q)
        return g


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

    def towards(self, goal: Tensor) -> GoalField:
        """T(·, goal) for a goal (dim,), which is encoded once instead of once per query."""
        scene = self.problem.env.condition()
        with torch.no_grad():
            return GoalField(self.model, scene, self.model.encode_goal(goal, scene))

    def plan(self, start: Tensor, goal: Tensor, planner: str | None = None) -> PlanResult:
        """Single query with `planner` (default: the method's default planner)."""
        return PLANNERS[planner or self.method.default_planner](self, start, goal)
