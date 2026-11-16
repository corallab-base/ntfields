from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor

from ntfields.base import Method, TravelTimeModel
from ntfields.inference import InferenceModel, autograd_inference
from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem

if TYPE_CHECKING:
    from ntfields.accel import AcceleratedField, Backend, Precision


class Field(Protocol):
    """What planners and evaluation query: a TrainedField or an AcceleratedField."""

    @property
    def problem(self) -> Problem: ...
    @property
    def device(self) -> torch.device: ...
    def travel_time(self, q_s: Tensor, q_g: Tensor) -> Tensor: ...
    def gradient(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor]: ...
    def towards(self, goal: Tensor) -> "GoalField": ...


class GoalField(Protocol):
    """T(·, q_g) for one fixed goal: what `Field.towards` returns."""

    def travel_time(self, q: Tensor) -> Tensor: ...  # (B, dim) -> (B,)
    def gradient(self, q: Tensor) -> Tensor: ...  # (B, dim) -> ∇_q T (B, dim)
    def travel_time_and_gradient(self, q: Tensor) -> tuple[Tensor, Tensor]: ...  # both from one evaluation


@dataclass
class EncodedGoalField:
    """A network's T(·, q_g) for one goal, encoded once (see `TrainedField.towards`)."""

    inference: InferenceModel
    e_g: Tensor  # inference.encode_goal(q_g)

    def travel_time(self, q: Tensor) -> Tensor:
        with torch.no_grad():
            return self.inference.travel_time(q, self.e_g)

    def gradient(self, q: Tensor) -> Tensor:
        """∇_q T(q, q_g)."""
        return self.travel_time_and_gradient(q)[1]

    def travel_time_and_gradient(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """(T(q, q_g), ∇_q T(q, q_g)), from the inference model's gradient function, which computes both. Under a
        TensorRT fp16 build, T comes from the float32 gradient engine, so it can differ from `travel_time` by the
        fp16 rounding."""
        if self.inference.gradient is None:
            raise ValueError("this inference model has no gradients; use the TrainedField's")
        with torch.no_grad():
            return self.inference.gradient(q, self.e_g)


@dataclass
class TrainedField:
    method: Method[Any]
    problem: Problem
    model: TravelTimeModel
    ckpt_path: Path | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

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

    def towards(self, goal: Tensor) -> EncodedGoalField:
        """T(·, goal) for a goal (dim,), which is encoded once instead of once per query."""
        inference = autograd_inference(self.model, self.problem.env.condition())
        with torch.no_grad():
            return EncodedGoalField(inference, inference.encode_goal(goal[None]))

    def plan(self, start: Tensor, goal: Tensor, planner: str | None = None) -> PlanResult:
        """Single query with `planner` (default: the method's default planner)."""
        return PLANNERS[planner or self.method.default_planner](self, start, goal)

    def accelerate(
        self,
        backend: "Backend" = "compile",
        cuda_graphs: bool = True,
        precision: "Precision" = "fp32",
        batch: tuple[int, int, int] = (1, 128, 65536),
    ) -> "AcceleratedField":
        """This field, evaluated through its model's inference form (see `ntfields.accel.accelerate`)."""
        from ntfields.accel import accelerate

        return accelerate(self, backend, cuda_graphs, precision, batch)
