"""Faster evaluation of a trained field, through its model's inference form (`TravelTimeModel.inference_model`).

    fast = field.accelerate("compile")  # or ntfields.accel.accelerate(field, ...)
    fast.plan(start, goal)               # planners and benchmarks take it in place of the field
    t = fast.towards(goal).travel_time(q)

Backends:
- "eager": the inference form as is (folded weights, forward-mode gradients)
- "compile": torch.compile (inductor) of each inference function, specialized per batch size (past torch._dynamo's
  recompile limit, 8 sizes per function, further sizes run eagerly)

With `cuda_graphs` (on a CUDA field), each function is also replayed from a CUDA graph per batch size, which
removes the launch overhead that dominates at planning batch sizes.

The scene is the environment's condition when `accelerate` is called: unlike a TrainedField, an AcceleratedField does
not follow later changes to the environment (no model reads the scene yet).
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from ntfields.accel.cuda_graphs import CudaGraphs
from ntfields.base import Method
from ntfields.field import GoalField, TrainedField
from ntfields.inference import InferenceModel
from ntfields.plan import PLANNERS, PlanResult
from ntfields.problem import Problem

Backend = Literal["eager", "compile"]
# on the UR5e MPPI benchmark, max-autotune builds in 52 s instead of 4 s and plans at most 15% faster; CUDA graphs
# come from CudaGraphs, not inductor's modes
COMPILE_MODE = "default"


@dataclass
class AcceleratedField:
    field: TrainedField
    inference: InferenceModel

    @property
    def problem(self) -> Problem:
        return self.field.problem

    @property
    def method(self) -> Method[Any]:
        return self.field.method

    @property
    def device(self) -> torch.device:
        return self.field.device

    def travel_time(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        with torch.no_grad():
            return self.inference.pair_travel_time(q_s, q_g)

    def gradient(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor]:
        """(∇_{q_s} T, ∇_{q_g} T)."""
        if self.inference.pair_gradient is None:
            raise ValueError(
                f"{type(self.field.model).__name__}'s inference model has no gradients; use the TrainedField's"
            )
        with torch.no_grad():
            _, g_s, g_g = self.inference.pair_gradient(q_s, q_g)
        return g_s, g_g

    def towards(self, goal: Tensor) -> GoalField:
        """T(·, goal) for a goal (dim,), which is encoded once instead of once per query."""
        with torch.no_grad():
            return GoalField(self.inference, self.inference.encode_goal(goal[None]))

    def plan(self, start: Tensor, goal: Tensor, planner: str | None = None) -> PlanResult:
        """Single query with `planner` (default: the method's default planner)."""
        return PLANNERS[planner or self.method.default_planner](self, start, goal)


def accelerate(field: TrainedField, backend: Backend = "compile", cuda_graphs: bool = True) -> AcceleratedField:
    """`field` evaluated through its model's inference form on `backend`; values and gradients match the field's
    up to rounding. Gradients the inference form leaves out are absent: asking for them raises."""
    inference = field.model.inference_model(field.problem.env.condition())
    if inference is None:
        raise ValueError(f"{type(field.model).__name__} ({field.method.name}) has no inference model to accelerate")
    if backend == "compile":
        inference = inference.map(lambda fn: torch.compile(fn, mode=COMPILE_MODE, dynamic=False))
    elif backend != "eager":
        raise ValueError(f"unknown backend {backend!r}; available: eager, compile")
    if cuda_graphs and field.device.type == "cuda":
        inference = inference.map(CudaGraphs)
    return AcceleratedField(field, inference)


__all__ = ["AcceleratedField", "Backend", "CudaGraphs", "accelerate"]
