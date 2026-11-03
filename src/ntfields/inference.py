"""Inference-time evaluation of a travel-time model, as functions of tensors only.

A model's inference form (`TravelTimeModel.inference_model`) is what backends compile or export (ntfields.accel):
plain forward networks, so no autograd at query time. The gradients are optional: a model without forward-mode ones
leaves them out, and its accelerated field has none. `autograd_inference` builds every function from any model's own
methods, which is what TrainedField queries through.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from ntfields.base import SceneCondition, TravelTimeModel

Function = Literal["encode_goal", "travel_time", "pair_travel_time", "gradient", "pair_gradient"]


@dataclass
class InferenceModel:
    encode_goal: Callable[[Tensor], Tensor]  # q_g (1, dim) -> e_g (1, n)
    travel_time: Callable[[Tensor, Tensor], Tensor]  # q (B, dim), e_g (1, n) -> T (B,)
    pair_travel_time: Callable[[Tensor, Tensor], Tensor]  # q_s, q_g (B, dim) -> T (B,)
    gradient: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None  # q, e_g -> T, ∇_q T (B, dim)
    # q_s, q_g -> T, ∇_{q_s} T, ∇_{q_g} T
    pair_gradient: Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]] | None = None
    # the functions are nn.Modules that also run with float16 weights (`.half()`) on float32 inputs, giving float32
    # outputs (what a TensorRT fp16 build needs)
    float16_weights: bool = False

    def map(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> "InferenceModel":
        """The same model with every function it has wrapped (e.g. compiled)."""
        return InferenceModel(
            wrap(self.encode_goal),
            wrap(self.travel_time),
            wrap(self.pair_travel_time),
            None if self.gradient is None else wrap(self.gradient),
            None if self.pair_gradient is None else wrap(self.pair_gradient),
            self.float16_weights,
        )


def autograd_inference(model: TravelTimeModel, scene: SceneCondition) -> InferenceModel:
    """The inference functions of any model, from its forward, encode_goal and travel_time_from_encoded_goal, with
    gradients by autograd."""

    def gradient(q: Tensor, e_g: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            t = model.travel_time_from_encoded_goal(q, e_g, scene)
            (g,) = torch.autograd.grad(t.sum(), q)
        return t.detach(), g

    def pair_gradient(q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            q_s = q_s.detach().requires_grad_(True)
            q_g = q_g.detach().requires_grad_(True)
            t = model(q_s, q_g, scene)
            g_s, g_g = torch.autograd.grad(t.sum(), (q_s, q_g))
        return t.detach(), g_s, g_g

    return InferenceModel(
        encode_goal=lambda q_g: model.encode_goal(q_g[0], scene),
        travel_time=lambda q, e_g: model.travel_time_from_encoded_goal(q, e_g, scene),
        pair_travel_time=lambda q_s, q_g: model(q_s, q_g, scene),
        gradient=gradient,
        pair_gradient=pair_gradient,
    )
