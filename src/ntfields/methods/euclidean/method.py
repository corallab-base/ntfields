from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ntfields.autodiff import grad
from ntfields.base import Batch, LossOut, Method, MethodConfig, SceneCondition, TrainState, TravelTimeModel
from ntfields.problem import Problem
from ntfields.registry import register


@dataclass
class EuclideanConfig(MethodConfig):
    lr: float = 1e-2


class EuclideanModel(TravelTimeModel):
    """T = c·‖q_g - q_s‖ with a single learned scale c."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition) -> Tensor:
        return self.log_scale.exp() * (q_g - q_s).norm(dim=-1)


@register
class Euclidean(Method[EuclideanConfig]):
    """Sanity baseline: ignores obstacles, fits c to the eikonal residual at the goal."""

    name = "euclidean"
    Config = EuclideanConfig

    def build_model(self, problem: Problem) -> TravelTimeModel:
        return EuclideanModel()

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        q_g = batch.q_g.detach().requires_grad_(True)
        t = model(batch.q_s, q_g, batch.scene)
        residual = grad(t, q_g).norm(dim=-1) * batch.speed_g - 1
        loss = residual.pow(2).mean()
        return LossOut(loss, {"eikonal": loss.item()})
