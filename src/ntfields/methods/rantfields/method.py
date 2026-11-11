import torch

from ntfields.base import Batch, LossOut, Method, TrainState, TravelTimeModel
from ntfields.data import SurfaceMix, Uniform
from ntfields.methods.rantfields.config import RaNTFieldsConfig
from ntfields.methods.rantfields.loss import rantfields_loss
from ntfields.methods.rantfields.model import RaNTFieldsModel
from ntfields.methods.rantfields.rollback import RetryRollback
from ntfields.methods.rantfields.sampler import JointSpaceMix
from ntfields.problem import Problem
from ntfields.registry import register, register_preset
from ntfields.train import Callback


@register
class RaNTFields(Method[RaNTFieldsConfig]):
    """RaNTFields: a quasimetric travel-time field trained on the Randers eikonal equation of q̇ = f(q) + u,
    uᵀG(q)u ≤ 1. Drift and metric come from the problem's `Metric`; without one this is an isotropic eikonal field."""

    name = "rantfields"
    Config = RaNTFieldsConfig
    default_planner = "riemannian_gradient"

    def build_model(self, problem: Problem) -> TravelTimeModel:
        c = self.cfg
        return RaNTFieldsModel(problem.dim, c.embed, c.hidden, c.n_residual_blocks, c.n_finsler_blocks)

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        return rantfields_loss(model, batch, self.cfg.flow_scale, self.cfg.w_boundary)

    def callbacks(self) -> list[Callback]:
        if self.cfg.rollback_threshold is None:
            return []
        return [RetryRollback(self.cfg.rollback_threshold)]

    def default_sampler(self) -> SurfaceMix | Uniform | JointSpaceMix:
        return self.cfg.sampler

    def scheduler(
        self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int | None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        epochs = self.cfg.schedule_epochs or epochs
        if epochs is None:
            raise ValueError("RaNTFields' OneCycle schedule needs an Epochs budget or cfg.schedule_epochs")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.cfg.max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=self.cfg.pct_start
        )


register_preset("rantfields_2d", RaNTFields, RaNTFieldsConfig)
register_preset("rantfields_arm", RaNTFields, RaNTFieldsConfig.arm)
