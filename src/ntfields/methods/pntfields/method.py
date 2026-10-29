from typing import cast

from ntfields.base import Batch, LossOut, Method, TrainState, TravelTimeModel
from ntfields.data import NearSurface
from ntfields.methods.pntfields.config import PNTFieldsConfig
from ntfields.methods.pntfields.loss import pntfields_loss
from ntfields.methods.pntfields.model import PNTFieldsModel
from ntfields.methods.pntfields.schedule import LRRamp
from ntfields.problem import Problem
from ntfields.registry import register, register_preset
from ntfields.train import Callback, LossRatioRollback, LossScaleByPrevLoss


@register
class PNTFields(Method[PNTFieldsConfig]):
    """P-NTFields (Ni & Qureshi): factorized travel time trained on a viscosity-regularized speed ratio, with a
    progressive speed curriculum."""

    name = "pntfields"
    Config = PNTFieldsConfig

    def build_model(self, problem: Problem) -> TravelTimeModel:
        c = self.cfg
        return PNTFieldsModel(
            problem.dim, c.hidden, c.encoder_blocks, c.generator_blocks, c.fourier_features, c.fourier_scale, c.beta
        )

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        return pntfields_loss(cast(PNTFieldsModel, model), batch, state, self.cfg)

    def callbacks(self) -> list[Callback]:
        c = self.cfg
        callbacks: list[Callback] = [
            LossScaleByPrevLoss(c.loss_scale_numerator),
            LRRamp(
                c.lr, c.lr_min, c.lr_ramp_start, c.lr_ramp_length
            ),  # after the rollback's snapshot, as in the reference
        ]
        if c.rollback_threshold is not None:
            callbacks.insert(0, LossRatioRollback(c.rollback_threshold))
        return callbacks

    def default_sampler(self) -> NearSurface:
        return self.cfg.sampler


register_preset("pntfields_arm", PNTFields, PNTFieldsConfig.arm)
register_preset("pntfields_2d", PNTFields, PNTFieldsConfig.planar)
