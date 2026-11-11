from typing import cast

from ntfields.base import Batch, LossOut, Method, TrainState, TravelTimeModel
from ntfields.data import ManifoldPairs
from ntfields.methods.cntfields.config import CNTFieldsConfig
from ntfields.methods.cntfields.loss import cntfields_loss
from ntfields.nn import FactorizedTauModel
from ntfields.problem import Problem
from ntfields.registry import register, register_preset
from ntfields.train import Callback, LossRatioRollback


@register
class CNTFields(Method[CNTFieldsConfig]):
    """CNTFields (Ni & Qureshi): travel time on a constraint manifold, from a speed that falls off with the distance to
    it. The manifold lives in the problem (`ConstrainedEnv`, `speed.GaussianBand`), so the method is a factorized
    network with a viscous speed-ratio loss and a linear curriculum."""

    name = "cntfields"
    Config = CNTFieldsConfig

    def build_model(self, problem: Problem) -> TravelTimeModel:
        c = self.cfg
        return FactorizedTauModel(
            problem.dim,
            c.hidden,
            c.encoder_blocks,
            c.generator_blocks,
            c.fourier_features,
            c.fourier_scale,
            c.beta,
            c.fourier_truncate,
        )

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        return cntfields_loss(cast(FactorizedTauModel, model), batch, state, self.cfg)

    def callbacks(self) -> list[Callback]:
        threshold = self.cfg.rollback_threshold
        return [] if threshold is None else [LossRatioRollback(threshold)]

    def default_sampler(self) -> ManifoldPairs:
        return ManifoldPairs()


register_preset("cntfields_3d", CNTFields, CNTFieldsConfig.compressed)
