from ntfields.base import Batch, LossOut, Method, TrainState, TravelTimeModel
from ntfields.data import NearSurface, SurfaceMix, UR5eSurfacePairs
from ntfields.methods.td_ntfields.config import TDNTFieldsConfig
from ntfields.methods.td_ntfields.loss import td_ntfields_loss
from ntfields.methods.td_ntfields.model import TDMetricModel
from ntfields.problem import Problem
from ntfields.registry import register, register_preset
from ntfields.train import Callback, LossRatioRollback, LossScaleByPrevLoss


@register
class TDNTFields(Method[TDNTFieldsConfig]):
    """TD-NTFields (Ni & Qureshi): metric-embedding travel time trained with eikonal, TD and normal terms."""

    name = "td_ntfields"
    Config = TDNTFieldsConfig
    default_planner = "mppi"

    def build_model(self, problem: Problem) -> TravelTimeModel:
        c = self.cfg
        return TDMetricModel(problem.dim, c.hidden, c.n_blocks, c.group, c.temperature, c.out_scale, c.fourier_scale)

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        return td_ntfields_loss(model, batch, self.cfg)

    def callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = [LossScaleByPrevLoss(self.cfg.loss_scale_numerator)]
        if self.cfg.rollback_threshold is not None:
            callbacks.insert(0, LossRatioRollback(self.cfg.rollback_threshold))
        return callbacks

    def default_sampler(self) -> NearSurface | SurfaceMix | UR5eSurfacePairs:
        return self.cfg.sampler


register_preset("td_ntfields_arm", TDNTFields, TDNTFieldsConfig.arm)
register_preset("td_ntfields_arm_ntrl", TDNTFields, TDNTFieldsConfig.arm_ntrl)
register_preset("td_ntfields_2d", TDNTFields, TDNTFieldsConfig.planar)
register_preset("td_ntfields_stable", TDNTFields, TDNTFieldsConfig.stable)
