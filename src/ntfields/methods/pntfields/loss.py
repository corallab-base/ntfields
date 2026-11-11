from ntfields.base import Batch, LossOut, TrainState
from ntfields.losses import viscous_speed_ratio
from ntfields.methods.pntfields.config import PNTFieldsConfig
from ntfields.methods.pntfields.model import PNTFieldsModel


def pntfields_loss(model: PNTFieldsModel, batch: Batch, state: TrainState, cfg: PNTFieldsConfig) -> LossOut:
    """P-NTFields' `Model.Loss`: mean over both ends of x + 1/x - 2, x = √(S_visc / S*_α)."""
    dim = batch.q_s.shape[1]
    tau, dtau, hess = model.tau_jet(batch.q_s, batch.q_g)
    d = batch.q_g - batch.q_s

    alpha = cfg.alpha(state.epoch)
    target_s, target_g = alpha * batch.speed_s + 1 - alpha, alpha * batch.speed_g + 1 - alpha
    start = viscous_speed_ratio(tau, dtau[:, :dim], hess[:, :dim].sum(-1), -d, target_s, cfg.viscosity)
    goal = viscous_speed_ratio(tau, dtau[:, dim:], hess[:, dim:].sum(-1), d, target_g, cfg.viscosity)
    total = (start + goal).mean()
    return LossOut(total, {"start": float(start.detach().mean()), "goal": float(goal.detach().mean()), "alpha": alpha})
