import torch
from torch import Tensor

from ntfields.base import Batch, LossOut, TrainState
from ntfields.methods.pntfields.config import PNTFieldsConfig
from ntfields.methods.pntfields.model import PNTFieldsModel


def pntfields_loss(model: PNTFieldsModel, batch: Batch, state: TrainState, cfg: PNTFieldsConfig) -> LossOut:
    """P-NTFields' `Model.Loss`: mean over both ends of x + 1/x - 2, x = √(S_visc / S*_α)."""
    dim = batch.q_s.shape[1]
    tau, dtau, hess = model.tau_jet(batch.q_s, batch.q_g)
    d = batch.q_g - batch.q_s

    alpha = cfg.alpha(state.epoch)
    start = _ratio_loss(tau, dtau[:, :dim], hess[:, :dim].sum(-1), -d, batch.speed_s, alpha, cfg.viscosity)
    goal = _ratio_loss(tau, dtau[:, dim:], hess[:, dim:].sum(-1), d, batch.speed_g, alpha, cfg.viscosity)
    total = (start + goal).mean()
    return LossOut(total, {"start": float(start.detach().mean()), "goal": float(goal.detach().mean()), "alpha": alpha})


def _ratio_loss(tau: Tensor, dtau: Tensor, lap: Tensor, d: Tensor, speed: Tensor, alpha: float, gamma: float) -> Tensor:
    """x + 1/x - 2 with x = √(S_visc / S*_α) at one end, where `d` points from that end to the other.

    S = 1/‖∇T‖ = τ² / √(‖d‖²‖∇τ‖² - 2τ ∇τ·d + τ²) for T = ‖d‖/τ, and S_visc = 1/(1/S + γ·Δτ).
    """
    t2 = tau**2
    s = t2 / torch.sqrt((d * d).sum(-1) * (dtau * dtau).sum(-1) - 2 * tau * (dtau * d).sum(-1) + t2)
    s_visc = 1 / (1 / s + gamma * lap)
    target = alpha * speed + 1 - alpha
    r = torch.sqrt(s_visc) / torch.sqrt(target)
    return r + 1 / r - 2
