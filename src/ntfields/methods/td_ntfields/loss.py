import torch
from torch import Tensor

from ntfields.base import Batch, LossOut, TravelTimeModel
from ntfields.methods.td_ntfields.config import TDNTFieldsConfig


def td_ntfields_loss(model: TravelTimeModel, batch: Batch, cfg: TDNTFieldsConfig) -> LossOut:
    """ntrl-demo's `Function.Loss`: (eikonal + TD + normal alignment) · exp(-λ_C·T), batch mean.

    The operation order is fixed (autograd sums parameter gradients in creation order), so runs are reproducible
    bit for bit.
    """
    q_s = batch.q_s.detach().requires_grad_(True)
    q_g = batch.q_g.detach().requires_grad_(True)
    tau = model(q_s, q_g, batch.scene)
    ones = torch.ones_like(tau)
    grad_s = torch.autograd.grad(tau, q_s, ones, retain_graph=True, create_graph=True)[0]
    grad_g = torch.autograd.grad(tau, q_g, ones, retain_graph=True, create_graph=True)[0]

    with torch.no_grad():
        step_s, length_s = _td_step(grad_s, batch.speed_s, cfg.dt)
        tau_s = model(q_s - step_s, q_g, batch.scene)
        step_g, length_g = _td_step(grad_g, batch.speed_g, cfg.dt)
        tau_g = model(q_s, q_g - step_g, batch.scene)
    td = _td(tau, tau_s, length_s, cfg) + _td(tau, tau_g, length_g, cfg)
    eikonal = _eikonal(grad_s, batch.speed_s, cfg) + _eikonal(grad_g, batch.speed_g, cfg)
    normal = cfg.w_normal * (
        _normal(grad_s, batch.speed_s, batch.normal_s) + _normal(grad_g, batch.speed_g, batch.normal_g)
    )

    t = tau.detach() if cfg.detach_causality else tau
    weight = torch.exp(-cfg.causality * t)
    n = tau.shape[0]
    total = torch.sum((eikonal + normal + td) * weight) / n
    terms = {"eikonal": eikonal, "td": td, "normal": normal}
    return LossOut(total, {k: float((v * weight).detach().sum() / n) for k, v in terms.items()})


def _eikonal(grad: Tensor, speed: Tensor, cfg: TDNTFieldsConfig) -> Tensor:
    """λ_E (√(S*·‖∇T‖) - 1)²."""
    grad_norm = torch.sqrt(torch.einsum("ij,ij->i", grad, grad) + 1e-8)
    return cfg.w_eikonal * (torch.sqrt(speed * grad_norm) - 1) ** 2


def _td_step(grad: Tensor, speed: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """Step (Δt/S*)·S*²·∇T down the field, and the travel time it should take, Δt/S*."""
    length = dt / speed.unsqueeze(1)
    return length * (grad * speed.unsqueeze(1) ** 2), length.squeeze(1)


def _td(tau: Tensor, tau_after_step: Tensor, length: Tensor, cfg: TDNTFieldsConfig) -> Tensor:
    """λ_TD (T - (T_after_step + Δt/S*))², zero where T < Δt/S*."""
    residual = cfg.w_td * (tau - (tau_after_step + length)) ** 2
    return torch.where(tau < length, torch.zeros_like(residual), residual)


def _normal(grad: Tensor, speed: Tensor, normal: Tensor) -> Tensor:
    """(1.01 - S*)·‖S*·∇T + n‖²: near obstacles T should increase toward them."""
    s = speed.unsqueeze(1)
    return torch.sum((1.01 - s) * (s * grad + normal) ** 2, dim=1)
