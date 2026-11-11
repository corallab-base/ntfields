import torch
from torch import Tensor

from ntfields.base import Batch, LossOut, TravelTimeModel


def rantfields_loss(model: TravelTimeModel, batch: Batch, flow_scale: float, w_boundary: float) -> LossOut:
    """The Randers eikonal residual at the start, plus T(q_g, q_g)², both weighted by 1/(1 + T); batch mean.

    The residual is (√(∇Tᵀ G⁻¹ ∇T) - ⟨∇T, κ·f⟩ - 1/S*)², from the HJB equation of q̇ = f(q) + u with uᵀGu ≤ 1 and cost
    ∫ dt/S*. ∇ is taken at q_s, the departure point, so the drift term enters with a plus sign. κ = `flow_scale`.
    """
    q_s = batch.q_s.detach().requires_grad_(True)
    tau = model(q_s, batch.q_g, batch.scene)
    grad = torch.autograd.grad(tau, q_s, torch.ones_like(tau), retain_graph=True, create_graph=True)[0]

    tau_goal = model(batch.q_g, batch.q_g, batch.scene)
    boundary = (tau_goal**2).mean()

    slowness = 1.0 / (batch.speed_s + 1e-6)
    if batch.drift_s is not None:
        slowness = torch.sum(grad * (flow_scale * batch.drift_s), dim=-1) + slowness
    randers = (_g_norm(grad, batch.metric_s) - slowness) ** 2

    total = ((randers + w_boundary * boundary) * (1 / (1 + tau))).mean()
    return LossOut(
        total, {"randers": float(randers.detach().mean()), "boundary": float(w_boundary * boundary.detach())}
    )


def _g_norm(grad: Tensor, metric: Tensor | None) -> Tensor:
    """√(∇Tᵀ G⁻¹ ∇T); the Euclidean norm without a metric."""
    if metric is None:
        return torch.norm(grad, dim=-1)
    quad = torch.einsum("...i,...ij,...j->...", grad, torch.inverse(metric), grad)
    return torch.sqrt(quad.clamp_min(1e-12))
