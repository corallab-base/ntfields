import torch
from torch import Tensor


def viscous_speed_ratio(tau: Tensor, dtau: Tensor, lap: Tensor, d: Tensor, target: Tensor, gamma: float) -> Tensor:
    """x + 1/x - 2 with x = √(S_visc / target) at one end of a factorized T = ‖d‖/τ, where `d` points from that end
    to the other.

    S = 1/‖∇T‖ = τ² / √(‖d‖²‖∇τ‖² - 2τ ∇τ·d + τ²) and S_visc = 1/(1/S + γ·Δτ), as in P-NTFields' `Model.Loss`.
    """
    t2 = tau**2
    s = t2 / torch.sqrt((d * d).sum(-1) * (dtau * dtau).sum(-1) - 2 * tau * (dtau * d).sum(-1) + t2)
    s_visc = 1 / (1 / s + gamma * lap)
    r = torch.sqrt(s_visc) / torch.sqrt(target)
    return r + 1 / r - 2
