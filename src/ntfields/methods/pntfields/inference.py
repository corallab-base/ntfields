"""Inference-only form of PNTFieldsModel.

The model is already plain linear layers and softplus, so nothing is folded: the goal is encoded once as
[q_g, f(q_g)] (the numerator ‖q_g - q_s‖ needs q_g itself), and gradients are forward mode. They push first-order
jets (x, dx), dx (B, k, n) the derivatives along k input coordinates, through the encoder and the generator, as
`PNTFieldsModel.tau_jet` does without its second derivatives: k = dim for a fixed goal (its encoding is constant),
2·dim for pairs.
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ntfields.inference import InferenceModel
from ntfields.methods.pntfields.model import PNTFieldsModel, ResidualBlock

Jet = tuple[Tensor, Tensor]


def inference_model(model: PNTFieldsModel) -> InferenceModel:
    return InferenceModel(
        encode_goal=EncodeGoal(model),
        travel_time=GoalTravelTime(model),
        pair_travel_time=PairTravelTime(model),
        gradient=GoalTravelTimeGrad(model),
        pair_gradient=PairTravelTimeGrad(model),
    )


def _linear_jet(layer: nn.Linear, jet: Jet) -> Jet:
    x, dx = jet
    return layer(x), dx @ layer.weight.T


def _softplus_jet(jet: Jet, beta: float) -> Jet:
    x, dx = jet
    return F.softplus(x, beta=beta), dx * torch.sigmoid(beta * x)[:, None]


def _block_jet(block: ResidualBlock, jet: Jet) -> Jet:
    h = _linear_jet(block.lin2, _softplus_jet(_linear_jet(block.lin1, jet), block.beta))
    return _softplus_jet((h[0] + jet[0], h[1] + jet[1]), block.beta)


class _PNTFields(nn.Module):
    def __init__(self, model: PNTFieldsModel) -> None:
        super().__init__()
        self.model = model
        self.dim = model.fourier.b.shape[1]

    def split(self, e_g: Tensor) -> tuple[Tensor, Tensor]:
        """q_g (1, dim) and f(q_g) (1, hidden) from e_g = [q_g, f(q_g)]."""
        return e_g[:, : self.dim], e_g[:, self.dim :]

    def encode_jet(self, q: Tensor) -> Jet:
        """f(q) (B, hidden) and its derivatives along each coordinate of q (B, dim, hidden)."""
        m = self.model
        w = 2 * torch.pi * m.fourier.b.T  # (dim, n)
        p = (q @ w)[:, None]
        sin, cos = p.sin(), p.cos()
        jet: Jet = (torch.cat([sin, cos], -1)[:, 0], torch.cat([w * cos, -w * sin], -1))
        jet = _softplus_jet(_linear_jet(m.encoder_in, jet), m.beta)
        for block in cast(list[ResidualBlock], list(m.encoder_blocks)):
            jet = _block_jet(block, jet)
        return _linear_jet(m.encoder_out, jet)

    def tau_jet(self, jet: Jet) -> Jet:
        """τ (B,) and its derivatives (B, k) from the generator input's jet: [smooth max, smooth min] (B, 2·hidden)
        and its k derivatives."""
        m = self.model
        for block in cast(list[ResidualBlock], list(m.generator_blocks)):
            jet = _block_jet(block, jet)
        y, dy = _linear_jet(m.generator_out, _softplus_jet(_linear_jet(m.generator_hidden, jet), m.beta))
        tau = torch.sigmoid(0.1 * y)
        return tau[:, 0], (dy * (0.1 * tau * (1 - tau))[:, None])[..., 0]

    def smooth_max_min(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """The generator input [smooth max, smooth min] of the encodings a, b (B or 1, hidden), and
        s = ∂max/∂a = ∂min/∂b = sigmoid(β(a - b)) (so ∂max/∂b = ∂min/∂a = 1 - s)."""
        beta = self.model.beta
        f = torch.stack(torch.broadcast_tensors(a, b))
        x = torch.cat([torch.logsumexp(beta * f, 0), -torch.logsumexp(-beta * f, 0)], dim=-1) / beta
        return x, torch.sigmoid(beta * (a - b))


def _travel_time_grads(diff: Tensor, tau: Tensor, dtau: Tensor) -> tuple[Tensor, Tensor]:
    """T = ‖diff‖ / τ with diff = q_g - q_s, and ∂T/∂q_s = -diff / (‖diff‖·τ) - (T / τ)·∂τ/∂q_s for dtau = ∂τ/∂q_s.
    At diff = 0 the norm's derivative is taken as 0, as autograd does."""
    r = diff.norm(dim=-1)
    t = r / tau
    unit = diff / torch.where(r > 0, r, 1.0)[:, None]
    return t, -unit / tau[:, None] - (t / tau)[:, None] * dtau


class EncodeGoal(_PNTFields):
    """q_g (1, dim) -> [q_g, f(q_g)] (1, dim + hidden)."""

    def forward(self, q_g: Tensor) -> Tensor:
        return torch.cat([q_g, self.model.encode(q_g)], dim=-1)


class GoalTravelTime(_PNTFields):
    """q (B, dim), e_g (1, dim + hidden) -> T (B,)."""

    def forward(self, q: Tensor, e_g: Tensor) -> Tensor:
        q_g, f_g = self.split(e_g)
        return (q_g - q).norm(dim=-1) / self.model.tau_from_encodings(self.model.encode(q), f_g)


class PairTravelTime(_PNTFields):
    """q_s, q_g (B, dim) -> T (B,)."""

    def forward(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        f = self.model.encode(torch.cat([q_s, q_g]))
        f_s, f_g = f[: q_s.shape[0]], f[q_s.shape[0] :]
        return (q_g - q_s).norm(dim=-1) / self.model.tau_from_encodings(f_s, f_g)


class GoalTravelTimeGrad(_PNTFields):
    """q (B, dim), e_g (1, dim + hidden) -> T (B,), ∇_q T (B, dim)."""

    def forward(self, q: Tensor, e_g: Tensor) -> tuple[Tensor, Tensor]:
        q_g, f_g = self.split(e_g)
        a, da = self.encode_jet(q)
        x, s = self.smooth_max_min(a, f_g)
        tau, dtau = self.tau_jet((x, torch.cat([da * s[:, None], da * (1 - s)[:, None]], -1)))
        return _travel_time_grads(q_g - q, tau, dtau)


class PairTravelTimeGrad(_PNTFields):
    """q_s, q_g (B, dim) -> T (B,), ∇_{q_s} T (B, dim), ∇_{q_g} T (B, dim)."""

    def forward(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a, da = self.encode_jet(q_s)
        b, db = self.encode_jet(q_g)
        x, s = self.smooth_max_min(a, b)
        s, c = s[:, None], 1 - s[:, None]
        # tangents along q_s (rows :dim) and q_g (rows dim:): max moves with s·da and c·db, min with c·da and s·db
        dmax, dmin = torch.cat([da * s, db * c], 1), torch.cat([da * c, db * s], 1)
        tau, dtau = self.tau_jet((x, torch.cat([dmax, dmin], -1)))
        t, dt_dqs = _travel_time_grads(q_g - q_s, tau, dtau[:, : self.dim])
        # T is symmetric in its ends' roles: ∂T/∂q_g = diff / (‖diff‖·τ) - (T / τ)·∂τ/∂q_g
        _, dt_dqg = _travel_time_grads(q_s - q_g, tau, dtau[:, self.dim :])
        return t, dt_dqs, dt_dqg
