import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ntfields.base import SceneCondition, TravelTimeModel
from ntfields.nn import RandomFourierFeatures


def _linear(in_features: int, out_features: int) -> nn.Linear:
    """Linear layer with the reference init: weights and biases U(±2/√fan_in)."""
    layer = nn.Linear(in_features, out_features)
    bound = 2 / math.sqrt(in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)
    return layer


# Forward-mode jets (x, dx, lx): a value (B, n), its derivatives (B, k, n) along k input coordinates, and the second
# derivatives along the same coordinates. They give τ's gradient and Hessian diagonal in one pass, as the reference's
# `out_laplace` does (3.6x faster than double backward on the arm).
Jet = tuple[Tensor, Tensor, Tensor]


def _linear_jet(layer: nn.Linear, jet: Jet) -> Jet:
    x, dx, lx = jet
    return layer(x), dx @ layer.weight.T, lx @ layer.weight.T


def _softplus_jet(jet: Jet, beta: float) -> Jet:
    x, dx, lx = jet
    s = torch.sigmoid(beta * x)[:, None]
    return F.softplus(x, beta=beta), dx * s, dx * dx * (beta * s * (1 - s)) + lx * s


class ResidualBlock(nn.Module):
    """act(lin2(act(lin1(x))) + x)."""

    def __init__(self, width: int, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.lin1 = _linear(width, width)
        self.lin2 = _linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        h = F.softplus(self.lin1(x), beta=self.beta)
        return F.softplus(self.lin2(h) + x, beta=self.beta)

    def jet(self, jet: Jet) -> Jet:
        h = _linear_jet(self.lin2, _softplus_jet(_linear_jet(self.lin1, jet), self.beta))
        return _softplus_jet((h[0] + jet[0], h[1] + jet[1], h[2] + jet[2]), self.beta)


def _blocks_jet(blocks: nn.ModuleList, jet: Jet) -> Jet:
    for block in blocks:
        jet = cast(ResidualBlock, block).jet(jet)
    return jet


class PNTFieldsModel(TravelTimeModel):
    """P-NTFields: T = ‖q_g - q_s‖ / τ, with τ = g(f(q_s) ⊗ f(q_g)) in (0, 1).

    f is Fourier features -> a softplus ResNet encoder; ⊗ is the logsumexp-smoothed [max, min]; g is a softplus
    ResNet generator ending in sigmoid(0.1·y).
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 128,
        encoder_blocks: int = 2,
        generator_blocks: int = 3,
        fourier_features: int = 128,
        fourier_scale: float = 0.5,
        beta: float = 10.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.fourier = RandomFourierFeatures(dim, fourier_features, fourier_scale)
        self.encoder_in = _linear(2 * fourier_features, hidden)
        self.encoder_blocks = nn.ModuleList(ResidualBlock(hidden, beta) for _ in range(encoder_blocks))
        self.encoder_out = _linear(hidden, hidden)
        self.generator_blocks = nn.ModuleList(ResidualBlock(2 * hidden, beta) for _ in range(generator_blocks))
        self.generator_hidden = _linear(2 * hidden, hidden)
        self.generator_out = _linear(hidden, 1)

    def encode(self, q: Tensor) -> Tensor:
        x = F.softplus(self.encoder_in(self.fourier(q)), beta=self.beta)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_out(x)

    def tau(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        """τ (B,), the factor with T = ‖q_g - q_s‖ / τ."""
        f = torch.stack([self.encode(q_s), self.encode(q_g)])
        x = torch.cat([torch.logsumexp(self.beta * f, 0), -torch.logsumexp(-self.beta * f, 0)], dim=-1) / self.beta
        for block in self.generator_blocks:
            x = block(x)
        x = F.softplus(self.generator_hidden(x), beta=self.beta)
        return torch.sigmoid(0.1 * self.generator_out(x)).squeeze(-1)

    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition) -> Tensor:
        return (q_g - q_s).norm(dim=-1) / self.tau(q_s, q_g)

    def tau_jet(self, q_s: Tensor, q_g: Tensor) -> Jet:
        """τ (B,), and its gradient and Hessian diagonal (B, 2·dim) w.r.t. [q_s, q_g]."""
        a, da, la = self._encode_jet(q_s)
        b, db, lb = self._encode_jet(q_g)
        beta = self.beta
        f = torch.stack([a, b])
        x = torch.cat([torch.logsumexp(beta * f, 0), -torch.logsumexp(-beta * f, 0)], dim=-1) / beta
        # smooth max m and min n of (a, b): ∂m/∂a = s0, ∂m/∂b = s1, ∂n/∂a = s1, ∂n/∂b = s0, with s0 = sigmoid(β(a - b))
        s0 = torch.sigmoid(beta * (a - b))[:, None]
        s1 = 1 - s0
        c = beta * s0 * s1
        dm, dn = torch.cat([da * s0, db * s1], 1), torch.cat([da * s1, db * s0], 1)
        lm = torch.cat([la * s0 + c * da * da, lb * s1 + c * db * db], 1)
        ln = torch.cat([la * s1 - c * da * da, lb * s0 - c * db * db], 1)
        jet: Jet = (x, torch.cat([dm, dn], -1), torch.cat([lm, ln], -1))
        jet = _blocks_jet(self.generator_blocks, jet)
        y, dy, ly = _linear_jet(self.generator_out, _softplus_jet(_linear_jet(self.generator_hidden, jet), beta))
        t = torch.sigmoid(0.1 * y)
        d1 = (0.1 * t * (1 - t))[:, None]
        d2 = d1 * 0.1 * (1 - 2 * t)[:, None]
        return t[:, 0], (dy * d1)[..., 0], (dy * dy * d2 + ly * d1)[..., 0]

    def _encode_jet(self, q: Tensor) -> Jet:
        w = 2 * math.pi * self.fourier.b.T  # (dim, n)
        p = (q @ w)[:, None]
        sin, cos = p.sin(), p.cos()
        jet: Jet = (
            torch.cat([sin, cos], -1)[:, 0],
            torch.cat([w * cos, -w * sin], -1),
            -torch.cat([w * w * sin, w * w * cos], -1),
        )
        jet = _softplus_jet(_linear_jet(self.encoder_in, jet), self.beta)
        jet = _blocks_jet(self.encoder_blocks, jet)
        return _linear_jet(self.encoder_out, jet)
