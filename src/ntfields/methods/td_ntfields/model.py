import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ntfields.base import SceneCondition, TravelTimeModel
from ntfields.nn import RandomFourierFeatures


class LipschitzLinear(nn.Linear):
    """Linear layer whose weight rows are softly rescaled to norm <= 1 (ntrl-demo's `lip_norm`)."""

    def __init__(self, in_features: int, out_features: int, beta: float = 10.0) -> None:
        super().__init__(in_features, out_features)
        self.beta = beta
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        row_norm = self.weight.pow(2).sum(dim=1).sqrt().detach()
        scale = 1 + 1e-5 - F.softplus(1 - 1 / row_norm, beta=self.beta)
        return input @ (self.weight * scale[:, None]).T + self.bias


class GatedBlock(nn.Module):
    """PirateNet-style block: two u/v-gated sine layers, then a residual mix with the block input, weighted by
    sigmoid(0.1 · gate)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin1 = LipschitzLinear(hidden, hidden)
        self.lin2 = LipschitzLinear(hidden, hidden)
        self.lin3 = LipschitzLinear(hidden, hidden)

    def forward(self, x: Tensor, u: Tensor, v: Tensor, gate: Tensor) -> Tensor:
        # lin1 and the sines are evaluated twice: same values, but this graph fixes the gradient rounding
        x_in = x
        x = u * torch.sin(self.lin1(x)) + v * (1.0 - torch.sin(self.lin1(x_in)))
        h = self.lin2(x)
        x = u * torch.sin(h) + v * (1.0 - torch.sin(h))
        w = torch.sigmoid(0.1 * gate)
        return (1.0 - w) * x_in + w * torch.sin(self.lin3(x))


class TDMetricModel(TravelTimeModel):
    """TD-NTFields metric embedding: T = out_scale · Σ_groups smoothmax_group |e(q_s) - e(q_g)|.

    e(q) is Fourier features -> u/v gates -> gated sine blocks -> Lipschitz projection -> per-row normalization.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 512,
        n_blocks: int = 2,
        group: int = 16,
        temperature: float = 10.0,
        out_scale: float = 0.1,
        fourier_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.group = group
        self.temperature = temperature
        self.out_scale = out_scale
        self.fourier = RandomFourierFeatures(dim, hidden // 2, fourier_scale)
        self.gate_u = LipschitzLinear(hidden, hidden)
        self.gate_v = LipschitzLinear(hidden, hidden)
        self.blocks = nn.ModuleList(GatedBlock(hidden) for _ in range(n_blocks))
        # registered after the blocks: gradient clipping sums the norms in parameter order
        self.block_gates = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in range(n_blocks))
        self.final = LipschitzLinear(hidden, hidden)

    def embed(self, q: Tensor) -> Tensor:
        x = self.fourier(q)
        u = self.gate_u(x).sin()
        v = self.gate_v(x).sin()
        for block, gate in zip(self.blocks, self.block_gates, strict=True):
            x = block(x, u, v, gate)
        # InstanceNorm1d on a 2D tensor, as in the reference: each row normalized over its features
        return F.instance_norm(self.final(x)[None])[0]

    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition) -> Tensor:
        e_s, e_g = self.embed(q_s), self.embed(q_g)
        d = ((e_s - e_g) ** 2 + 1e-6).sqrt().view(q_s.shape[0], -1, self.group)
        t = self.temperature
        smooth_max = (torch.logsumexp(t * d, dim=-1) - math.log(self.group)) / t
        return self.out_scale * smooth_max.sum(dim=-1)
