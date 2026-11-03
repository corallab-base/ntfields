"""Inference-only form of TDMetricModel.

Training-time indirections are removed so the graph is plain matmuls and pointwise ops, which every backend
(torch.compile, TensorRT, a hand-written kernel) handles well:
- each LipschitzLinear's row rescale is folded into a constant weight
- GatedBlock's duplicated sin(lin1(x)) is computed once
- the block gates' sigmoid(0.1 · gate) are folded into constants
- instance norm is written out (biased variance, eps 1e-5, no affine), as F.instance_norm computes it

Outputs have the inputs' dtype whatever the weights': with float16 weights (`.half()`) and float32 inputs, the
matmuls and sines run in float16 and the instance norm and travel-time head in float32.

Gradients are forward mode. The configuration has few dimensions (6 for an arm), so the embedding's full Jacobian
is pushed forward alongside the value: every activation is carried as a stack (1 + dim, B, F), row 0 the primal and
rows 1.. its derivative along each input coordinate. A dense layer is then one matmul over all 1 + dim rows, so the
launch count matches the plain forward and only the matmul height grows.
"""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ntfields.inference import InferenceModel
from ntfields.methods.td_ntfields.model import GatedBlock, LipschitzLinear, TDMetricModel, metric_travel_time

INSTANCE_NORM_EPS = 1e-5


def fold(model: TDMetricModel) -> InferenceModel:
    embed = FoldedEmbedding(model)
    return InferenceModel(
        encode_goal=embed,
        travel_time=GoalTravelTime(embed, model),
        pair_travel_time=PairTravelTime(embed, model),
        gradient=GoalTravelTimeGrad(embed, model),
        pair_gradient=PairTravelTimeGrad(embed, model),
    )


def fold_linear(layer: LipschitzLinear) -> nn.Linear:
    with torch.no_grad():
        row_norm = layer.weight.pow(2).sum(dim=1).sqrt()
        scale = 1 + 1e-5 - F.softplus(1 - 1 / row_norm, beta=layer.beta)
        out = nn.Linear(layer.in_features, layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype)
        out.weight.copy_(layer.weight * scale[:, None])
        out.bias.copy_(layer.bias)
    return out.requires_grad_(False)


class FoldedEmbedding(nn.Module):
    """e(q) of a TDMetricModel, with every constant folded: q (B, dim) -> e (B, hidden)."""

    b: Tensor
    mix: Tensor

    def __init__(self, model: TDMetricModel) -> None:
        super().__init__()
        self.register_buffer("b", 2 * math.pi * model.fourier.b.detach().clone())
        self.gate_u = fold_linear(model.gate_u)
        self.gate_v = fold_linear(model.gate_v)
        blocks = cast(list[GatedBlock], list(model.blocks))
        self.lin1 = nn.ModuleList(fold_linear(b.lin1) for b in blocks)
        self.lin2 = nn.ModuleList(fold_linear(b.lin2) for b in blocks)
        self.lin3 = nn.ModuleList(fold_linear(b.lin3) for b in blocks)
        self.register_buffer("mix", torch.stack([torch.sigmoid(0.1 * g.detach()) for g in model.block_gates]))
        self.final = fold_linear(model.final)

    def blocks(self) -> list[tuple[nn.Linear, nn.Linear, nn.Linear]]:
        return cast(
            list[tuple[nn.Linear, nn.Linear, nn.Linear]], list(zip(self.lin1, self.lin2, self.lin3, strict=True))
        )

    def forward(self, q: Tensor) -> Tensor:
        proj = q.to(self.b.dtype) @ self.b.T
        x = torch.cat([proj.sin(), proj.cos()], dim=-1)
        u = self.gate_u(x).sin()
        v = self.gate_v(x).sin()
        uv = u - v  # u·s + v·(1 - s) = v + (u - v)·s
        for i, (lin1, lin2, lin3) in enumerate(self.blocks()):
            x_in = x
            x = v + uv * lin1(x).sin()
            x = v + uv * lin2(x).sin()
            w = self.mix[i]
            x = (1.0 - w) * x_in + w * lin3(x).sin()
        y = self.final(x).to(q.dtype)
        mean = y.mean(dim=-1, keepdim=True)
        var = (y - mean).pow(2).mean(dim=-1, keepdim=True)
        return (y - mean) * torch.rsqrt(var + INSTANCE_NORM_EPS)

    def stacked(self, q: Tensor) -> Tensor:
        """The embedding and its tangents along each coordinate of q, as one (1 + dim, B, hidden) stack."""
        proj = q.to(self.b.dtype) @ self.b.T
        dproj = self.b.T[:, None, :].expand(-1, q.shape[0], -1)  # d proj / d q_k = row k of b.T
        c, s = proj.cos(), proj.sin()
        x = torch.cat([torch.cat([s, c], dim=-1)[None], torch.cat([c * dproj, -s * dproj], dim=-1)])
        u = _sin(_linear(x, self.gate_u))
        v = _sin(_linear(x, self.gate_v))
        uv = u - v
        for i, (lin1, lin2, lin3) in enumerate(self.blocks()):
            x_in = x
            x = v + _mul(uv, _sin(_linear(x, lin1)))
            x = v + _mul(uv, _sin(_linear(x, lin2)))
            w = self.mix[i]
            x = (1.0 - w) * x_in + w * _sin(_linear(x, lin3))
        y = _linear(x, self.final).to(q.dtype)
        # instance norm over the hidden axis: e = c·r with c = y - mean(y), r = rsqrt(mean(c²) + eps)
        c = y - y.mean(dim=-1, keepdim=True)
        r = torch.rsqrt((c[0] ** 2).mean(dim=-1, keepdim=True) + INSTANCE_NORM_EPS)
        dvar = 2.0 * (c[:1] * c[1:]).mean(dim=-1, keepdim=True)
        return torch.cat([c[:1] * r, c[1:] * r - 0.5 * r**3 * dvar * c[:1]])


def _linear(s: Tensor, layer: nn.Linear) -> Tensor:
    z = s @ layer.weight.T
    return torch.cat([z[:1] + layer.bias, z[1:]])


def _sin(s: Tensor) -> Tensor:
    p = s[:1]
    return torch.cat([p.sin(), p.cos() * s[1:]])


def _mul(a: Tensor, b: Tensor) -> Tensor:
    return torch.cat([a[:1] * b[:1], a[1:] * b[:1] + a[:1] * b[1:]])


class _Metric(nn.Module):
    """The travel-time head on a folded embedding."""

    def __init__(self, embed: FoldedEmbedding, model: TDMetricModel) -> None:
        super().__init__()
        self.embed = embed
        self.group = model.group
        self.temperature = model.temperature
        self.out_scale = model.out_scale

    def head(self, e_s: Tensor, e_g: Tensor) -> Tensor:
        return metric_travel_time(e_s, e_g, self.group, self.temperature, self.out_scale)

    def head_grad(self, diff: Tensor) -> tuple[Tensor, Tensor]:
        """T and ∂T/∂e_s from diff = e_s - e_g (B, hidden)."""
        d = (diff**2 + 1e-6).sqrt()
        dg = d.view(diff.shape[0], -1, self.group)
        smooth_max = (torch.logsumexp(self.temperature * dg, dim=-1) - math.log(self.group)) / self.temperature
        # ∂T/∂e_s = out_scale · softmax_group(t·d) · (e_s - e_g) / d
        weight = torch.softmax(self.temperature * dg, dim=-1).view_as(d)
        return self.out_scale * smooth_max.sum(dim=-1), self.out_scale * weight * diff / d


class GoalTravelTime(_Metric):
    """q (B, dim), e_g (1, hidden) -> T (B,)."""

    def forward(self, q: Tensor, e_g: Tensor) -> Tensor:
        return self.head(self.embed(q), e_g)


class GoalTravelTimeGrad(_Metric):
    """q (B, dim), e_g (1, hidden) -> T (B,), ∇_q T (B, dim)."""

    def forward(self, q: Tensor, e_g: Tensor) -> tuple[Tensor, Tensor]:
        s = self.embed.stacked(q)
        t, dt_de = self.head_grad(s[0] - e_g)
        return t, (s[1:] * dt_de).sum(dim=-1).T


class PairTravelTime(_Metric):
    """q_s, q_g (B, dim) -> T (B,)."""

    def forward(self, q_s: Tensor, q_g: Tensor) -> Tensor:
        e = self.embed(torch.cat([q_s, q_g]))
        return self.head(e[: q_s.shape[0]], e[q_s.shape[0] :])


class PairTravelTimeGrad(_Metric):
    """q_s, q_g (B, dim) -> T (B,), ∇_{q_s} T (B, dim), ∇_{q_g} T (B, dim)."""

    def forward(self, q_s: Tensor, q_g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        s = self.embed.stacked(torch.cat([q_s, q_g]))
        ss, sg = s[:, : q_s.shape[0]], s[:, q_s.shape[0] :]
        t, dt_de = self.head_grad(ss[0] - sg[0])
        return t, (ss[1:] * dt_de).sum(dim=-1).T, -(sg[1:] * dt_de).sum(dim=-1).T
