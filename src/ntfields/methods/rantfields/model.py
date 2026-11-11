import torch
from torch import Tensor, nn

from ntfields.base import SceneCondition, TravelTimeModel
from ntfields.methods.rantfields.iqe import IQE


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.act = act
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), act, nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    """One projector with tied weights for the anchor, the query and their difference."""

    def __init__(self, dim: int, embed: int, act: nn.Module) -> None:
        super().__init__()
        self.coord_proj = nn.Sequential(nn.Linear(dim, 64), act, nn.Linear(64, embed), nn.LayerNorm(embed))

    def forward(self, anchor: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.coord_proj(anchor), self.coord_proj(query), self.coord_proj(query - anchor)


class ResidualCore(nn.Module):
    """softplus(ResNet([e_diff, e_anchor, e_query])) ≥ 0."""

    def __init__(self, embed: int, hidden: int, n_blocks: int, act: nn.Module) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(embed * 3, hidden), nn.LayerNorm(hidden), act)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, act) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, anchor: Tensor, query: Tensor, diff: Tensor) -> Tensor:
        h = self.proj(torch.cat([diff, anchor, query], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class FinslerCore(nn.Module):
    """d_IQE(e_anchor, e_query) · m, with m ≥ 0 a residual-network metric coefficient."""

    def __init__(self, embed: int, hidden: int, n_blocks: int, act: nn.Module) -> None:
        super().__init__()
        self.residual = ResidualCore(embed, hidden, n_blocks, act)
        self.d_iqe = IQE(embed, dim_per_component=16)

    def forward(self, anchor: Tensor, query: Tensor, diff: Tensor) -> Tensor:
        return self.d_iqe(anchor, query).unsqueeze(-1) * self.residual(anchor, query, diff)


class RaNTFieldsModel(TravelTimeModel):
    """T(q_s → q_g) = α·FinslerCore + (1 - α)·ResidualCore over a shared encoder, α = sigmoid(a) a learned scalar.

    Inside, the goal is the anchor (T = 0 at q_s = q_g) and the start the query, as in the reference, whose
    `start_pos` is our q_g and whose `goal_pts` is our q_s.
    """

    def __init__(self, dim: int, embed: int, hidden: int, n_residual_blocks: int, n_finsler_blocks: int) -> None:
        super().__init__()
        act = nn.Tanh()
        self.encoder = Encoder(dim, embed, act)
        self._residual = ResidualCore(embed, hidden, n_residual_blocks, act)
        self._finsler = FinslerCore(embed, hidden, n_finsler_blocks, act)
        self._alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition | None = None) -> Tensor:
        e = self.encoder(q_g, q_s)
        alpha = torch.sigmoid(self._alpha)
        return (alpha * self._finsler(*e) + (1 - alpha) * self._residual(*e)).squeeze(-1)
