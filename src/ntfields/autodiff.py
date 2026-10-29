import torch
from torch import Tensor


def grad(y: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
    """Gradient of per-sample scalars y (B,) w.r.t. x (B, dim)."""
    (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    return g
