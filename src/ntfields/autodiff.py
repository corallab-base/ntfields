import torch
from torch import Tensor


def grad(y: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
    """Gradient of per-sample scalars y (B,) w.r.t. x (B, dim)."""
    (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    return g


def grad_and_hessian_diag(y: Tensor, x: Tensor, create_graph: bool = True) -> tuple[Tensor, Tensor]:
    """∇y and the Hessian diagonal ∂²y/∂x_i² (both (B, dim)) of per-sample scalars y (B,); sum the diagonal over a
    block of coordinates for its Laplacian. One double-backward per coordinate."""
    g = grad(y, x, create_graph=True)
    diag = [
        torch.autograd.grad(g[:, i].sum(), x, create_graph=create_graph, retain_graph=True)[0][:, i]
        for i in range(x.shape[1])
    ]
    return g, torch.stack(diag, dim=1)
