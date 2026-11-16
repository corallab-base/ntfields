"""Values on a regular grid of nodes (array axis i = coordinate i), interpolated between them."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Interpolation = Literal["linear", "cubic"]


def interpolate(values: Tensor, bounds: Tensor, q: Tensor, interp: Interpolation) -> Tensor:
    """`values` (R_0, ..., R_{d-1}) on nodes spanning `bounds` (d, 2) end to end, at q (B, d) -> (B,) in q's dtype;
    clamped to the border outside `bounds`. Differentiable in q.

    - "linear": multilinear, whose gradient jumps at every cell edge;
    - "cubic": separable Catmull-Rom, which passes through the nodes and has a continuous gradient; it can overshoot
      slightly next to kinks.
    """
    values = values.to(q.device)
    lo, hi = bounds.to(q.device, torch.float64).unbind(-1)
    if interp == "linear":
        u = (2 * (q.double() - lo) / (hi - lo) - 1).flip(-1)  # grid_sample wants (x, y[, z]) = reversed axes
        dim = q.shape[-1]
        grid = u.reshape(1, -1, *[1] * (dim - 1), dim)
        out = F.grid_sample(values[None, None], grid, align_corners=True, padding_mode="border").reshape(-1)
    else:
        out = _catmull_rom(values, (q.double() - lo) / (hi - lo))
    return out.to(q.dtype)


def _catmull_rom(values: Tensor, u: Tensor) -> Tensor:
    """Separable Catmull-Rom interpolation of `values` (R_0, ..., R_{d-1}) at u (B, d) in [0, 1]^d (clamped), with
    the edge nodes repeated past the border: 4^d taps per query, gathered at once."""
    values = values.contiguous()
    device, dim = u.device, values.dim()
    size = torch.tensor(values.shape, device=device)
    x = u.clamp(0, 1) * (size - 1)
    i0 = x.detach().floor().clamp(max=size - 2).long()
    t = x - i0
    t2, t3 = t * t, t * t * t
    # per axis: the 4 nodes i0-1..i0+2 and their weights, (B, d, 4)
    nodes = (i0[..., None] + torch.arange(-1, 3, device=device)).clamp(min=0).minimum(size[:, None] - 1)
    weights = torch.stack(
        ((-t + 2 * t2 - t3) / 2, (2 - 5 * t2 + 3 * t3) / 2, (t + 4 * t2 - 3 * t3) / 2, (t3 - t2) / 2), -1
    )
    # every combination of one node per axis, (B, 4^d, d)
    taps = torch.cartesian_prod(*[torch.arange(4, device=device)] * dim).reshape(-1, dim)
    axes = torch.arange(dim, device=device)
    flat_index = (nodes[:, axes, taps] * torch.tensor(values.stride(), device=device)).sum(-1)
    return (weights[:, axes, taps].prod(-1) * values.view(-1)[flat_index]).sum(-1)
