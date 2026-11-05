from collections.abc import Callable

import torch
from torch import Tensor

Free = Callable[[Tensor], Tensor]  # (N, dim) -> (N,) bool


def _edges_free(free: Free, a: Tensor, b: Tensor, resolution: float) -> Tensor:
    """(E,) bool: every point of each straight segment a[i]→b[i], at most `resolution` apart, is free."""
    steps = int(((b - a).norm(dim=-1).amax() / resolution).ceil()) + 1
    t = torch.linspace(0, 1, steps).to(a)[None, :, None]
    points = a[:, None] + t * (b - a)[:, None]
    return free(points.reshape(-1, a.shape[1])).view(len(a), steps).all(dim=1)


def _trace(nodes: Tensor, parents: list[int], i: int) -> Tensor:
    order = []
    while i >= 0:
        order.append(i)
        i = parents[i]
    return nodes[torch.tensor(order, device=nodes.device)]


def bidirectional_rrt(
    free: Free,
    start: Tensor,
    goal: Tensor,
    lo: Tensor,
    hi: Tensor,
    generator: torch.Generator,
    resolution: float,
    step: float = 0.08,
    reach: float = 0.4,
    width: int = 64,
    rounds: int = 300,
) -> Tensor | None:
    """A path start→goal (N, dim), or None. Two trees each grow by up to `width` edges per round, checked as one batch,
    and join by a straight edge of at most `reach`. A witness that a query is solvable, not a short path."""
    device = start.device
    if bool(_edges_free(free, start[None], goal[None], resolution)[0]):
        return torch.stack([start, goal])
    trees = [(start[None], [-1]), (goal[None], [-1])]
    for r in range(rounds):
        grow, other = trees[r % 2], trees[1 - r % 2]
        nodes, parents = grow
        aim = lo + (hi - lo) * torch.rand(width, len(lo), generator=generator).to(lo)
        near = torch.cdist(aim, nodes).argmin(dim=1)
        delta = aim - nodes[near]
        new = nodes[near] + delta * (step / delta.norm(dim=-1, keepdim=True).clamp_min(step))
        ok = _edges_free(free, nodes[near], new, resolution)
        if not ok.any():
            continue
        new, near = new[ok], near[ok]
        first = len(nodes)
        nodes = torch.cat([nodes, new])
        parents = parents + near.tolist()
        trees[r % 2] = (nodes, parents)
        distance, nearest = torch.cdist(new, other[0]).min(dim=1)
        candidates = (distance <= reach).nonzero().view(-1)
        if not len(candidates):
            continue
        candidates = candidates[distance[candidates].argsort()[:width]]
        joined = _edges_free(free, new[candidates], other[0][nearest[candidates]], resolution)
        if joined.any():
            k = candidates[joined.nonzero()[0, 0]]
            here = _trace(nodes, parents, first + int(k))
            there = _trace(other[0], other[1], int(nearest[k]))
            path = torch.cat([here.flip(0), there]) if r % 2 == 0 else torch.cat([there.flip(0), here])
            return path.to(device)
    return None
