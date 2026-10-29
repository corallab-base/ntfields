import torch
from torch import Tensor

from ntfields.field import TrainedField
from ntfields.gt import GridField
from ntfields.plan import PlanResult
from ntfields.problem import Problem


def path_length(path: Tensor) -> float:
    return float(path.diff(dim=0).norm(dim=-1).sum())


def path_cost(problem: Problem, path: Tensor) -> float:
    """Travel time ∫ ds / S* along the path (midpoint rule)."""
    seg = path.diff(dim=0)
    mid = path[:-1] + seg / 2
    return float((seg.norm(dim=-1) / problem.speed_at(mid)).sum())


def is_collision_free(problem: Problem, path: Tensor, resolution: float = 0.005) -> bool:
    """Checks every segment at roughly `resolution` spacing."""
    seg = path.diff(dim=0)
    n = max(1, int(seg.norm(dim=-1).max() / resolution) + 1)
    t = torch.linspace(0, 1, n + 1, dtype=path.dtype, device=path.device)
    dense = (path[:-1, None] + t[:, None] * seg[:, None]).reshape(-1, path.shape[-1])
    return bool(problem.env.is_valid(torch.cat([dense, path[-1:]])).all())


def plan_succeeded(problem: Problem, result: PlanResult) -> bool:
    """The planner reached the goal along a collision-free path."""
    return result.success and is_collision_free(problem, result.path)


def travel_time_error(field: TrainedField, gt: GridField, starts: Tensor) -> float:
    """Mean |T_field(start, goal) - T_gt(start)| over starts with a finite ground truth."""
    goals = gt.goal.to(starts).expand_as(starts)
    t_gt = gt.value(starts)
    finite = t_gt.isfinite()
    return float((field.travel_time(starts, goals) - t_gt).abs()[finite].mean())
