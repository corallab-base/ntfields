import time
from typing import Any

import torch
from torch import Tensor

from ntfields.envs.constraint import ConstrainedEnv
from ntfields.eval.metrics import path_length, plan_succeeded
from ntfields.field import TrainedField
from ntfields.plan import gradient_bidirectional


def manifold_margin(env: ConstrainedEnv, path: Tensor) -> tuple[float, float]:
    """Mean and largest d_M along a path, the paper's "margin"."""
    d = env.constraint.violation(path)
    return float(d.mean()), float(d.max())


def evaluate_constrained(field: TrainedField, starts: Tensor, goals: Tensor, **kwargs: Any) -> dict[str, float]:
    """The paper's metrics over query pairs for `gradient_bidirectional(**kwargs)`: success rate, and time, length and
    margin over the successes.

    Success means the fronts met along a path that stays valid: within the env's tolerance and clear of obstacles.
    """
    env = field.problem.env
    if not isinstance(env, ConstrainedEnv):
        raise TypeError(f"evaluate_constrained needs a ConstrainedEnv, got {type(env).__name__}")
    met, solved = 0, 0
    times, lengths, means, peaks = [], [], [], []
    for start, goal in zip(starts, goals, strict=True):
        begin = time.perf_counter()
        result = gradient_bidirectional(field, start, goal, **kwargs)
        elapsed = time.perf_counter() - begin
        met += result.success
        if plan_succeeded(field.problem, result):
            solved += 1
            times.append(elapsed)
            lengths.append(path_length(result.path))
            mean, peak = manifold_margin(env, result.path)
            means.append(mean)
            peaks.append(peak)

    def avg(xs: list[float]) -> float:
        return float(torch.tensor(xs).mean()) if xs else float("nan")

    n = len(starts)
    return {
        "pairs": n,
        "met": met / n,
        "success": solved / n,
        "time_s": avg(times),
        "length": avg(lengths),
        "margin_mean": avg(means),
        "margin_max": max(peaks, default=float("nan")),
    }
