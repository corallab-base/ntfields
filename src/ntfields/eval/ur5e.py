from collections.abc import Sequence

import torch

from ntfields.envs.arm import ArmEnv, ntrl_random_pairs, ntrl_scenarios
from ntfields.field import TrainedField
from ntfields.plan import mppi


def benchmark_ur5e(field: TrainedField, seeds: Sequence[int] = (2024, 1, 2)) -> dict[str, list[dict[str, int]]]:
    """The UR5e evaluation: ntrl-demo's 5 scenarios and 50 random pairs, planned with MPPI once per
    seed. A plan succeeds if MPPI converges and every waypoint has a raw clearance (without the offset) above 0.

    Returns {"scenarios" | "random": [per-seed counts of "success", "converged", "collision_free", "n"]}.
    """
    env = field.problem.env
    assert isinstance(env, ArmEnv), "benchmark_ur5e needs the UR5e ArmEnv"
    device = next(field.model.parameters()).device
    results: dict[str, list[dict[str, int]]] = {}
    for name, (starts, goals) in (("scenarios", ntrl_scenarios()), ("random", ntrl_random_pairs())):
        results[name] = []
        for seed in seeds:
            torch.manual_seed(seed)
            counts = {"success": 0, "converged": 0, "collision_free": 0, "n": len(starts)}
            for start, goal in zip(starts.to(device), goals.to(device), strict=True):
                result = mppi(field, start, goal)
                free = bool((env.distance(result.path) + env.clearance_offset > 0).all())
                counts["converged"] += result.success
                counts["collision_free"] += free
                counts["success"] += result.success and free
            results[name].append(counts)
    return results
