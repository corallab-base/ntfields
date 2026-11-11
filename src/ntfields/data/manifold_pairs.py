from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.envs.constraint import ConstrainedEnv
from ntfields.problem import Problem


@dataclass
class ManifoldPairs:
    """CNTFields: start and goal drawn independently near the constraint manifold. Needs a `ConstrainedEnv`."""

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        env = problem.env
        if not isinstance(env, ConstrainedEnv):
            raise TypeError(f"ManifoldPairs needs a ConstrainedEnv, got {type(env).__name__}")
        return env.constraint.sample(n, generator, device), env.constraint.sample(n, generator, device)
