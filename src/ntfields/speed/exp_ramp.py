import itertools
import math
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from ntfields.base import Environment


@dataclass
class ExpRamp:
    """RaNTFields' S* (OUM's speed field): S* = clip(exp(min(d/d_max, 1)) - 1, 0, 0.2)/0.2, at least `min_speed` in free
    space, `inside` at or inside an obstacle, and 1 where there are no obstacles (d = inf).

    `d_max` normalizes the clearance; `for_env` takes it as the largest clearance over the corners and the center of
    the bounds, as the reference generator does before it sees samples.
    """

    d_max: float
    min_speed: float = 0.1
    inside: float = 1e-4

    @classmethod
    def for_env(cls, env: Environment, min_speed: float = 0.1) -> Self:
        b = env.bounds.tolist()
        candidates = torch.tensor([*itertools.product(*b), [(lo + hi) / 2 for lo, hi in b]], dtype=torch.float32)
        d = env.distance(candidates)
        return cls(max(float(d.max()), 1e-8), min_speed)

    def __call__(self, d: Tensor) -> Tensor:
        s = ((d / self.d_max).clamp(max=1.0).exp() - 1).clamp(0.0, 0.2) / 0.2
        s = torch.where(d <= 0, torch.full_like(s, self.inside), s.clamp(min=self.min_speed))
        if math.isinf(self.d_max):
            return torch.ones_like(s)
        return torch.where(d.isinf(), torch.ones_like(s), s)
