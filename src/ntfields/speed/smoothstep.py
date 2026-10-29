from dataclasses import dataclass

from torch import Tensor


@dataclass
class Smoothstep:
    """S* = floor + (1 - floor) · smootherstep(d / dmax)."""

    dmax: float = 0.1
    floor: float = 0.1

    def __call__(self, d: Tensor) -> Tensor:
        t = (d / self.dmax).clamp(0.0, 1.0)
        return self.floor + (1.0 - self.floor) * t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
