from dataclasses import dataclass

from torch import Tensor


@dataclass
class Clipped:
    """S* = clamp(d, dmin, dmax) / dmax."""

    dmin: float = 0.01
    dmax: float = 0.1

    def __call__(self, d: Tensor) -> Tensor:
        return d.clamp(self.dmin, self.dmax) / self.dmax
