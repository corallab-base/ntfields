from dataclasses import dataclass

from torch import Tensor


def ease(s: Tensor) -> Tensor:
    """TD-NTFields' easing of a linear speed s in [0, 1]: s²(2 - s)²."""
    return s**2 * (2 - s) ** 2


@dataclass
class ClippedEased:
    """TD-NTFields: s = clamp(d, dmin, dmax) / dmax, then S* = s²(2 - s)²."""

    dmin: float = 0.01
    dmax: float = 0.1

    def __call__(self, d: Tensor) -> Tensor:
        return ease(d.clamp(self.dmin, self.dmax) / self.dmax)
