from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.speed.clipped_eased import ease


@dataclass
class GaussianBand:
    """CNTFields' speed around a constraint manifold: S* = exp(-d²/(λε²)) with d = ε - c, c being a `ConstrainedEnv`'s
    distance. The reference then eases it, S*²(2 - S*)²; `eased=False` is the paper's Eq. 5."""

    epsilon: float = 0.7
    lam: float = 0.1
    eased: bool = True

    def __call__(self, d: Tensor) -> Tensor:
        s = torch.exp(-((self.epsilon - d) ** 2) / (self.lam * self.epsilon**2))
        return ease(s) if self.eased else s
