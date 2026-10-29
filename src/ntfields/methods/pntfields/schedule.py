from dataclasses import dataclass

from ntfields.base import TrainState
from ntfields.train import Callback


@dataclass
class AlphaSchedule:
    """P-NTFields' progressive speed: S*_α = (1 - α) + α·S*, with α a function of the (0-indexed) epoch.

    Reproduces the reference loop: α = clip(start + (1 - start)·(u - delay)/period, start, end), where u sums
    1/(⌊k/period⌋ + 1) over epochs k = 1..epoch, so α rises more slowly after each `period` epochs. The defaults
    hold α = 0.5 until epoch 2000 and reach 1.05 at epoch ≈9200.
    """

    start: float = 0.5
    end: float = 1.05
    delay: int = 2000
    period: int = 4000

    def __call__(self, epoch: int) -> float:
        u, k = 0.0, 1
        while k <= epoch:
            j = k // self.period
            last = min(epoch, (j + 1) * self.period - 1)
            u += (last - k + 1) / (j + 1)
            k = last + 1
        alpha = self.start + (1 - self.start) * (u - self.delay) / self.period
        return min(max(alpha, self.start), self.end)


class LRRamp(Callback):
    """lr = clip(lr_max·(1 - (e - start)/length), lr_min, lr_max) at each epoch start (e is 1-indexed), as the
    reference sets it."""

    def __init__(self, lr_max: float, lr_min: float, start: int, length: int) -> None:
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.start = start
        self.length = length

    def lr(self, epoch: int) -> float:
        return min(max(self.lr_max * (1 - (epoch + 1 - self.start) / self.length), self.lr_min), self.lr_max)

    def on_epoch_start(self, state: TrainState) -> None:
        for group in state.optimizer.param_groups:
            group["lr"] = self.lr(state.epoch)
