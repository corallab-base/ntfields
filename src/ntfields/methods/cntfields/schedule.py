from dataclasses import dataclass


@dataclass
class LinearAlpha:
    """The reference's speed curriculum, S*_α = (1 - α) + α·S*, with α = min(start + step·epoch, end)."""

    start: float = 0.5
    step: float = 1e-4
    end: float = 0.95

    def __call__(self, epoch: int) -> float:
        return min(self.start + epoch * self.step, self.end)
