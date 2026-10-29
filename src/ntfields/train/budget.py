from dataclasses import dataclass
from typing import Protocol


class Budget(Protocol):
    def exhausted(self, epochs: int, seconds: float) -> bool: ...


@dataclass
class Epochs:
    n: int

    def exhausted(self, epochs: int, seconds: float) -> bool:
        return epochs >= self.n


@dataclass
class TimeLimit:
    seconds: float

    def exhausted(self, epochs: int, seconds: float) -> bool:
        return seconds >= self.seconds
