from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult, gradient_bidirectional
from ntfields.plan.mppi import mppi

if TYPE_CHECKING:
    from ntfields.field import TrainedField

Planner = Callable[["TrainedField", Tensor, Tensor], PlanResult]

PLANNERS: dict[str, Planner] = {"gradient_bidirectional": gradient_bidirectional, "mppi": mppi}

__all__ = ["PLANNERS", "PlanResult", "Planner", "gradient_bidirectional", "mppi"]
