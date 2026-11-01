from collections.abc import Callable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult, gradient_bidirectional
from ntfields.plan.mppi import mppi

if TYPE_CHECKING:
    from ntfields.field import TrainedField

Planner = Callable[["TrainedField", Tensor, Tensor], PlanResult]

PLANNERS: dict[str, Planner] = {"gradient_bidirectional": gradient_bidirectional, "mppi": mppi}


def register_planner(name: str, planner: Planner) -> None:
    PLANNERS[name] = planner


def get_planner(name: str) -> Planner:
    """A built-in planner, or one registered through the `ntfields.planners` entry-point group."""
    for ep in entry_points(group="ntfields.planners"):
        PLANNERS.setdefault(ep.name, ep.load())
    if name not in PLANNERS:
        raise KeyError(f"unknown planner {name!r}; available: {sorted(PLANNERS)}")
    return PLANNERS[name]


__all__ = ["PLANNERS", "PlanResult", "Planner", "get_planner", "gradient_bidirectional", "mppi", "register_planner"]
