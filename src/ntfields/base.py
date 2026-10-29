from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from ntfields.data.pair_sampler import Sampler
    from ntfields.problem import Problem
    from ntfields.train.callbacks import Callback


@dataclass
class SceneCondition:
    scene_id: int = 0
    points: Tensor | None = None  # (N, dim) point cloud, for obstacle-conditioned methods


class Environment(Protocol):
    bounds: Tensor  # (dim, 2)

    @property
    def dim(self) -> int: ...
    def distance(self, q: Tensor) -> Tensor: ...  # (B, dim) -> (B,) clearance
    def is_valid(self, q: Tensor) -> Tensor: ...  # (B, dim) -> (B,) bool
    def condition(self) -> SceneCondition: ...


class SpeedModel(Protocol):
    def __call__(self, d: Tensor) -> Tensor: ...  # clearance -> S*


@dataclass
class Batch:
    q_s: Tensor  # (B, dim)
    q_g: Tensor
    speed_s: Tensor  # (B,)
    speed_g: Tensor
    normal_s: Tensor  # (B, dim) unit clearance gradient ∇d/‖∇d‖ (0 where undefined)
    normal_g: Tensor
    scene: SceneCondition


@dataclass
class LossOut:
    total: Tensor
    terms: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainState:
    model: "TravelTimeModel"
    optimizer: torch.optim.Optimizer
    epoch: int = 0
    loss: float = float("nan")  # mean loss of the last epoch
    prev_loss: float = float("nan")  # mean loss of the last accepted epoch
    batches: int = 0  # batches in the last epoch
    loss_scale: float = 1.0  # the trainer backprops loss_scale * total
    terms: dict[str, float] = field(default_factory=dict)
    retry: bool = False  # set by a callback to re-run the epoch
    stop: bool = False


class TravelTimeModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition) -> Tensor: ...  # (B,)


@dataclass
class MethodConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None  # clip the global gradient norm before each step


C = TypeVar("C", bound=MethodConfig)


class Method(ABC, Generic[C]):
    name: ClassVar[str]
    Config: ClassVar[type[MethodConfig]]
    default_planner: ClassVar[str] = "gradient_bidirectional"

    def __init__(self, cfg: C) -> None:
        self.cfg = cfg

    @abstractmethod
    def build_model(self, problem: "Problem") -> TravelTimeModel: ...

    @abstractmethod
    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut: ...

    def callbacks(self) -> list["Callback"]:
        return []

    def default_sampler(self) -> "Sampler":
        from ntfields.data.pair_sampler import Uniform

        return Uniform()

    def optimizer(self, model: TravelTimeModel) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
