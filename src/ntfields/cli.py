import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

try:
    import tyro
except ImportError as e:
    raise SystemExit("the ntfields CLI needs tyro: pip install 'ntfields[cli]'") from e

import ntfields
from ntfields.accel import Precision
from ntfields.base import MethodConfig, SpeedModel
from ntfields.envs import load_env
from ntfields.inference import Function
from ntfields.registry import available, default_config, get_method
from ntfields.speed import Clipped, ClippedEased, Smoothstep

C = TypeVar("C", bound=MethodConfig)


@dataclass
class Speed:
    kind: Literal["clipped", "clipped_eased", "smoothstep"] = "clipped"
    dmin: float = 0.01  # clipped, clipped_eased
    dmax: float = 0.1
    floor: float = 0.1  # smoothstep

    def build(self) -> SpeedModel:
        if self.kind == "smoothstep":
            return Smoothstep(self.dmax, self.floor)
        return (Clipped if self.kind == "clipped" else ClippedEased)(self.dmin, self.dmax)


@dataclass
class Train(Generic[C]):
    """Train a method on a preset environment."""

    method: C  # chosen by name; its config fields are --method.*
    env: str = "box2d/one_sphere"  # preset from ntfields.envs.PRESETS
    speed: Speed = field(default_factory=Speed)
    epochs: int = 2000
    train: ntfields.TrainConfig = field(default_factory=ntfields.TrainConfig)


@dataclass
class Export:
    """Export TensorRT engines of a trained field, for runtimes outside Python (needs the [tensorrt] extra)."""

    ckpt: Path
    out: Path
    functions: tuple[Function, ...] = ()  # default: every function the model's inference form has
    precision: tuple[Precision, ...] = ("fp32",)  # of travel_time and pair_travel_time; the others are fp32
    batch: tuple[int, int, int] = (1, 128, 65536)  # min, opt, max
    device: str = "cuda"


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = available()
    defaults = {name: default_config(name) for name in names}
    choice: Any = Union[  # noqa: UP007  (built at runtime from the registry)
        tuple(
            Annotated[type(cfg), tyro.conf.subcommand(name, default=cfg, prefix_name=False)]
            for name, cfg in defaults.items()
        )
    ]
    command: Train[Any] | Export = tyro.extras.subcommand_cli_from_dict(
        {"train": Train[choice], "export": Export}, args=args, config=(tyro.conf.CascadeSubcommandArgs,)
    )
    if isinstance(command, Export):
        from ntfields.accel.tensorrt import export

        field_ = ntfields.load(command.ckpt, device=command.device)
        export(field_, command.out, command.functions, command.precision, command.batch)
        print(f"engines: {command.out}")
        return
    name = next(n for n, cfg in defaults.items() if type(cfg) is type(command.method))
    method = get_method(name)(command.method)
    problem = ntfields.Problem(load_env(command.env), command.speed.build())
    trained = ntfields.train(method, problem, ntfields.Epochs(command.epochs), train_cfg=command.train)
    print(f"checkpoint: {trained.ckpt_path}")
