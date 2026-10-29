from dataclasses import asdict, fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, get_args, get_type_hints

import torch

import ntfields
from ntfields.base import Method, TravelTimeModel
from ntfields.field import TrainedField
from ntfields.problem import Problem
from ntfields.registry import get_method


def save(path: Path, method: Method[Any], problem: Problem, model: TravelTimeModel, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "ntfields_version": ntfields.__version__,
        "method": method.name,
        "config": asdict(method.cfg),
        "env": _to_spec(problem.env),
        "speed": _to_spec(problem.speed),
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, path)


def load(path: str | Path, device: str | torch.device = "cpu") -> TrainedField:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    method_cls = get_method(ckpt["method"])
    method = method_cls(config_from_dict(method_cls.Config, ckpt["config"]))
    problem = Problem(_from_spec(ckpt["env"]), _from_spec(ckpt["speed"]))
    model = method.build_model(problem)
    model.load_state_dict(ckpt["state_dict"])
    return TrainedField(method, problem, model.to(device).eval(), ckpt_path=Path(path))


C = TypeVar("C")


def config_from_dict(cls: type[C], data: dict[str, Any]) -> C:
    """Inverse of `dataclasses.asdict`, rebuilding nested dataclass fields (for a union, the member whose field
    names match)."""
    hints = get_type_hints(cls)
    return cls(**{name: _rebuild(hints[name], value) for name, value in data.items()})


def _rebuild(hint: Any, value: Any) -> Any:
    if isinstance(value, dict):
        for t in get_args(hint) or (hint,):
            if isinstance(t, type) and is_dataclass(t) and {f.name for f in fields(t)} == value.keys():
                return config_from_dict(t, value)
    return value


def _to_spec(obj: object) -> dict[str, Any]:
    """Envs and speed models are saved as their dataclass fields plus an import path."""
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{type(obj).__name__} must be a dataclass to be checkpointed")
    cls = type(obj)
    return {
        "type": f"{cls.__module__}:{cls.__qualname__}",
        "fields": {f.name: getattr(obj, f.name) for f in fields(obj) if f.init},
    }


def _from_spec(spec: dict[str, Any]) -> Any:
    module, name = spec["type"].split(":")
    return getattr(import_module(module), name)(**spec["fields"])
