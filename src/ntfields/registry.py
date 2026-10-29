from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

from ntfields.base import Method, MethodConfig

M = TypeVar("M", bound=type[Method[Any]])

# name -> (method class, default config); presets are extra names for the same class
_METHODS: dict[str, tuple[type[Method[Any]], Callable[[], MethodConfig]]] = {}


def register(cls: M) -> M:
    _METHODS[cls.name] = (cls, cls.Config)
    return cls


def register_preset(name: str, cls: type[Method[Any]], config: Callable[[], MethodConfig]) -> None:
    _METHODS[name] = (cls, config)


def available() -> list[str]:
    _load_plugins()
    return sorted(_METHODS)


def get_method(name: str) -> type[Method[Any]]:
    return _entry(name)[0]


def default_config(name: str) -> MethodConfig:
    return _entry(name)[1]()


def create(name: str, cfg: MethodConfig | None = None) -> Method[Any]:
    cls, config = _entry(name)
    return cls(cfg or config())


def _entry(name: str) -> tuple[type[Method[Any]], Callable[[], MethodConfig]]:
    _load_plugins()
    if name not in _METHODS:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_METHODS)}")
    return _METHODS[name]


def _load_plugins() -> None:
    import ntfields.methods  # noqa: F401  (registers built-ins)

    for ep in entry_points(group="ntfields.methods"):
        if ep.name not in _METHODS:
            register(ep.load())
