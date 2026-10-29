from dataclasses import asdict
from typing import Any

from ntfields.base import Method, TrainState
from ntfields.checkpoint import config_from_dict
from ntfields.data import sample_pairs
from ntfields.envs import load_env
from ntfields.problem import Problem
from ntfields.speed import Clipped


def check_method(method_cls: type[Method[Any]], problem: Problem | None = None) -> None:
    """Smoke checks every Method plugin should pass; raises AssertionError on failure."""
    problem = problem or Problem(load_env("box2d/one_sphere"), Clipped())
    cfg = method_cls.Config()
    assert config_from_dict(method_cls.Config, asdict(cfg)) == cfg, (
        "Config must round-trip through asdict (checkpoints)"
    )
    method = method_cls(cfg)
    model = method.build_model(problem)
    batch = next(sample_pairs(problem, 64).batches(64, problem.env.condition()))

    t = model(batch.q_s, batch.q_g, batch.scene)
    assert t.shape == (64,), f"travel time must have shape (B,), got {tuple(t.shape)}"
    t_self = model(batch.q_s, batch.q_s, batch.scene)
    assert t_self.abs().max() < 1e-2 * t.abs().mean() + 1e-6, "T(q, q) must be ~0"

    out = method.loss(model, batch, TrainState(model=model, optimizer=method.optimizer(model)))
    assert out.total.ndim == 0 and out.total.isfinite(), "loss must be a finite scalar"
    out.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), "loss has no gradient"
