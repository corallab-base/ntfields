import copy
from dataclasses import asdict
from typing import Any

import torch

from ntfields.base import Batch, Method, TrainState, TravelTimeModel
from ntfields.checkpoint import config_from_dict
from ntfields.data import sample_pairs
from ntfields.envs import load_env
from ntfields.inference import autograd_inference
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
    q_g = batch.q_g[0]
    t_goal = model.travel_time_from_encoded_goal(batch.q_s, model.encode_goal(q_g, batch.scene), batch.scene)
    torch.testing.assert_close(
        t_goal,
        model(batch.q_s, q_g.expand_as(batch.q_s), batch.scene),
        msg="travel_time_from_encoded_goal must match forward",
    )
    _check_inference_model(model, batch)

    out = method.loss(model, batch, TrainState(model=model, optimizer=method.optimizer(model)))
    assert out.total.ndim == 0 and out.total.isfinite(), "loss must be a finite scalar"
    out.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), "loss has no gradient"


def _check_inference_model(model: TravelTimeModel, batch: Batch) -> None:
    """A model's inference form, if it has one, matches autograd through the model in the functions it provides (in
    float64, where folding and forward-mode differentiation only change the rounding)."""
    model = copy.deepcopy(model).double().eval()
    inference = model.inference_model(batch.scene)
    if inference is None:
        return
    reference = autograd_inference(model, batch.scene)
    q_s, q_g = batch.q_s.double(), batch.q_g.double()
    e_g = reference.encode_goal(q_g[:1])
    for name, args in (
        ("encode_goal", (q_g[:1],)),
        ("travel_time", (q_s, e_g)),
        ("gradient", (q_s, e_g)),
        ("pair_travel_time", (q_s, q_g)),
        ("pair_gradient", (q_s, q_g)),
    ):
        if getattr(inference, name) is None:
            continue
        with torch.no_grad():
            ours, expected = getattr(inference, name)(*args), getattr(reference, name)(*args)
        torch.testing.assert_close(ours, expected, msg=f"inference_model().{name} must match autograd")
