from dataclasses import replace
from typing import cast

import pytest
import torch

from ntfields import Problem, TrainedField
from ntfields.accel import accelerate
from ntfields.envs import load_env
from ntfields.inference import autograd_inference
from ntfields.methods.td_ntfields.inference import fold
from ntfields.methods.td_ntfields.model import TDMetricModel
from ntfields.plan import mppi
from ntfields.registry import create
from ntfields.speed import Clipped

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _td_field(dim: int = 2, dtype: torch.dtype = torch.float64, device: str = "cpu") -> TrainedField:
    """A small TD-NTFields field with nonzero block gates, so every folded constant matters."""
    torch.manual_seed(0)
    method = create("td_ntfields")
    problem = Problem(load_env("box2d/empty"), Clipped())
    model = cast(TDMetricModel, method.build_model(problem))
    with torch.no_grad():
        for gate in model.block_gates:
            gate.normal_()
    return TrainedField(method, problem, model.to(dtype=dtype, device=device).eval())


def _pntfields_field() -> TrainedField:
    torch.manual_seed(0)
    method = create("pntfields")
    problem = Problem(load_env("box2d/empty"), Clipped())
    return TrainedField(method, problem, method.build_model(problem).double().eval())


@pytest.mark.parametrize("method", ["td_ntfields", "pntfields"])
@pytest.mark.parametrize("name", ["encode_goal", "travel_time", "gradient", "pair_travel_time", "pair_gradient"])
def test_inference_model_matches_autograd(method: str, name: str):
    field = _td_field() if method == "td_ntfields" else _pntfields_field()
    scene = field.problem.env.condition()
    folded, reference = field.model.inference_model(scene), autograd_inference(field.model, scene)
    assert folded is not None
    q, y = torch.rand(32, 2, dtype=torch.float64) - 0.5, torch.rand(32, 2, dtype=torch.float64) - 0.5
    e_g = reference.encode_goal(y[:1])
    args = {"encode_goal": (y[:1],), "travel_time": (q, e_g), "gradient": (q, e_g)}.get(name, (q, y))
    with torch.no_grad():
        torch.testing.assert_close(getattr(folded, name)(*args), getattr(reference, name)(*args))


def test_pntfields_pair_gradient_at_coincident_ends():
    """‖q_g - q_s‖ has no derivative at q_s = q_g; the forward-mode gradients take it as 0, as autograd does."""
    field = _pntfields_field()
    scene = field.problem.env.condition()
    folded, reference = field.model.inference_model(scene), autograd_inference(field.model, scene)
    assert folded is not None and folded.pair_gradient is not None and reference.pair_gradient is not None
    q = torch.rand(8, 2, dtype=torch.float64) - 0.5
    for ours, expected in zip(folded.pair_gradient(q, q), reference.pair_gradient(q, q), strict=True):
        torch.testing.assert_close(ours, expected)


def test_accelerated_field_matches_field():
    field = _td_field()
    fast = accelerate(field, "eager")
    q, y = torch.rand(16, 2, dtype=torch.float64) - 0.5, torch.rand(16, 2, dtype=torch.float64) - 0.5
    goal = y[0]
    torch.testing.assert_close(fast.travel_time(q, y), field.travel_time(q, y))
    for ours, expected in zip(fast.gradient(q, y), field.gradient(q, y), strict=True):
        torch.testing.assert_close(ours, expected)
    torch.testing.assert_close(fast.towards(goal).travel_time(q), field.towards(goal).travel_time(q))
    torch.testing.assert_close(fast.towards(goal).gradient(q), field.towards(goal).gradient(q))


def test_accelerated_field_plans_as_the_field():
    """Over a few steps: on an untrained field MPPI wanders, and rounding differences grow (1e-11 by step 20)."""
    field = _td_field()
    start, goal = torch.tensor([-0.6, -0.3], dtype=torch.float64), torch.tensor([0.5, 0.4], dtype=torch.float64)
    paths = [
        mppi(f, start, goal, max_steps=15, generator=torch.Generator().manual_seed(0)).path
        for f in (field, field.accelerate("eager"))
    ]
    torch.testing.assert_close(paths[0], paths[1])


def test_gradients_are_absent_without_forward_mode_ones(monkeypatch: pytest.MonkeyPatch):
    field = _td_field()
    values_only = replace(fold(cast(TDMetricModel, field.model)), gradient=None, pair_gradient=None)
    monkeypatch.setattr(field.model, "inference_model", lambda scene: values_only)
    fast = accelerate(field, "eager")
    start, goal = torch.tensor([-0.6, -0.3], dtype=torch.float64), torch.tensor([0.5, 0.4], dtype=torch.float64)
    mppi(fast, start, goal, max_steps=5)  # values only
    with pytest.raises(ValueError, match="no gradients"):
        fast.gradient(start[None], goal[None])
    with pytest.raises(ValueError, match="no gradients"):
        fast.towards(goal).gradient(start[None])
    with pytest.raises(ValueError, match="no gradients"):
        fast.plan(start, goal, "gradient_bidirectional")


def test_accelerate_needs_an_inference_model():
    problem = Problem(load_env("box2d/empty"), Clipped())
    method = create("euclidean")
    with pytest.raises(ValueError, match="no inference model"):
        accelerate(TrainedField(method, problem, method.build_model(problem)))
    with pytest.raises(ValueError, match="unknown backend"):
        accelerate(_td_field(), "onnx")  # pyright: ignore[reportArgumentType]


@requires_cuda
def test_compiled_cuda_graphs_match_eager():
    field = _td_field(dtype=torch.float32, device="cuda")
    eager, fast = field.accelerate("eager", cuda_graphs=False), field.accelerate("compile")
    q = torch.rand(100, 2, device="cuda") - 0.5
    goals = torch.rand(2, 2, device="cuda") - 0.5
    # one graph serves every goal, and outputs are not overwritten by later replays
    t = [fast.towards(g).travel_time(q) for g in goals]
    for g, t_g in zip(goals, t, strict=True):
        torch.testing.assert_close(t_g, eager.towards(g).travel_time(q), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fast.towards(g).gradient(q), eager.towards(g).gradient(q), rtol=1e-4, atol=1e-4)
    for ours, expected in zip(fast.gradient(q, q.flip(0)), eager.gradient(q, q.flip(0)), strict=True):
        torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)
