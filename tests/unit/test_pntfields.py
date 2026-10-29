from dataclasses import asdict
from pathlib import Path

import torch

import ntfields
from ntfields import Batch, Problem, SceneCondition, TrainConfig, TrainState
from ntfields.autodiff import grad_and_hessian_diag
from ntfields.checkpoint import config_from_dict
from ntfields.envs import load_env
from ntfields.methods.pntfields import AlphaSchedule, LRRamp, PNTFields, PNTFieldsConfig
from ntfields.methods.pntfields.loss import pntfields_loss
from ntfields.methods.pntfields.model import PNTFieldsModel
from ntfields.registry import default_config
from ntfields.speed import Clipped
from ntfields.testing import check_method


def test_check_method():
    check_method(PNTFields)


def test_presets_are_registered_and_round_trip():
    assert default_config("pntfields_arm") == PNTFieldsConfig.arm()
    for name in ("pntfields", "pntfields_arm", "pntfields_2d"):
        cfg = default_config(name)
        assert config_from_dict(PNTFieldsConfig, asdict(cfg)) == cfg


def test_alpha_schedule_matches_reference_loop():
    """model_res_sigmoid.py::Model.train's α, accumulated as the reference does."""
    schedule, step = AlphaSchedule(), -2000.0 / 4000.0
    for epoch in range(1, 10001):
        alpha = min(max(0.5, 0.5 + 0.5 * step), 1.05)
        step += 1.0 / 4000 / (int(epoch / 4000) + 1.0)
        assert abs(schedule(epoch - 1) - alpha) < 1e-12, epoch


def test_lr_ramp_matches_reference():
    ramp = LRRamp(1e-3, 5e-4, 8000, 1000)
    for epoch in (1, 8000, 8001, 8250, 8500, 9000):
        expected = min(max(1e-3 * (1 - (epoch - 8000) / 1000.0), 5e-4), 1e-3)
        assert abs(ramp.lr(epoch - 1) - expected) < 1e-15, epoch


def test_travel_time_is_symmetric():
    torch.manual_seed(0)
    model = PNTFieldsModel(3, hidden=32)
    q_s, q_g = torch.rand(16, 3) - 0.5, torch.rand(16, 3) - 0.5
    torch.testing.assert_close(model(q_s, q_g, SceneCondition()), model(q_g, q_s, SceneCondition()))


def test_tau_jet_matches_autograd():
    torch.manual_seed(0)
    model = PNTFieldsModel(3, hidden=32, encoder_blocks=1, generator_blocks=2).double()
    x = (torch.rand(16, 6, dtype=torch.float64) - 0.5).requires_grad_(True)
    tau = model.tau(x[:, :3], x[:, 3:])
    dtau, hess = grad_and_hessian_diag(tau, x)
    for ours, expected in zip(model.tau_jet(x[:, :3], x[:, 3:]), (tau, dtau, hess), strict=True):
        torch.testing.assert_close(ours, expected, rtol=1e-6, atol=1e-7)  # autograd's softplus is linear past β·x > 20


def test_loss_speed_is_one_over_grad_t():
    """With γ = 0 and α = 1 the loss is x + 1/x - 2 at both ends with x = √(S/S*), S = 1/‖∇T‖ from autograd on T;
    the Laplacian term moves it as S_visc = 1/(1/S + γ·Δτ)."""
    torch.manual_seed(0)
    model = PNTFieldsModel(2, hidden=32).double()
    q_s, q_g = (torch.rand(32, 2, dtype=torch.float64) - 0.5 for _ in range(2))
    speed_s, speed_g = (0.1 + 0.9 * torch.rand(32, dtype=torch.float64) for _ in range(2))
    batch = Batch(q_s, q_g, speed_s, speed_g, q_s * 0, q_g * 0, SceneCondition())
    state = TrainState(model=model, optimizer=torch.optim.SGD(model.parameters()), epoch=10_000)

    x = torch.cat([q_s, q_g], dim=1).requires_grad_(True)
    g_t = torch.autograd.grad(model(x[:, :2], x[:, 2:], SceneCondition()).sum(), x)[0]
    _, hess = grad_and_hessian_diag(model.tau(x[:, :2], x[:, 2:]), x)

    def expected(gamma: float) -> torch.Tensor:
        total = torch.zeros(32, dtype=torch.float64)
        for side, speed in ((slice(0, 2), speed_s), (slice(2, 4), speed_g)):
            s = 1 / (1 / (1 / g_t[:, side].norm(dim=-1)) + gamma * hess[:, side].sum(-1))
            r = (s / speed).sqrt()
            total += r + 1 / r - 2
        return total.mean()

    alpha_one = AlphaSchedule(start=1.0, end=1.0)
    for gamma in (0.0, 1e-3):
        cfg = PNTFieldsConfig(hidden=32, viscosity=gamma, alpha=alpha_one)
        torch.testing.assert_close(pntfields_loss(model, batch, state, cfg).total, expected(gamma).detach())


def test_alpha_moves_the_speed_target_toward_one():
    torch.manual_seed(0)
    model = PNTFieldsModel(2, hidden=16)
    q = torch.rand(8, 2) - 0.5
    batch = Batch(q, -q, torch.full((8,), 0.2), torch.full((8,), 0.2), q * 0, q * 0, SceneCondition())
    state = TrainState(model=model, optimizer=torch.optim.SGD(model.parameters()))
    assert pntfields_loss(model, batch, state, PNTFieldsConfig(hidden=16)).terms["alpha"] == 0.5


def test_train_and_reload(tmp_path: Path):
    problem = Problem(load_env("box2d/one_sphere"), Clipped())
    tc = TrainConfig(n_pairs=256, batch_size=64, batches_per_epoch=2, device="cpu", out_dir=tmp_path)
    field = ntfields.train("pntfields", problem, ntfields.Epochs(3), cfg=PNTFieldsConfig(hidden=16), train_cfg=tc)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    assert loaded.method.cfg == field.method.cfg
    q = torch.rand(8, 2) - 0.5
    torch.testing.assert_close(loaded.travel_time(q, -q), field.travel_time(q, -q))
