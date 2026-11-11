from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from pytorch_kinematics.transforms import matrix_to_euler_angles

import ntfields
from ntfields import Problem, TrainConfig
from ntfields.checkpoint import config_from_dict
from ntfields.data import ManifoldPairs, sample_pairs
from ntfields.envs import ConstrainedEnv, SphereBoxEnv, SphereConstraint
from ntfields.envs.tsr import TSRConstraint, ur5e_cup_env, ur5e_cup_problem
from ntfields.eval.metrics import path_length
from ntfields.methods.cntfields import CNTFields, CNTFieldsConfig, LinearAlpha
from ntfields.nn import RandomFourierFeatures
from ntfields.plan import gradient_bidirectional
from ntfields.registry import default_config
from ntfields.speed import Clipped, GaussianBand
from ntfields.testing import check_method


def _sphere_problem(obstacle: bool = False) -> Problem:
    spheres = [((0.4, 0.0, 0.0), 0.1)] if obstacle else []
    env = ConstrainedEnv(
        SphereBoxEnv.build([(-0.6, 0.6)] * 3, spheres), SphereConstraint(), epsilon=0.25, tolerance=0.1
    )
    return Problem(env, GaussianBand(epsilon=0.25))


def test_check_method():
    check_method(CNTFields)


def test_presets_round_trip():
    for name in ("cntfields", "cntfields_3d"):
        cfg = default_config(name)
        assert isinstance(cfg, CNTFieldsConfig)
        assert config_from_dict(CNTFieldsConfig, asdict(cfg)) == cfg


def test_linear_alpha_matches_reference_expression():
    alpha = LinearAlpha()
    for epoch in (0, 1, 2500, 4499, 4500, 4999):
        assert alpha(epoch) == min(0.5 + epoch * 1e-4, 0.95)


def test_truncated_fourier_features_stay_in_range():
    b = RandomFourierFeatures(3, 4096, 1.0, truncate=2.0).b
    assert b.abs().max() <= 2.0 and b.std() > 0.8


def test_gaussian_band_matches_reference_speed():
    d = torch.linspace(0, 1, 50)
    s = torch.exp(-((d / 0.7) ** 2) / 0.1)  # sample_tsr.py::compute_speed(minimum=0.7, scaling_factor=0.1)
    torch.testing.assert_close(GaussianBand(eased=False)(0.7 - d), s)
    torch.testing.assert_close(GaussianBand()(0.7 - d), s * s * (2 - s) * (2 - s))  # base_model.py::train


def test_constrained_env_reproduces_eq_4():
    env = ConstrainedEnv(
        SphereBoxEnv.build([(-1, 1)] * 3, [((0.4, 0.0, 0.0), 0.1)]), SphereConstraint(), epsilon=0.25, tolerance=0.05
    )
    q = torch.rand(500, 3) * 1.2 - 0.6
    d_m, d_c = env.constraint.violation(q), env.base.distance(q)
    torch.testing.assert_close(env.distance(q), 0.25 - torch.maximum(d_m, 0.25 - d_c))
    assert (env.is_valid(q) == ((d_m < 0.05) & (d_c > 0) & env.base.in_bounds(q))).all()  # type: ignore[attr-defined]


def test_clearance_rescales_only_the_obstacle_term():
    env = _sphere_problem(obstacle=True).env
    assert isinstance(env, ConstrainedEnv)
    q = torch.rand(32, 3) - 0.5
    expected = torch.minimum(env.epsilon - env.constraint.violation(q), env.base.distance(q) * env.epsilon / 0.05)
    torch.testing.assert_close(replace(env, clearance=0.05).distance(q), expected)


def test_no_obstacles_leaves_only_the_manifold_distance():
    env = _sphere_problem().env
    q = torch.rand(100, 3) - 0.5
    torch.testing.assert_close(env.distance(q), 0.25 - env.constraint.violation(q))  # type: ignore[attr-defined]


def test_sphere_tangent_removes_the_normal():
    c = SphereConstraint()
    q, v = torch.randn(100, 3), torch.randn(100, 3)
    t = c.tangent(q, v)
    torch.testing.assert_close((t * q).sum(-1), torch.zeros(100), rtol=0, atol=1e-5)
    torch.testing.assert_close(c.tangent(q, t), t)  # a projection


def test_manifold_pairs_stay_in_the_band_and_need_a_constraint():
    pairs = sample_pairs(_sphere_problem(), 500, ManifoldPairs())
    assert SphereConstraint().violation(pairs.q_s).max() <= 0.1 + 1e-6
    assert SphereConstraint().violation(pairs.q_g).max() <= 0.1 + 1e-6
    assert (pairs.q_s != pairs.q_g).any()
    with pytest.raises(TypeError, match="ConstrainedEnv"):
        sample_pairs(Problem(SphereBoxEnv.build([(-1, 1)] * 3), Clipped()), 10, ManifoldPairs())


def test_tangent_planner_needs_a_constraint():
    problem = Problem(SphereBoxEnv.build([(-1, 1)] * 3), Clipped())
    field = ntfields.TrainedField(
        CNTFields(CNTFieldsConfig()), problem, CNTFields(CNTFieldsConfig()).build_model(problem)
    )
    with pytest.raises(ValueError, match="ConstrainedEnv"):
        gradient_bidirectional(field, torch.zeros(3), torch.ones(3), tangent=True)


def test_trains_reloads_and_plans(tmp_path: Path):
    train_cfg = TrainConfig(n_pairs=500, batch_size=100, device="cpu", out_dir=tmp_path)
    cfg = CNTFieldsConfig(hidden=16, fourier_features=8)
    field = ntfields.train("cntfields", _sphere_problem(obstacle=True), ntfields.Epochs(2), cfg, train_cfg)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    start, goal = torch.tensor([0.0, 0.4, 0.0]), torch.tensor([0.0, 0.0, 0.4])
    torch.testing.assert_close(loaded.travel_time(start[None], goal[None]), field.travel_time(start[None], goal[None]))
    path = gradient_bidirectional(loaded, start, goal, tangent=True).path
    assert path.isfinite().all() and path_length(path) > 0


def _cup() -> TSRConstraint:
    constraint = ur5e_cup_env().constraint
    assert isinstance(constraint, TSRConstraint)
    return constraint


def test_tsr_violation_matches_the_reference_distance():
    """Against a copy of ntfields-old's `TSRSampler.distance` body, in terms of the same FK."""
    c = _cup()
    q = c.sample(300, torch.Generator().manual_seed(0))
    T_o_s = c.arm.end_pose(q)
    T_o_sp = T_o_s @ torch.inverse(c.T_w_e)
    T_w_sp = torch.inverse(c.T_o_w) @ T_o_sp
    d_w = torch.cat([T_w_sp[:, :3, 3], matrix_to_euler_angles(T_w_sp[:, :3, :3], "XYZ")], dim=1)
    d_w[:, 3:] = torch.remainder(d_w[:, 3:] + torch.pi, 2 * torch.pi)
    d_w[:, 3:] = torch.where(d_w[:, 3:] < 0, d_w[:, 3:] + 2 * torch.pi, d_w[:, 3:]) - torch.pi
    dx = torch.zeros_like(d_w)
    for i, (lower, higher) in enumerate(c.bounds):
        dx[d_w[:, i] < lower, i] = d_w[d_w[:, i] < lower, i] - lower
        dx[d_w[:, i] > higher, i] = d_w[d_w[:, i] > higher, i] - higher
    torch.testing.assert_close(c.violation(q), dx.norm(dim=1))


def test_tsr_samples_reach_their_band_and_clear_the_arm():
    c = _cup()
    q = c.sample(500, torch.Generator().manual_seed(1))
    assert len(q) == 500 and c.arm.in_bounds(q).all()
    assert (c.arm.self_distance(q) > 0).all()
    # ±0.25 rad on roll and pitch: at most √2·0.25 from the manifold
    assert c.violation(q).max() < 2**0.5 * 0.25 + 1e-3
    assert c.violation(q).mean() > 0.05


def test_cup_sheet_is_one_piece_and_crosses_the_wrist_singularity():
    """Every pair must have an on-manifold path, so the samples form one component of a neighbourhood graph."""
    q = replace(_cup(), offsets=torch.zeros(6)).sample(1500, torch.Generator().manual_seed(0))
    assert (q[:, 5] < 0).all() and (q[:, 4] > 0).any() and (q[:, 4] < 0).any()
    near = torch.cdist(q, q) < 0.15
    reached = torch.zeros(len(q), dtype=torch.bool)
    reached[0] = True
    while (grown := near[reached].any(0) | reached).sum() > reached.sum():
        reached = grown
    assert reached.all()


def test_sample_pairs_chunks_do_not_change_the_labels():
    problem = _sphere_problem(obstacle=True)
    whole, chunked = sample_pairs(problem, 64), sample_pairs(problem, 64, chunk=7)
    torch.testing.assert_close(chunked.speed_s, whole.speed_s)
    torch.testing.assert_close(chunked.normal_g, whole.normal_g)


def test_tsr_on_manifold_samples_have_no_violation():
    c = replace(_cup(), offsets=torch.zeros(6))
    q = c.sample(200, torch.Generator().manual_seed(2))
    assert c.violation(q).max() < 1e-2  # IK accuracy in float32


def test_tsr_tangent_is_a_projection_that_keeps_the_cup_upright():
    c = _cup()
    q = c.sample(50, torch.Generator().manual_seed(3))
    t = c.tangent(q, torch.randn(50, 6))
    torch.testing.assert_close(c.tangent(q, t), t, rtol=1e-3, atol=1e-3)
    # a small tangent step changes roll and pitch to second order only, a raw step to first order
    eps = 1e-3
    drift = (c.displacement(q + eps * t) - c.displacement(q))[:, 3:5].abs().amax(dim=1)
    assert drift.median() < 1e-4 * (1 + t.norm(dim=-1).median())


def test_cup_problem_checkpoints_round_trip(tmp_path: Path):
    problem = ur5e_cup_problem(table=True)
    cfg = CNTFieldsConfig(hidden=16, fourier_features=8)
    train_cfg = TrainConfig(n_pairs=200, batch_size=100, device="cpu", out_dir=tmp_path)
    field = ntfields.train("cntfields", problem, ntfields.Epochs(1), cfg, train_cfg)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    q = problem.env.constraint.sample(5, torch.Generator().manual_seed(4))  # type: ignore[attr-defined]
    torch.testing.assert_close(loaded.problem.speed_at(q), problem.speed_at(q))
    torch.testing.assert_close(loaded.travel_time(q, q.flip(0)), field.travel_time(q, q.flip(0)))
