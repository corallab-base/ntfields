import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

import ntfields
from ntfields import Problem, SceneCondition, TrainConfig
from ntfields.base import Batch, TravelTimeModel
from ntfields.control import GeodesicTauConfig, GeodesicTauController, PointMassDynamics
from ntfields.data import PairDataset, sample_pairs
from ntfields.envs import ArmEnv, load_env
from ntfields.field import TrainedField
from ntfields.methods.rantfields import RaNTFields, RaNTFieldsConfig
from ntfields.methods.rantfields.iqe import IQE
from ntfields.methods.rantfields.loss import rantfields_loss
from ntfields.metric import FlowField, Terrain
from ntfields.speed import ExpRamp
from ntfields.testing import check_method

FLOW = (0.5, 0.2)


class Zermelo(TravelTimeModel):
    """Exact travel time from q_s to q_g at unit speed under the constant drift f: the T with ‖d - fT‖ = T."""

    def __init__(self, flow: tuple[float, float]) -> None:
        super().__init__()
        self.f: Tensor
        self.register_buffer("f", torch.tensor(flow))

    def forward(self, q_s: Tensor, q_g: Tensor, scene: SceneCondition | None = None) -> Tensor:
        d, f = q_g - q_s, self.f
        df, ff = d @ f, f @ f
        return (-df + torch.sqrt(df**2 + (d * d).sum(-1) * (1 - ff))) / (1 - ff)


@pytest.fixture
def flow_problem() -> Problem:
    env = load_env("box2d/empty")
    return Problem(env, ExpRamp.for_env(env), FlowField("constant", vector=FLOW))


def _field(problem: Problem, model: TravelTimeModel) -> TrainedField:
    return TrainedField(RaNTFields(RaNTFieldsConfig()), problem, model)


def test_check_method(flow_problem: Problem):
    check_method(RaNTFields, flow_problem, soft_boundary=True)


def test_iqe_is_a_quasimetric():
    torch.manual_seed(0)
    d = IQE(32)
    x, y, z = torch.randn(3, 64, 32).unbind()
    assert d(x, x).abs().max() < 1e-6
    assert (d(x, z) <= d(x, y) + d(y, z) + 1e-5).all()
    assert not torch.allclose(d(x, y), d(y, x))


def test_loss_vanishes_on_the_exact_zermelo_time(flow_problem: Problem):
    """The residual's drift sign: zero for the closed-form travel time under a constant drift (S* = 1)."""
    pairs = sample_pairs(flow_problem, 256)
    batch = next(pairs.batches(256, SceneCondition()))
    out = rantfields_loss(Zermelo(FLOW), batch, flow_scale=1.0, w_boundary=5.0)
    assert out.terms["randers"] < 1e-8 and out.terms["boundary"] < 1e-10
    flipped = rantfields_loss(Zermelo((-FLOW[0], -FLOW[1])), batch, flow_scale=1.0, w_boundary=5.0)
    assert flipped.terms["randers"] > 1e-2


def test_sample_pairs_labels_the_metric(flow_problem: Problem, tmp_path: Path):
    pairs = sample_pairs(flow_problem, 64)
    assert pairs.metric_s is not None and pairs.drift_g is not None
    torch.testing.assert_close(pairs.drift_s, torch.tensor(FLOW).expand(64, -1))
    torch.testing.assert_close(pairs.metric_g, torch.eye(2).expand(64, -1, -1))
    pairs.save(tmp_path / "pairs.h5")
    loaded = PairDataset.load(tmp_path / "pairs.h5")
    assert loaded.metric_s is not None
    torch.testing.assert_close(loaded.metric_s, pairs.metric_s)
    plain = sample_pairs(Problem(flow_problem.env, flow_problem.speed), 8)
    assert plain.metric_s is None and plain.drift_s is None


def test_terrain_metric_is_feasible():
    q = torch.cartesian_prod(*[torch.linspace(-1, 1, 101, dtype=torch.float64)] * 2)
    for t in (Terrain("single_hill"), Terrain("smooth_ridge", height=0.4)):
        G, f = t.G(q), t.drift(q)
        assert (torch.linalg.eigvalsh(G) >= 1 - 1e-12).all()
        assert torch.einsum("bi,bij,bj->b", f, G, f).max() <= t.max_norm_g_sq + 1e-9


def test_flow_fields():
    q = torch.tensor([[0.5, 0.0], [0.0, 2.0], [0.0, 0.0]])
    torch.testing.assert_close(FlowField("circular", magnitude=0.7).drift(q)[:2], torch.tensor([[0, 0.7], [-0.7, 0]]))
    assert FlowField("circular").drift(q)[2].abs().sum() == 0
    torch.testing.assert_close(FlowField("saddle", magnitude=0.9).drift(q)[1], torch.tensor([0.0, -0.9]))


def test_exp_ramp():
    speed = ExpRamp(d_max=1.0)
    s = speed(torch.tensor([-0.1, 0.0, 1e-3, math.log(1.1), 5.0, math.inf]))
    torch.testing.assert_close(s, torch.tensor([1e-4, 1e-4, 0.1, 0.5, 1.0, 1.0]))


def test_riemannian_planner_reaches_the_goal(flow_problem: Problem):
    field = _field(flow_problem, Zermelo(FLOW))
    start, goal = torch.tensor([-0.7, 0.4]), torch.tensor([0.6, -0.5])
    result = field.plan(start, goal)
    assert result.success
    torch.testing.assert_close(result.path[0], start)
    torch.testing.assert_close(result.path[-1], goal)


def test_geodesic_controller_on_a_point_mass(flow_problem: Problem):
    field = _field(flow_problem, Zermelo(FLOW))
    goal = np.array([0.5, 0.5])
    cfg = GeodesicTauConfig(cruise_speed=1.0, lam=0.0, replan_every=1)
    c = GeodesicTauController(field, PointMassDynamics(2.0), goal, np.full(2, np.inf), 0.01, cfg)
    q = np.array([-0.5, -0.5])
    qdot_des = c.desired_velocity(q)  # λ = 0: straight at the goal, at cruise speed outside decel_radius
    np.testing.assert_allclose(qdot_des, np.full(2, 1 / math.sqrt(2)), atol=1e-6)
    tau = c.compute_torque(q, np.zeros(2))
    np.testing.assert_allclose(tau, 2.0 * c.desired_velocity(q) / 0.01)  # m · q̈_des, τ_track = replan_every · dt
    qdot = np.zeros(2)
    for _ in range(500):  # semi-implicit Euler on m q̈ = τ
        if c.is_converged(q, qdot):
            break
        qdot = qdot + 0.01 * c.compute_torque(q, qdot) / 2.0
        q = q + 0.01 * qdot
    assert c.is_converged(q, qdot)


def test_train_and_reload(flow_problem: Problem, tmp_path: Path):
    tc = TrainConfig(n_pairs=512, batch_size=128, device="cpu", out_dir=tmp_path)
    field = ntfields.train("rantfields_2d", flow_problem, ntfields.Epochs(2), RaNTFieldsConfig(hidden=16), tc)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    assert loaded.problem.metric == flow_problem.metric
    q = torch.rand(8, 2) * 2 - 1
    torch.testing.assert_close(loaded.travel_time(q, q.flip(0)), field.travel_time(q, q.flip(0)))


def test_rollback_restores_the_schedule(flow_problem: Problem):
    """A rejected epoch rewinds model, optimizer and LR schedule together, and the next epoch is retried."""
    from ntfields.methods.rantfields.rollback import RetryRollback
    from ntfields.train import Callback, fit

    method = RaNTFields(RaNTFieldsConfig(hidden=16))
    model = method.build_model(flow_problem)
    rollback = RetryRollback(threshold=1e-9)  # rejects every epoch after the first
    lrs: list[float] = []

    class Record(Callback):
        def on_epoch_start(self, state: ntfields.TrainState) -> None:
            assert state.scheduler is not None
            lrs.append(float(state.scheduler.get_last_lr()[0]))

    data = sample_pairs(flow_problem, 256)
    state = fit(method, model, data, SceneCondition(), ntfields.Epochs(2), 64, [rollback, Record()], None, 2)
    assert state.epoch == 2
    # epoch 0 accepted; epoch 1 rejected twice (restored to the start of epoch 0), then accepted on the third try
    assert lrs[2] == lrs[0] and lrs[3] == lrs[0]


def test_batch_split_keeps_metric_fields(flow_problem: Problem):
    from ntfields.train.trainer import _split

    batch = next(sample_pairs(flow_problem, 10).batches(10, SceneCondition()))
    parts = _split(batch, 4)
    assert [len(p.q_s) for p in parts] == [4, 4, 2]
    assert all(isinstance(p, Batch) and p.metric_s is not None and len(p.metric_s) == len(p.q_s) for p in parts)


def test_arm_metric_and_env():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pytorch_kinematics")
    from ntfields.metric import ArmDynamics

    env = load_env("ur5e/5j_sphere")
    assert isinstance(env, ArmEnv)
    assert env.dim == 5 and env.scale(torch.zeros(1))[2] == pytest.approx(math.pi)
    assert float(env.distance(torch.zeros(1, 5))[0]) == pytest.approx(0.2667, abs=1e-4)
    q = torch.rand(6, 5, dtype=torch.float64) * 2 - 1
    m = ArmDynamics(lambda_reg=0.1, max_norm_g_sq=0.5)
    G, f = m.G(q), m.drift(q)
    assert (torch.linalg.eigvalsh(G) > 0.1 - 1e-9).all()
    theta_f = f * torch.as_tensor(np.asarray(m._cache["upper"]))  # back to joint radians, where the bound holds
    assert torch.einsum("bi,bij,bj->b", theta_f, G, theta_f).max() <= 0.5 * 0.9 + 1e-6
    via = ArmDynamics(lambda_reg=0.1, via_anisotropy=True, via_anisotropy_scale=50.0, clip_norm_g_sq=0.95)
    fv = via.drift(q)
    assert torch.einsum("bi,bij,bj->b", fv, G, fv).max() <= 0.95 + 1e-9
    assert ArmDynamics(with_drift=False).drift(q).abs().max() == 0


def test_surface_mix_anchor_end(flow_problem: Problem):
    """`anchor_end="goal"` puts every surface end on the goal side (the 2D recipe's pairs)."""
    from ntfields.data import SurfaceMix

    problem = Problem(load_env("box2d/one_sphere"), flow_problem.speed)
    q_s, q_g = SurfaceMix(shell=0.1, anchor_end="goal").sample(problem, 400, torch.Generator().manual_seed(0))
    d_s, d_g = problem.env.distance(q_s[200:]), problem.env.distance(q_g[200:])
    assert (d_g < 0.1 + 1e-6).all() and (d_g > 0).all() and d_s.mean() > d_g.mean()


def test_joint_space_mix():
    pytest.importorskip("pytorch_kinematics")
    from ntfields.methods.rantfields.sampler import JointSpaceMix

    env = load_env("ur5e/5j_sphere")
    assert isinstance(env, ArmEnv)
    problem = Problem(env, ExpRamp(1.4))
    sampler = JointSpaceMix()
    q_s, q_g = sampler.sample(problem, 200, torch.Generator().manual_seed(0))
    assert q_s.shape == q_g.shape == (200, 5) and env.in_bounds(q_s).all() and env.in_bounds(q_g).all()
    d_g = env.distance(q_g[100:]) - sampler.shift
    assert ((d_g > 0) & (d_g < sampler.margin)).all()
    step = ((q_s[100:] - q_g[100:]) * env.scale(q_s)).norm(dim=-1)
    assert (step <= sampler.max_step + 1e-5).all()
