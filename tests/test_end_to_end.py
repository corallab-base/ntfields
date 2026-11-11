from pathlib import Path

import pytest
import torch

import ntfields
from ntfields import Problem, TrainConfig
from ntfields.cli import main
from ntfields.envs import ConstrainedEnv, SphereBoxEnv, SphereConstraint
from ntfields.eval import is_collision_free, travel_time_error
from ntfields.eval.constrained import evaluate_constrained
from ntfields.gt import fmm_travel_time
from ntfields.methods.cntfields import CNTFieldsConfig
from ntfields.methods.euclidean import Euclidean
from ntfields.speed import GaussianBand
from ntfields.testing import check_method


def test_check_method_euclidean():
    check_method(Euclidean)


def test_train_plan_evaluate(problem: Problem, tmp_path: Path):
    tc = TrainConfig(n_pairs=256, batch_size=64, device="cpu", out_dir=tmp_path)
    trained = ntfields.train("euclidean", problem, ntfields.Epochs(20), train_cfg=tc)
    assert trained.ckpt_path is not None
    field = ntfields.load(trained.ckpt_path)

    start, goal = torch.tensor([-0.8, 0.7]), torch.tensor([0.8, 0.7])
    result = field.plan(start, goal)
    assert result.success and is_collision_free(problem, result.path)

    gt = fmm_travel_time(problem, goal, resolution=101)
    starts = torch.tensor([[-0.8, 0.7], [0.0, -0.8], [0.5, 0.5]])
    assert torch.isfinite(torch.tensor(travel_time_error(field, gt, starts)))


def test_cli_train(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    args = ["train", "euclidean", "--epochs", "2", "--train.n-pairs", "64", "--train.device", "cpu"]
    main([*args, "--train.out-dir", str(tmp_path)])
    assert "checkpoint:" in capsys.readouterr().out
    assert list(tmp_path.glob("euclidean/*/ckpt.pt"))


def test_cntfields_tangent_planner_stays_on_a_sphere() -> None:
    """A short CPU run: the fronts meet along the surface, well within the tolerance."""
    env = ConstrainedEnv(SphereBoxEnv.build([(-0.6, 0.6)] * 3), SphereConstraint(), epsilon=0.25, tolerance=0.1)
    problem = Problem(env, GaussianBand(epsilon=0.25))
    cfg = CNTFieldsConfig(hidden=32, fourier_features=16)
    train_cfg = TrainConfig(n_pairs=4000, batch_size=500, device="cpu", out_dir=None)
    field = ntfields.train("cntfields", problem, ntfields.Epochs(60), cfg, train_cfg)
    generator = torch.Generator().manual_seed(0)
    starts = torch.nn.functional.normalize(torch.randn(10, 3, generator=generator), dim=-1) * 0.4
    goals = torch.nn.functional.normalize(torch.randn(10, 3, generator=generator), dim=-1) * 0.4
    result = evaluate_constrained(field, starts, goals, beta=0.01, goal_tol=0.03, max_steps=200, tangent=True)
    assert result["success"] >= 0.8
    assert result["margin_max"] < 0.05
