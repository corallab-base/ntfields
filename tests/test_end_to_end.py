from pathlib import Path

import pytest
import torch

import ntfields
from ntfields import Problem, TrainConfig
from ntfields.cli import main
from ntfields.eval import is_collision_free, travel_time_error
from ntfields.gt import fmm_travel_time
from ntfields.methods.euclidean import Euclidean
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
