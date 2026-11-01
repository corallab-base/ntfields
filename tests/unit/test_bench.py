import json
import math
from pathlib import Path

import pytest
import torch

from ntfields import Problem
from ntfields.bench import (
    Protocol,
    Record,
    RunConfig,
    SuiteIntegrityError,
    evaluate_path,
    load_suite,
    read_records,
    reference_check,
    report,
    run_suite,
    spl,
    validate,
)
from ntfields.bench.build import build_suite
from ntfields.bench.cli import main as bench_main
from ntfields.bench.protocol import densify
from ntfields.field import TrainedField
from ntfields.gt import fmm_travel_time
from ntfields.methods.euclidean.method import Euclidean, EuclideanConfig, EuclideanModel
from ntfields.plan import PLANNERS, PlanResult, get_planner, register_planner

SUITES = Path(__file__).parents[2] / "benchmarks" / "suites"
PROTOCOL = Protocol(margin=0.01, resolution=0.005)


def test_densify_keeps_vertices_and_bounds_segments():
    path = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.3]], dtype=torch.float64)
    dense = densify(path, 0.1)
    assert dense.diff(dim=0).norm(dim=-1).max() <= 0.1 + 1e-12
    assert all((dense == v).all(dim=-1).any() for v in path)


def test_evaluate_path(problem: Problem):
    start, goal = torch.tensor([-0.8, 0.7]), torch.tensor([0.8, 0.7])
    straight = evaluate_path(problem, PROTOCOL, start, goal, torch.stack([start, goal]))
    assert straight.success and straight.length == pytest.approx(1.6)
    assert straight.cost == pytest.approx(1.6, rel=1e-6)  # clearance > 0.1 everywhere along y = 0.7: S* = 1

    through = evaluate_path(problem, PROTOCOL, torch.tensor([-0.8, 0.0]), torch.tensor([0.8, 0.0]),
                            torch.tensor([[-0.8, 0.0], [0.8, 0.0]]))  # fmt: skip
    assert not through.success and through.cost is None
    short = evaluate_path(problem, PROTOCOL, start, goal, torch.stack([start, goal - 0.01]))
    assert not short.success  # must end at the goal


def test_spl():
    assert spl(True, 2.0, 1.0) == 0.5
    assert spl(True, 0.9, 1.0) == 1.0  # beating the reference caps at 1
    assert spl(False, None, 1.0) == 0.0


def test_fmm_margin_blocks_obstacles_and_descend_matches(problem: Problem):
    goal = torch.tensor([0.8, 0.0])
    gt = fmm_travel_time(problem, goal, 201, margin=0.01)
    assert not torch.isfinite(gt.value(torch.tensor([[0.0, 0.0]]))).any()
    start = torch.tensor([-0.8, 0.0])  # on the symmetry line behind the sphere: T has a kink here
    path = gt.descend(start, 0.005)
    assert path is not None
    outcome = evaluate_path(problem, PROTOCOL, start, goal, path)
    assert outcome.success and outcome.cost == pytest.approx(float(gt.value(start[None])), rel=0.01)


def _tiny_suite(root: Path) -> None:
    scene = {"kind": "sphere_box", "bounds": [[-1, 1], [-1, 1]], "spheres": [{"center": [0, 0], "radius": 0.3}]}
    build_suite(root, "tiny", 1, "test suite", {"one_sphere": scene}, {"kind": "clipped", "dmin": 0.01, "dmax": 0.1},
                PROTOCOL, {"dev": 2, "test": 3}, min_distance=0.5, fmm_resolution=101, seed=0, min_seeds=1)  # fmt: skip


def test_build_load_and_checksums(tmp_path: Path):
    _tiny_suite(tmp_path)
    suite = load_suite("tiny-v1", tmp_path)
    q = suite.get("one_sphere").queries["test"]
    assert len(q) == 3 and q.c_star is not None and q.t_star is not None
    assert ((q.start - q.goal).norm(dim=-1) >= 0.5).all()
    with pytest.raises(FileExistsError):
        _tiny_suite(tmp_path)  # released suites are never overwritten
    path = tmp_path / "tiny-v1" / "queries" / "one_sphere.json"
    data = json.loads(path.read_text())
    data["test"]["start"][0][0] += 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(SuiteIntegrityError):
        load_suite("tiny-v1", tmp_path)


def test_record_round_trip_and_rejects_unknown(tmp_path: Path):
    r = Record("s", 1, "abc", "p", "test", "m", "m", "pkg", "0", {"lr": 1e-3}, "planner", "0.1", 0, "cpu", "2", "d")
    path = tmp_path / "r.jsonl"
    path.write_text(r.to_json() + "\n")
    assert read_records(path) == [r]
    with pytest.raises(ValueError, match="unknown"):
        Record.from_dict({**json.loads(r.to_json()), "extra": 1})
    with pytest.raises(ValueError, match="schema_version"):
        Record.from_dict({**json.loads(r.to_json()), "schema_version": 999})


def test_planner_registry(problem: Problem):
    def straight(field: TrainedField, start: torch.Tensor, goal: torch.Tensor) -> PlanResult:
        return PlanResult(torch.stack([start, goal]), True)

    register_planner("test_straight", straight)
    try:
        field = TrainedField(Euclidean(EuclideanConfig()), problem, EuclideanModel())
        assert field.plan(torch.zeros(2), torch.ones(2), "test_straight").path.shape == (2, 2)
        with pytest.raises(KeyError):
            get_planner("no_such_planner")
    finally:
        del PLANNERS["test_straight"]


def test_run_report_validate_and_reuse(tmp_path: Path):
    cfg = RunConfig(suite="smoke-v1", entries=["euclidean"], seeds=[0], suites_dir=SUITES, device="cpu",
                    out=tmp_path / "r.jsonl", checkpoint_dir=tmp_path / "ckpt", epochs=2, n_pairs=256)  # fmt: skip
    (first,) = run_suite(cfg)
    assert first.n == 4 and len(first.success) == 4 and first.spl is not None and first.t_error_mean is not None
    assert first.checkpoint is not None and Path(first.checkpoint).exists()
    assert first.method_package == "ntfields" and first.planner == "gradient_bidirectional"

    (again,) = run_suite(cfg)  # same config and recipe: evaluated from the checkpoint, not retrained
    assert again.train_seconds == first.train_seconds and again.success == first.success

    records = read_records(cfg.out)  # pyright: ignore[reportArgumentType]
    assert len(records) == 2 and "euclidean+gradient_bidirectional" in report(records)
    assert validate(records[:1], SUITES) == []
    assert reference_check(records, SUITES) == []
    cheat = Record(**{**vars(records[0]), "success": [True] * 4, "cost": [0.1] * 4})
    assert len(reference_check([cheat], SUITES)) == 1
    tampered = Record(**{**vars(records[0]), "suite_sha256": "0" * 64, "split": "dev"})
    errors = validate([tampered], SUITES)
    assert any("checksum" in e for e in errors) and any("split" in e for e in errors)


def test_bench_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bench_main(["list", "--suites-dir", str(SUITES)])
    assert "smoke-v1 [not ranked]" in capsys.readouterr().out
    out = tmp_path / "r.jsonl"
    bench_main(["run", "--suite", "smoke-v1", "--entries", "euclidean", "--seeds", "0", "--planners", "all",
                "--suites-dir", str(SUITES), "--device", "cpu", "--out", str(out), "--checkpoint-dir", str(tmp_path),
                "--epochs", "1", "--n-pairs", "64"])  # fmt: skip
    assert {r.planner for r in read_records(out)} == {"gradient_bidirectional", "mppi"}
    bench_main(["report", "--records", str(out)])
    assert "smoke-v1 (test)" in capsys.readouterr().out
    bench_main(["validate", "--records", str(out), "--suites-dir", str(SUITES)])
    assert capsys.readouterr().out.strip() == "valid"


def test_committed_suites_load():
    for name in ("smoke-v1", "box2d-v1"):
        suite = load_suite(name, SUITES)
        for p in suite.problems:
            q = p.queries["test"]
            assert q.c_star is not None and q.t_star is not None
            assert math.isclose(float((q.c_star / q.t_star).mean()), 1.0, rel_tol=0.02)


def test_rollbacks_are_counted(problem: Problem):
    from ntfields import Epochs, TrainConfig, train
    from ntfields.bench.run import _TrainStats  # pyright: ignore[reportPrivateUsage]
    from ntfields.train import LossRatioRollback

    stats = _TrainStats()
    tc = TrainConfig(n_pairs=128, batch_size=64, device="cpu", out_dir=None)
    rollback = LossRatioRollback(threshold=1e-9, max_retries=3)  # rejects every epoch until the retry cap
    train("euclidean", problem, Epochs(2), train_cfg=tc, callbacks=[rollback, stats])
    assert stats.epochs == 2 and stats.rollbacks == 2 * 2  # each epoch: 2 rejections, then accepted at the cap
