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
from ntfields.bench.build import build_challenge, build_suite, tunneling_share
from ntfields.bench.cli import main as bench_main
from ntfields.bench.protocol import densify
from ntfields.bench.run import backfill_challenge
from ntfields.bench.suite import Queries, SuiteProblem, sha256
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
    meta = json.loads((tmp_path / "tiny-v1" / "suite.json").read_text())
    assert meta["reference"]["tunneling"] == {"one_sphere": 0.0}  # a 0.6-wide sphere is never worth crossing
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
    for name in ("smoke-v1", "box2d-v1", "box3d-v1"):
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


def test_tunneling_share_flags_thin_walls():
    wall = {"kind": "sphere_box", "bounds": [[-1, 1], [-1, 1]], "boxes": [{"center": [0, 0.1], "size": [0.04, 1.8]}]}
    problem = SuiteProblem("wall", wall, {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}, {}).problem()
    start, goal = torch.tensor([[-0.5, -0.5]]), torch.tensor([[0.5, -0.5]])  # the gap is at the bottom edge
    t_star = fmm_travel_time(problem, goal[0], 101, margin=0.01).value(start)
    assert tunneling_share(problem, Queries(start, goal, t_star=t_star.double()), 101, 0.02) == 0.0  # gap is close
    start, goal = torch.tensor([[-0.5, 0.8]]), torch.tensor([[0.5, 0.8]])  # far from the gap: crossing is cheaper
    t_star = fmm_travel_time(problem, goal[0], 101, margin=0.01).value(start)
    assert tunneling_share(problem, Queries(start, goal, t_star=t_star.double()), 101, 0.02) == 1.0


def _challenge_suite(tmp_path: Path) -> Path:
    scene = {"kind": "sphere_box", "bounds": [[-1, 1]] * 2, "spheres": [{"center": [0.0, 0.0], "radius": 0.3}]}
    empty = {"kind": "sphere_box", "bounds": [[-1, 1]] * 2}
    speed = {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}
    build_suite(tmp_path, "c", 1, "challenge", {"one": scene, "empty": empty}, speed, PROTOCOL, {"dev": 1, "test": 2},
                min_distance=0.5, fmm_resolution=51, seed=0, min_seeds=1)  # fmt: skip
    return tmp_path / "c-v1"


def test_challenge_pairs_need_a_detour_and_leave_suite_json_alone(tmp_path: Path):
    base = _challenge_suite(tmp_path)
    before = load_suite("c-v1", tmp_path).sha256
    build_challenge(base, 3, seed=1, skip=("empty",))
    suite = load_suite("c-v1", tmp_path)
    assert suite.sha256 == before
    assert suite.challenge_sha256 is not None
    assert suite.get("empty").challenge is None
    pairs = suite.get("one").challenge
    assert pairs is not None and len(pairs) == 3
    problem = suite.get("one").problem()
    for s, g in zip(pairs.start, pairs.goal, strict=True):
        assert not evaluate_path(problem, suite.protocol, s, g, torch.stack([s, g])).success
    assert pairs.straight_free is not None and not pairs.straight_free.any()

    file = base / "challenge" / "one.json"
    file.write_text(file.read_text().replace("0.", "0.1", 1))
    with pytest.raises(SuiteIntegrityError):
        load_suite("c-v1", tmp_path)


def test_challenge_metric_run_report_backfill_and_validate(tmp_path: Path):
    base = _challenge_suite(tmp_path)
    build_challenge(base, 3, seed=1, skip=("empty",))
    out = tmp_path / "records.jsonl"
    cfg = RunConfig("c-v1", ["euclidean"], seeds=[0], planners=["gradient_bidirectional"], out=out, suites_dir=tmp_path,
                    checkpoint_dir=tmp_path / "ckpt", device="cpu", epochs=1, n_pairs=16)  # fmt: skip
    records = {r.problem: r for r in run_suite(cfg)}
    assert len(records["one"].challenge_success) == 3
    assert records["one"].challenge_success_rate == sum(records["one"].challenge_success) / 3
    assert records["empty"].challenge_success_rate is None
    assert "challenge %" in report(list(records.values()))
    assert validate(read_records(out), tmp_path) == []

    rows = read_records(out)
    for r in rows:
        if r.problem == "one":
            r.challenge_success, r.challenge_success_rate, r.challenge_sha256 = [], None, None
    out.write_text("".join(r.to_json() + "\n" for r in rows))
    before = [{k: v for k, v in vars(r).items() if "challenge" not in k} for r in read_records(out)]
    assert backfill_challenge(out, tmp_path, "cpu") == 1
    after = read_records(out)
    assert [{k: v for k, v in vars(r).items() if "challenge" not in k} for r in after] == before
    assert next(r for r in after if r.problem == "one").challenge_success == records["one"].challenge_success
    assert backfill_challenge(out, tmp_path, "cpu") == 0


def test_all_detour_suites_report_their_success_as_the_challenge(tmp_path: Path):
    base = _challenge_suite(tmp_path)
    path = base / "queries" / "one.json"
    data = json.loads(path.read_text())
    data["test"]["straight_free"] = [False, False]
    path.write_text(json.dumps(data))
    meta = json.loads((base / "suite.json").read_text())
    meta["files"]["queries/one.json"] = sha256(path)
    (base / "suite.json").write_text(json.dumps(meta))
    cfg = RunConfig("c-v1", ["euclidean"], seeds=[0], planners=["gradient_bidirectional"], problems=["one"],
                    out=tmp_path / "r.jsonl", suites_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt", device="cpu",
                    epochs=1, n_pairs=16)  # fmt: skip
    record = run_suite(cfg)[0]
    assert record.challenge_success_rate == record.success_rate
    assert record.challenge_success == []
