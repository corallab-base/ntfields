import json
import logging
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from typing import Any

import torch

import ntfields
from ntfields.base import Method, TrainState
from ntfields.bench.protocol import evaluate_path, percentile, spl
from ntfields.bench.recipes import get_recipe
from ntfields.bench.schema import Record, append_records
from ntfields.bench.suite import DEFAULT_ROOT, Suite, SuiteProblem, load_suite, sha256
from ntfields.checkpoint import load, save
from ntfields.field import TrainedField
from ntfields.plan import PLANNERS
from ntfields.problem import Problem
from ntfields.registry import create
from ntfields.train import Callback

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    suite: str
    entries: list[str]  # method or preset names
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    planners: list[str] = field(default_factory=lambda: ["default"])  # "default", "all" or planner names
    split: str = "test"
    problems: list[str] | None = None  # None: every problem in the suite
    out: Path | None = None  # records are appended here (default results/<suite id>.jsonl)
    suites_dir: Path = DEFAULT_ROOT
    checkpoint_dir: Path = Path("benchmarks/checkpoints")
    reuse: bool = True  # evaluate a matching checkpoint instead of retraining
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int | None = None  # override every recipe (smoke tests)
    n_pairs: int | None = None
    submitter: str | None = None


def run_suite(cfg: RunConfig) -> list[Record]:
    suite = load_suite(cfg.suite, cfg.suites_dir)
    out = cfg.out or Path("results") / f"{suite.id}.jsonl"
    records = []
    for sp in [suite.get(p) for p in cfg.problems] if cfg.problems else suite.problems:
        problem = sp.problem()
        for preset in cfg.entries:
            for seed in cfg.seeds:
                trained, training = _field(cfg, suite, sp, problem, preset, seed)
                for planner in _planners(cfg.planners, trained.method):
                    record = evaluate(cfg, suite, sp, trained, preset, planner, seed, training)
                    append_records(out, [record])
                    records.append(record)
                    log.info("%s %s %s seed %d: success %.3f spl %s", suite.id, sp.name, record.entry, seed,
                             record.success_rate, record.spl)  # fmt: skip
    return records


def evaluate(
    cfg: RunConfig,
    suite: Suite,
    sp: SuiteProblem,
    trained: TrainedField,
    preset: str,
    planner: str,
    seed: int,
    training: dict[str, Any],
) -> Record:
    problem, queries = trained.problem, sp.queries[cfg.split]
    starts, goals = queries.start.to(cfg.device), queries.goal.to(cfg.device)
    t_err_mean = t_err_median = None
    if queries.t_star is not None:
        t = trained.travel_time(starts, goals).double().cpu()
        if t.isfinite().all():
            rel = ((t - queries.t_star) / queries.t_star).abs().tolist()
            t_err_mean, t_err_median = statistics.fmean(rel), statistics.median(rel)
        else:
            log.warning("%s %s seed %d: non-finite travel times; T error not recorded", sp.name, preset, seed)

    success, converged, cost, length, seconds, scores = [], [], [], [], [], []
    for i, (s, g) in enumerate(zip(starts, goals, strict=True)):
        torch.manual_seed(seed * 1_000_003 + i)  # stochastic planners are reproducible per (seed, query)
        _sync(cfg.device)
        t0 = time.perf_counter()
        result = trained.plan(s, g, planner)
        _sync(cfg.device)
        seconds.append(time.perf_counter() - t0)
        outcome = evaluate_path(problem, suite.protocol, s, g, result.path)
        success.append(outcome.success)
        converged.append(bool(result.success))
        cost.append(outcome.cost)
        length.append(outcome.length)
        if queries.c_star is not None:
            scores.append(spl(outcome.success, outcome.cost, float(queries.c_star[i])))

    ok = [i for i, s in enumerate(success) if s]
    ratios = [cost[i] / float(queries.c_star[i]) for i in ok if queries.c_star is not None]  # pyright: ignore[reportOptionalOperand]
    method = trained.method
    package, pkg_version = _package(method)
    return Record(
        suite=suite.name,
        suite_version=suite.version,
        suite_sha256=suite.sha256,
        problem=sp.name,
        split=cfg.split,
        method=method.name,
        preset=preset,
        method_package=package,
        method_version=pkg_version,
        config=_jsonable(asdict(method.cfg)),
        planner=planner,
        ntfields_version=ntfields.__version__,
        seed=seed,
        device=cfg.device,
        torch_version=torch.__version__,
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        gpu=torch.cuda.get_device_name() if cfg.device.startswith("cuda") else None,
        submitter=cfg.submitter,
        train_scenes=[sp.name],
        test_scenes=[sp.name],
        t_error_mean=t_err_mean,
        t_error_median=t_err_median,
        n=len(queries),
        success_rate=len(ok) / len(queries),
        cost_mean=statistics.fmean(cost[i] for i in ok) if ok else None,  # pyright: ignore[reportArgumentType]
        length_mean=statistics.fmean(length[i] for i in ok) if ok else None,  # pyright: ignore[reportArgumentType]
        spl=statistics.fmean(scores) if scores else None,
        cost_ratio_mean=statistics.fmean(ratios) if ratios else None,
        plan_seconds_mean=statistics.fmean(seconds),
        plan_seconds_p95=percentile(seconds, 95),
        success=success,
        converged=converged,
        cost=cost,
        length=length,
        plan_seconds=seconds,
        **training,
    )


def _field(
    cfg: RunConfig, suite: Suite, sp: SuiteProblem, problem: Problem, preset: str, seed: int
) -> tuple[TrainedField, dict[str, Any]]:
    """A trained field and its training fields of the record: a matching checkpoint if `reuse`, else a new run."""
    recipe = get_recipe(preset)
    recipe = replace(recipe, epochs=cfg.epochs or recipe.epochs, n_pairs=cfg.n_pairs or recipe.n_pairs)
    method = create(preset)
    ckpt = cfg.checkpoint_dir / suite.id / sp.name / preset / f"seed{seed}.pt"
    meta_path = ckpt.with_suffix(".json")
    expected = {"preset": preset, "config": _jsonable(asdict(method.cfg)), "recipe": asdict(recipe)}
    if cfg.reuse and ckpt.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in expected} == expected and meta.get("checkpoint_sha256") == sha256(ckpt):
            log.info("reusing %s", ckpt)
            return load(ckpt, cfg.device), meta["training"]
        log.info("retraining %s: its config, recipe or checksum does not match", ckpt)

    stats = _TrainStats()
    tc = ntfields.TrainConfig(
        n_pairs=recipe.n_pairs, batch_size=recipe.batch_size, batches_per_epoch=recipe.batches_per_epoch,
        micro_batch_size=recipe.micro_batch_size, seed=seed, device=cfg.device, out_dir=None,
    )  # fmt: skip
    t0 = time.perf_counter()
    trained = ntfields.train(method, problem, ntfields.Epochs(recipe.epochs), train_cfg=tc, callbacks=[stats])
    seconds = time.perf_counter() - t0
    save(ckpt, method, problem, trained.model, stats.epochs - 1)
    training = {
        "epochs": stats.epochs, "train_seconds": seconds, "n_pairs": recipe.n_pairs, "final_loss": stats.loss,
        "rollbacks": stats.rollbacks, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
    }  # fmt: skip
    meta_path.write_text(json.dumps({**expected, "checkpoint_sha256": sha256(ckpt), "training": training}) + "\n")
    trained.ckpt_path = ckpt
    return trained, training


class _TrainStats(Callback):
    """Rejected epochs never reach callbacks after the rollback, but every attempt starts, so rollbacks are the
    epochs started minus the epochs accepted."""

    def __init__(self) -> None:
        self.started = 0
        self.epochs = 0
        self.loss: float | None = None

    @property
    def rollbacks(self) -> int:
        return self.started - self.epochs

    def on_epoch_start(self, state: TrainState) -> None:
        self.started += 1

    def on_epoch_end(self, state: TrainState) -> None:
        self.epochs = state.epoch + 1
        self.loss = state.loss


def _planners(names: Sequence[str], method: Method[Any]) -> list[str]:
    out: list[str] = []
    for name in names:
        expanded = [method.default_planner] if name == "default" else sorted(PLANNERS) if name == "all" else [name]
        out += [p for p in expanded if p not in out]
    return out


def _package(method: Method[Any]) -> tuple[str, str]:
    top = type(method).__module__.split(".")[0]
    dist = packages_distributions().get(top, [top])[0]
    try:
        return dist, version(dist)
    except PackageNotFoundError:
        return dist, "unknown"


def _git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty.strip() else "")


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _jsonable(value: Any) -> Any:
    """Config dicts as JSON values (tuples to lists, non-finite floats to strings)."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        return str(value)
    return value
