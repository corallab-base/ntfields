from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ntfields.base import Method, MethodConfig
from ntfields.data import PairDataset, PairSource, Sampler, sample_pairs
from ntfields.field import TrainedField
from ntfields.problem import Problem
from ntfields.registry import create
from ntfields.train import Budget, Callback, Checkpoint, Logger, fit


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    n_pairs: int = 10_000
    batch_size: int = 512
    batches_per_epoch: int | None = None  # None: a full pass over the data
    micro_batch_size: int | None = None  # accumulate gradients over parts of this size (None: whole batches)
    seed: int = 0
    device: str = field(default_factory=_default_device)
    out_dir: Path | None = Path("runs")  # None disables checkpointing
    ckpt_every: int = 0  # 0: only at the end
    log_every: int = 100


def train(
    method: str | Method[Any],
    problem: Problem,
    budget: Budget,
    cfg: MethodConfig | None = None,
    train_cfg: TrainConfig | None = None,
    sampler: Sampler | None = None,
    data: PairDataset | None = None,
) -> TrainedField:
    """Train `method` (a registered name or preset, or an instance) on `problem`; `sampler` overrides the
    method's default pair sampler, and `data` (pre-built pairs) replaces sampling."""
    method = resolve_method(method, cfg)
    return train_with(method, method, problem, budget, train_cfg or TrainConfig(), sampler, data)


def resolve_method(method: str | Method[Any], cfg: MethodConfig | None = None) -> Method[Any]:
    """A registered method or preset by name (with `cfg`), or an instance as is."""
    if isinstance(method, str):
        return create(method, cfg)
    if cfg is not None:
        raise ValueError("pass cfg only together with a method name")
    return method


def train_with(
    method: Method[Any],
    trainer: Method[Any],
    problem: Problem,
    budget: Budget,
    tc: TrainConfig,
    sampler: Sampler | None = None,
    data: PairDataset | None = None,
    source: Callable[[PairDataset], PairSource] = lambda pairs: pairs,
) -> TrainedField:
    """`train`, with the loss and callbacks of `trainer` (e.g. a Distillation of `method`) and batches from
    `source(pairs)`; the model is built, checkpointed and returned as `method`'s."""
    if data is not None and sampler is not None:
        raise ValueError("pass either sampler or data")
    if data is None:
        data = sample_pairs(problem, tc.n_pairs, sampler or method.default_sampler(), tc.seed, tc.device)

    # all random draws come from the global RNG in a fixed order (a data permutation, the init, then every shuffle),
    # so a seed fixes the whole run
    torch.manual_seed(tc.seed)
    data = data[torch.randperm(len(data))].to(tc.device)
    model = method.build_model(problem).to(tc.device)
    callbacks: list[Callback] = [*trainer.callbacks(), Logger(tc.log_every)]
    ckpt_path = None
    if tc.out_dir is not None:
        ckpt_path = tc.out_dir / method.name / datetime.now().strftime("%Y%m%d-%H%M%S") / "ckpt.pt"
        callbacks.append(Checkpoint(ckpt_path, method, problem, tc.ckpt_every))

    scene = problem.env.condition()
    fit(
        trainer,
        model,
        source(data),
        scene,
        budget,
        tc.batch_size,
        callbacks,
        torch.default_generator,
        tc.batches_per_epoch,
        tc.micro_batch_size,
    )
    return TrainedField(method, problem, model.eval(), ckpt_path)
