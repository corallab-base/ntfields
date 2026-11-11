import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import fields
from itertools import islice
from typing import Any

import torch

from ntfields.base import Batch, Method, SceneCondition, TrainState, TravelTimeModel
from ntfields.data import PairDataset
from ntfields.train.budget import Budget, Epochs
from ntfields.train.callbacks import Callback


def fit(
    method: Method[Any],
    model: TravelTimeModel,
    data: PairDataset,
    scene: SceneCondition,
    budget: Budget,
    batch_size: int,
    callbacks: Sequence[Callback] = (),
    generator: torch.Generator | None = None,
    batches_per_epoch: int | None = None,
    micro_batch_size: int | None = None,
) -> TrainState:
    """Shared training loop; all method-specific behaviour lives in `method.loss` and callbacks.

    `micro_batch_size` splits each batch and accumulates the gradients, weighting each part by its share of the
    batch; for a batch-mean loss this is the full batch's gradient (up to rounding) in less memory.
    """
    optimizer = method.optimizer(model)
    steps_per_epoch = batches_per_epoch or math.ceil(len(data) / batch_size)
    epochs = budget.n if isinstance(budget, Epochs) else None
    state = TrainState(model=model, optimizer=optimizer, scheduler=method.scheduler(optimizer, steps_per_epoch, epochs))
    start = time.monotonic()
    while not state.stop and not budget.exhausted(state.epoch, time.monotonic() - start):
        for cb in callbacks:
            cb.on_epoch_start(state)
        batches = islice(data.batches(batch_size, scene, generator), batches_per_epoch)
        _run_epoch(method, state, batches, micro_batch_size)
        for cb in callbacks:
            cb.on_epoch_end(state)
            if state.retry:  # later callbacks never see a rejected epoch
                break
        if state.retry:
            state.retry = False
        else:
            state.prev_loss = state.loss
            state.epoch += 1
    for cb in callbacks:
        cb.on_train_end(state)
    return state


def _run_epoch(method: Method[Any], state: TrainState, batches: Iterable[Batch], micro_batch_size: int | None) -> None:
    state.model.train()
    total, terms, n = 0.0, defaultdict[str, float](float), 0
    for batch in batches:
        state.optimizer.zero_grad()
        size = len(batch.q_s)
        for part in _split(batch, micro_batch_size or size):
            w = len(part.q_s) / size
            out = method.loss(state.model, part, state)
            (state.loss_scale * w * out.total).backward()
            total += w * out.total.item()
            for k, v in out.terms.items():
                terms[k] += w * v
        if method.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), method.cfg.grad_clip_norm)
        state.optimizer.step()
        if state.scheduler is not None:
            state.scheduler.step()
        n += 1
    state.loss = total / n
    state.batches = n
    state.terms = {k: v / n for k, v in terms.items()}


def _split(batch: Batch, size: int) -> list[Batch]:
    if size >= len(batch.q_s):
        return [batch]
    parts = {
        f.name: v.split(size)
        for f in fields(batch)
        if f.name not in ("scene", "extras") and (v := getattr(batch, f.name)) is not None
    }
    extras = {k: v.split(size) for k, v in batch.extras.items()}
    return [
        Batch(**{k: v[i] for k, v in parts.items()}, scene=batch.scene, extras={k: v[i] for k, v in extras.items()})
        for i in range(len(parts["q_s"]))
    ]
