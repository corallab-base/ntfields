import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import islice
from typing import Any

import torch

from ntfields.base import Batch, Method, SceneCondition, TrainState, TravelTimeModel
from ntfields.data import PairDataset
from ntfields.train.budget import Budget
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
) -> TrainState:
    """Shared training loop; all method-specific behaviour lives in `method.loss` and callbacks."""
    state = TrainState(model=model, optimizer=method.optimizer(model))
    start = time.monotonic()
    while not state.stop and not budget.exhausted(state.epoch, time.monotonic() - start):
        for cb in callbacks:
            cb.on_epoch_start(state)
        _run_epoch(method, state, islice(data.batches(batch_size, scene, generator), batches_per_epoch))
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


def _run_epoch(method: Method[Any], state: TrainState, batches: Iterable[Batch]) -> None:
    state.model.train()
    total, terms, n = 0.0, defaultdict[str, float](float), 0
    for batch in batches:
        out = method.loss(state.model, batch, state)
        state.optimizer.zero_grad()
        (state.loss_scale * out.total).backward()
        if method.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), method.cfg.grad_clip_norm)
        state.optimizer.step()
        total += out.total.item()
        for k, v in out.terms.items():
            terms[k] += v
        n += 1
    state.loss = total / n
    state.batches = n
    state.terms = {k: v / n for k, v in terms.items()}
