import copy
import logging
import math
import random
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from ntfields.base import Method, SceneCondition, TrainState

if TYPE_CHECKING:
    from ntfields.data import PairDataset
    from ntfields.problem import Problem

log = logging.getLogger("ntfields")


class Callback:
    """Hooks into `fit`. Per epoch: on_epoch_start, then per batch on_after_backward (the batch's gradients are
    accumulated; before clipping and the optimizer step) and on_batch_end (after the step; `state.batch_loss` is
    set), then on_epoch_end, which may set `state.retry` (later callbacks then skip the epoch) or `state.stop`."""

    def on_train_start(self, state: TrainState) -> None: ...  # once, after the optimizer is built
    def on_epoch_start(self, state: TrainState) -> None: ...
    def on_after_backward(self, state: TrainState) -> None: ...
    def on_batch_end(self, state: TrainState) -> None: ...
    def on_epoch_end(self, state: TrainState) -> None: ...  # may set state.retry or state.stop
    def on_train_end(self, state: TrainState) -> None: ...


class LossRatioRollback(Callback):
    """Re-run an epoch from a random recent state when loss / previous loss leaves (0, threshold).

    Matches ntrl-demo: the queue holds start-of-epoch states of the last `queue_size` accepted epochs
    (excluding the current one), and the first previous loss is 1000. A state is the model's and the optimizer's,
    and every object in `state.stateful` (e.g. an `LRScheduler`'s scheduler).
    """

    def __init__(self, threshold: float, queue_size: int = 5, max_retries: int = 200, seed: int = 0) -> None:
        self.threshold = threshold
        self.max_retries = max_retries
        self.queue: deque[dict[str, Any]] = deque(maxlen=queue_size)
        self.prev_loss = 1000.0
        self.retries = 0
        self._current: dict[str, Any] | None = None
        self._rng = random.Random(seed)

    def on_epoch_start(self, state: TrainState) -> None:
        if self.retries:
            return
        snapshot = {
            "model": copy.deepcopy(state.model.state_dict()),
            "optim": copy.deepcopy(state.optimizer.state_dict()),
            "stateful": {k: copy.deepcopy(v.state_dict()) for k, v in state.stateful.items()},
        }
        self.queue.append(self._current or snapshot)
        self._current = snapshot

    def on_epoch_end(self, state: TrainState) -> None:
        ratio = state.loss / self.prev_loss
        if not 0 < ratio < self.threshold:
            self.retries += 1
            if self.retries < self.max_retries:
                snapshot = self._rng.choice(self.queue)
                state.model.load_state_dict(snapshot["model"])
                state.optimizer.load_state_dict(snapshot["optim"])
                for k, v in snapshot["stateful"].items():
                    state.stateful[k].load_state_dict(v)
                state.retry = True
                return
            log.warning("epoch %d: %d rollbacks, accepting loss ratio %.3f", state.epoch, self.retries, ratio)
        self.prev_loss = state.loss
        self.retries = 0


class LossScaleByPrevLoss(Callback):
    """Scales the loss by numerator / (sum of the last accepted epoch's batch losses), as ntrl-demo's `beta` does
    (numerator 4: its epoch loss is the 5-batch sum / 4)."""

    def __init__(self, numerator: float = 1.0) -> None:
        self.numerator = numerator

    def on_epoch_start(self, state: TrainState) -> None:
        if not math.isnan(state.prev_loss):  # at least one accepted epoch
            state.loss_scale = self.numerator / (state.batches * state.prev_loss)


class LRScheduler(Callback):
    """Steps a torch LR scheduler, built by `factory` from the optimizer at the start of training, after every batch
    (`interval="batch"`, e.g. OneCycleLR) or every epoch. It is registered in `state.stateful`, so a rollback
    restores it with the optimizer."""

    def __init__(
        self,
        factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        interval: Literal["batch", "epoch"] = "epoch",
    ) -> None:
        self.factory = factory
        self.interval = interval
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def on_train_start(self, state: TrainState) -> None:
        self.scheduler = self.factory(state.optimizer)
        state.stateful["lr_scheduler"] = self.scheduler

    def on_batch_end(self, state: TrainState) -> None:
        if self.interval == "batch" and self.scheduler is not None:
            self.scheduler.step()

    def on_epoch_end(self, state: TrainState) -> None:
        if self.interval == "epoch" and self.scheduler is not None:
            self.scheduler.step()


class Validation(Callback):
    """Every `every` epochs, the mean `method.loss` over `data`'s batches (no optimizer step), as `state.val_loss`
    and `state.val_terms`. Gradients stay enabled: physics-informed losses differentiate T. Put it after any
    rollback callback, so a rejected epoch is not validated."""

    def __init__(
        self,
        method: Method[Any],
        data: "PairDataset",
        scene: SceneCondition,
        batch_size: int,
        every: int = 1,
        seed: int = 0,
    ) -> None:
        self.method = method
        self.data = data
        self.scene = scene
        self.batch_size = batch_size
        self.every = every
        self.seed = seed

    def on_epoch_end(self, state: TrainState) -> None:
        if (state.epoch + 1) % self.every:
            return
        state.model.eval()
        total, terms, n = 0.0, defaultdict[str, float](float), 0
        generator = torch.Generator().manual_seed(self.seed)  # the same batches every time
        for batch in self.data.batches(self.batch_size, self.scene, generator):
            out = self.method.loss(state.model, batch, state)
            total += out.total.item()
            for k, v in out.terms.items():
                terms[k] += v
            n += 1
        state.model.train()
        state.val_loss = total / n
        state.val_terms = {k: v / n for k, v in terms.items()}


class Logger(Callback):
    def __init__(self, every: int = 100) -> None:
        self.every = every

    def on_epoch_end(self, state: TrainState) -> None:
        if (state.epoch + 1) % self.every == 0:
            terms = " ".join(f"{k}={v:.4g}" for k, v in state.terms.items())
            val = "" if math.isnan(state.val_loss) else f" val_loss={state.val_loss:.4g}"
            log.info("epoch %d loss=%.4g%s %s", state.epoch + 1, state.loss, val, terms)


class Checkpoint(Callback):
    """Saves at the end of training, and every `every` epochs if every > 0. A `{epoch}` in the path's file name
    keeps one file per save, numbered by the epochs completed (also the checkpoint's "epoch"), instead of overwriting
    one."""

    def __init__(self, path: Path, method: Method[Any], problem: "Problem", every: int = 0) -> None:
        self.path = path
        self.method = method
        self.problem = problem
        self.every = every

    def on_epoch_end(self, state: TrainState) -> None:
        if self.every and (state.epoch + 1) % self.every == 0:
            self._save(state, state.epoch + 1)  # state.epoch is still the finished epoch's index

    def on_train_end(self, state: TrainState) -> None:
        self._save(state, state.epoch)

    def _save(self, state: TrainState, epochs: int) -> None:
        from ntfields.checkpoint import save

        path = self.path.with_name(self.path.name.format(epoch=epochs))
        save(path, self.method, self.problem, state.model, epochs)
