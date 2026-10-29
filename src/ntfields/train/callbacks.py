import copy
import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ntfields.base import Method, TrainState

if TYPE_CHECKING:
    from ntfields.problem import Problem

log = logging.getLogger("ntfields")


class Callback:
    def on_epoch_start(self, state: TrainState) -> None: ...
    def on_epoch_end(self, state: TrainState) -> None: ...  # may set state.retry or state.stop
    def on_train_end(self, state: TrainState) -> None: ...


class LossRatioRollback(Callback):
    """Re-run an epoch from a random recent state when loss / previous loss leaves (0, threshold).

    Matches ntrl-demo: the queue holds start-of-epoch states of the last `queue_size` accepted epochs
    (excluding the current one), and the first previous loss is 1000.
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


class Logger(Callback):
    def __init__(self, every: int = 100) -> None:
        self.every = every

    def on_epoch_end(self, state: TrainState) -> None:
        if (state.epoch + 1) % self.every == 0:
            terms = " ".join(f"{k}={v:.4g}" for k, v in state.terms.items())
            log.info("epoch %d loss=%.4g %s", state.epoch + 1, state.loss, terms)


class Checkpoint(Callback):
    """Saves at the end of training, and every `every` epochs if every > 0."""

    def __init__(self, path: Path, method: Method[Any], problem: "Problem", every: int = 0) -> None:
        self.path = path
        self.method = method
        self.problem = problem
        self.every = every

    def on_epoch_end(self, state: TrainState) -> None:
        if self.every and (state.epoch + 1) % self.every == 0:
            self._save(state)

    def on_train_end(self, state: TrainState) -> None:
        self._save(state)

    def _save(self, state: TrainState) -> None:
        from ntfields.checkpoint import save

        save(self.path, self.method, self.problem, state.model, state.epoch)
