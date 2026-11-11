import copy
import random
from typing import Any

from ntfields.base import TrainState
from ntfields.train import Callback


class RetryRollback(Callback):
    """The reference's rollback: when loss / previous loss leaves (0, threshold), restore a random state from the
    start of one of the last `queue_size` accepted epochs (model, optimizer and LR schedule) and re-run the epoch,
    at most `max_attempts - 1` times before accepting it anyway. The first epoch is always accepted, and with an
    empty queue a retry re-runs from where it is."""

    def __init__(self, threshold: float, max_attempts: int = 3, queue_size: int = 5, seed: int = 0) -> None:
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.queue_size = queue_size
        self.queue: list[dict[str, Any]] = []
        self.prev_loss = float("inf")
        self.attempts = 0
        self._current: dict[str, Any] | None = None
        self._rng = random.Random(seed)

    def on_epoch_start(self, state: TrainState) -> None:
        if not self.attempts:
            self._current = _snapshot(state)

    def on_epoch_end(self, state: TrainState) -> None:
        ratio = state.loss / self.prev_loss
        if state.epoch > 0 and not 0 < ratio < self.threshold:
            self.attempts += 1
            if self.attempts < self.max_attempts:
                if self.queue:
                    _restore(state, self._rng.choice(self.queue))
                self._current = _snapshot(state)
                state.retry = True
                return
        else:
            assert self._current is not None
            self.queue = [*self.queue, self._current][-self.queue_size :]
        self.prev_loss = state.loss
        self.attempts = 0


def _snapshot(state: TrainState) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(state.model.state_dict()),
        "optim": copy.deepcopy(state.optimizer.state_dict()),
        "sched": None if state.scheduler is None else copy.deepcopy(state.scheduler.state_dict()),
    }


def _restore(state: TrainState, snapshot: dict[str, Any]) -> None:
    state.model.load_state_dict(snapshot["model"])
    state.optimizer.load_state_dict(snapshot["optim"])
    if state.scheduler is not None:
        state.scheduler.load_state_dict(snapshot["sched"])
