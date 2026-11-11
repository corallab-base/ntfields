from dataclasses import fields
from pathlib import Path

import pytest
import torch

import ntfields
from ntfields import Epochs, Problem, SceneCondition, TrainConfig
from ntfields.data import PairDataset, sample_pairs
from ntfields.methods.euclidean import Euclidean, EuclideanConfig
from ntfields.train import Callback, LossRatioRollback, LossScaleByPrevLoss, fit


def test_train_fits_scale_and_round_trips(problem: Problem, tmp_path: Path):
    tc = TrainConfig(n_pairs=256, batch_size=64, device="cpu", out_dir=tmp_path)
    field = ntfields.train("euclidean", problem, ntfields.Epochs(30), train_cfg=tc)
    assert field.ckpt_path is not None and field.ckpt_path.exists()
    assert field.model.get_parameter("log_scale") > 0  # S* <= 1, so the eikonal fit needs c > 1

    q = torch.tensor([[-0.8, -0.8], [0.8, 0.8]])
    loaded = ntfields.load(field.ckpt_path)
    torch.testing.assert_close(loaded.travel_time(q[:1], q[1:]), field.travel_time(q[:1], q[1:]))
    assert loaded.method.cfg == field.method.cfg


def test_rollback_restores_previous_state(problem: Problem):
    method = Euclidean(EuclideanConfig())
    model = method.build_model(problem)
    state = ntfields.TrainState(model=model, optimizer=method.optimizer(model))
    rollback = LossRatioRollback(threshold=1.2)

    rollback.on_epoch_start(state)
    state.loss = 1.0
    rollback.on_epoch_end(state)
    assert not state.retry

    rollback.on_epoch_start(state)
    before = model.state_dict()["log_scale"].clone()
    with torch.no_grad():
        model.get_parameter("log_scale").add_(5.0)
    state.loss = 2.0
    rollback.on_epoch_end(state)
    assert state.retry
    torch.testing.assert_close(model.state_dict()["log_scale"], before)


def test_cfg_with_method_instance_is_rejected(problem: Problem):
    with pytest.raises(ValueError):
        ntfields.train(Euclidean(EuclideanConfig()), problem, ntfields.Epochs(1), cfg=EuclideanConfig())


def test_registry():
    assert "euclidean" in ntfields.available()
    with pytest.raises(KeyError):
        ntfields.get_method("nope")


def test_batches_per_epoch_and_loss_scale(problem: Problem):
    method = Euclidean(EuclideanConfig())
    model = method.build_model(problem)
    seen: list[tuple[float, float]] = []

    class Record(Callback):
        def on_epoch_start(self, state: ntfields.TrainState) -> None:
            seen.append((state.loss_scale, state.prev_loss))

    callbacks = [LossScaleByPrevLoss(), Record()]
    state = fit(method, model, sample_pairs(problem, 256), SceneCondition(), ntfields.Epochs(3), 32, callbacks, None, 2)
    assert state.epoch == 3
    assert seen[0][0] == 1.0
    assert all(scale == pytest.approx(1 / (2 * prev)) for scale, prev in seen[1:])  # 1 / Σ of 2 batch losses


def test_micro_batches_give_the_full_batch_gradient(problem: Problem):
    """Accumulating over parts of 40 gives the same training as whole batches of 96 (float64)."""
    from ntfields.methods.pntfields import PNTFields, PNTFieldsConfig

    pairs = sample_pairs(problem, 96)
    data = PairDataset(*(getattr(pairs, f.name).double() for f in fields(pairs)))
    states = []
    for micro in (None, 40):
        method = PNTFields(PNTFieldsConfig(hidden=16))
        torch.manual_seed(1)
        model = method.build_model(problem).double()
        g = torch.Generator().manual_seed(2)
        states.append(fit(method, model, data, SceneCondition(), Epochs(2), 96, (), g, micro_batch_size=micro))
    assert states[0].loss == pytest.approx(states[1].loss, rel=1e-12)
    for a, b in zip(states[0].model.parameters(), states[1].model.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_zero_loss_is_accepted_and_keeps_the_loss_scale():
    model = Euclidean(EuclideanConfig()).build_model(None)  # pyright: ignore[reportArgumentType]
    state = ntfields.TrainState(model, torch.optim.SGD(model.parameters(), lr=0.1), loss=0.5, batches=2)
    rollback, scale = LossRatioRollback(1.2), LossScaleByPrevLoss(4.0)
    for loss in (0.5, 0.0, 0.0):  # a perfect fit, e.g. constant speed in an empty scene
        rollback.on_epoch_start(state)
        state.loss = loss
        rollback.on_epoch_end(state)
        assert not state.retry
        state.prev_loss = state.loss
        scale.on_epoch_start(state)
        assert state.loss_scale == pytest.approx(4.0)  # still from the last nonzero epoch: 4 / (2 · 0.5)


def test_lone_nan_at_the_retry_cap_does_not_stop_training():
    model = Euclidean(EuclideanConfig()).build_model(None)  # pyright: ignore[reportArgumentType]
    state = ntfields.TrainState(model, torch.optim.SGD(model.parameters(), lr=0.1))
    rollback = LossRatioRollback(1.2, max_retries=3)
    rollback.on_epoch_start(state)
    state.loss = 0.5
    rollback.on_epoch_end(state)
    rollback.on_epoch_start(state)
    for loss in (2.0, 2.0, float("nan")):  # the last retry is the one to go NaN
        state.loss, state.retry = loss, False
        rollback.on_epoch_end(state)
        assert state.retry and not state.stop
    state.loss, state.retry = 2.0, False  # the next finite loss is accepted, as at the cap
    rollback.on_epoch_end(state)
    assert not state.retry and rollback.retries == 0 and rollback.nonfinite == 0


def test_diverged_loss_stops_at_the_last_accepted_state():
    model = Euclidean(EuclideanConfig()).build_model(None)  # pyright: ignore[reportArgumentType]
    state = ntfields.TrainState(model, torch.optim.SGD(model.parameters(), lr=0.1))
    rollback = LossRatioRollback(1.2, max_retries=3)
    rollback.on_epoch_start(state)
    state.loss = 0.5
    rollback.on_epoch_end(state)
    good = {k: v.clone() for k, v in model.state_dict().items()}
    rollback.on_epoch_start(state)
    for _ in range(3):  # every retry diverges
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))
        state.loss, state.retry = float("nan"), False
        rollback.on_epoch_end(state)
        rollback.on_epoch_start(state)
    assert state.stop and state.retry and state.loss == 0.5
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, good[k])
