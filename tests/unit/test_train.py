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
