import pytest
import torch

from ntfields import Epochs, Problem, SceneCondition, TrainedField, TrainState
from ntfields.base import Batch
from ntfields.data import PairDataset
from ntfields.envs import SphereBoxEnv
from ntfields.methods.cntfields import CNTFields, CNTFieldsConfig
from ntfields.methods.cntfields.loss import cntfields_loss
from ntfields.nn import FactorizedTauModel
from ntfields.plan import gradient_bidirectional
from ntfields.speed import Clipped
from ntfields.speed.clipped_eased import ease
from ntfields.train import Callback, fit
from tests.fidelity.cntf import DIM, ref_model, ref_train, requires_cntfields
from tests.fidelity.pntf import to_ntfields_state


def _ours(ref: object, b: torch.Tensor, cfg: CNTFieldsConfig | None = None) -> FactorizedTauModel:
    cfg = cfg or CNTFieldsConfig()
    model = FactorizedTauModel(DIM, cfg.hidden, cfg.encoder_blocks, cfg.generator_blocks, cfg.fourier_features)
    model.to(b.dtype).load_state_dict(to_ntfields_state(dict(ref.network.state_dict()), b))  # type: ignore[attr-defined]
    return model


def _points(n: int, seed: int) -> torch.Tensor:
    return torch.rand(n, 2 * DIM, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) - 0.5


@requires_cntfields
def test_tau_gradient_and_laplacian_match_reference():
    """Level 1: τ, ∇τ and the Hessian diagonal against the reference's forward mode (`out_laplace`)."""
    torch.manual_seed(0)
    ref, b = ref_model()
    x = _points(500, 1)
    tau, dtau, ltau, _ = ref.network.out_laplace(x)
    ours = _ours(ref, b).tau_jet(x[:, :DIM], x[:, DIM:])
    torch.testing.assert_close(ours[0], tau[:, 0])
    torch.testing.assert_close(ours[1], dtau)
    torch.testing.assert_close(ours[2], ltau)


@requires_cntfields
def test_network_and_travel_time_match_reference():
    """Level 1: the plain forward pass and T = ‖d‖/τ."""
    torch.manual_seed(0)
    ref, b = ref_model()
    x = _points(500, 2)
    ours = _ours(ref, b)
    torch.testing.assert_close(ours.tau(x[:, :DIM], x[:, DIM:]), ref.network(x)[:, 0])
    torch.testing.assert_close(ours(x[:, :DIM], x[:, DIM:], SceneCondition()), ref.travel_time(x))


@requires_cntfields
@pytest.mark.parametrize(("epoch", "alpha"), [(0, 0.5), (4500, 0.95)])
def test_loss_and_gradients_match_reference(epoch: int, alpha: float):
    """Level 2: same weights and batch give the reference's `CNTFields.loss` value and parameter gradients."""
    torch.manual_seed(0)
    ref, b = ref_model()
    cfg = CNTFieldsConfig()
    assert cfg.alpha(epoch) == pytest.approx(alpha)
    x = _points(1000, 3)
    speed = 0.05 + 0.95 * torch.rand(1000, 2, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    a = cfg.alpha(epoch)
    ref_loss = ref.loss(x, a * speed + 1 - a, torch.zeros_like(x))
    ref_loss.backward()

    model = _ours(ref, b)
    batch = Batch(x[:, :DIM], x[:, DIM:], speed[:, 0], speed[:, 1], torch.zeros(1), torch.zeros(1), SceneCondition())
    state = TrainState(model=model, optimizer=torch.optim.SGD(model.parameters()), epoch=epoch)
    out = cntfields_loss(model, batch, state, cfg)
    out.total.backward()

    torch.testing.assert_close(out.total, ref_loss)
    grads = {k: p.grad for k, p in ref.network.named_parameters() if p.grad is not None}
    expected = to_ntfields_state(grads, b)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, expected[name], msg=name)


@requires_cntfields
@pytest.mark.parametrize(("threshold", "batch", "n"), [(1.2, 500, 3000), (1.0, 40, 400)])
def test_short_training_matches_reference(threshold: float, batch: int, n: int):
    """Level 3: 12 epochs from the same init, data and seed. Raw speeds are eased by the speed model here and inside
    the reference's loop there. The loss falls every epoch this early, so the reference's 1.2 never rolls back; small
    noisy batches with threshold 1.0 do."""
    epochs, seed = 12, 0
    torch.manual_seed(seed)
    ref, b = ref_model()
    model = _ours(ref, b)
    x = _points(n, 5)
    raw = 0.05 + 0.95 * torch.rand(n, 2, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    stats: dict[str, int] = {}
    ref_losses = ref_train(ref, torch.cat([x, raw], dim=1), epochs, seed, batch, threshold, stats)

    zeros = torch.zeros(n, DIM, dtype=torch.float64)
    pairs = PairDataset(x[:, :DIM], x[:, DIM:], ease(raw[:, 0]), ease(raw[:, 1]), zeros, zeros)
    method = CNTFields(CNTFieldsConfig(rollback_threshold=threshold))
    losses: list[float] = []

    class Record(Callback):
        def on_epoch_end(self, state: TrainState) -> None:
            if not state.retry:
                losses.append(state.loss)

    generator = torch.Generator().manual_seed(seed)
    callbacks = [*method.callbacks(), Record()]
    fit(method, model, pairs, SceneCondition(), Epochs(epochs), batch, callbacks, generator, batches_per_epoch=5)

    assert len(ref_losses) == epochs
    assert threshold > 1 or stats.get("rollbacks", 0) > 0
    # 60 steps on noisy batches amplify float64 rounding (5e-16 at epoch 1, 7e-8 at epoch 12, without a jump at any
    # rollback), so the rollback run is compared to a tolerance; the other holds to assert_close's defaults
    rtol, atol = (1e-7, 1e-7) if threshold > 1 else (1e-6, 1e-5)
    torch.testing.assert_close(
        torch.tensor(losses, dtype=torch.float64), torch.tensor(ref_losses, dtype=torch.float64), rtol=rtol, atol=atol
    )
    expected = to_ntfields_state(dict(ref.network.state_dict()), b)
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, expected[name], rtol=rtol, atol=atol, msg=name)


@requires_cntfields
def test_bidirectional_planner_matches_reference():
    """`gradient_bidirectional` with the reference's settings against `predict_trajectory` without its straight-line
    bridge. The reference plans in float32."""
    torch.manual_seed(0)
    ref, b = ref_model(torch.float32)
    start = torch.tensor([0.1, -0.2, 0.15, 0.0, 0.2, -0.1])
    goal = torch.tensor([-0.15, 0.1, -0.1, 0.2, -0.2, 0.1])
    ref_path = torch.from_numpy(ref.predict_trajectory(start.tolist(), goal.tolist(), interpolate=False))

    problem = Problem(SphereBoxEnv.build(bounds=[(-0.5, 0.5)] * DIM), Clipped())
    field = TrainedField(CNTFields(CNTFieldsConfig()), problem, _ours(ref, b))
    result = gradient_bidirectional(field, start, goal, beta=0.01, goal_tol=0.03, max_steps=101)
    torch.testing.assert_close(result.path, ref_path, rtol=1e-4, atol=1e-5)
