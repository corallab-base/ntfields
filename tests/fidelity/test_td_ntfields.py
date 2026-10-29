from types import SimpleNamespace

import pytest
import torch

from ntfields import Epochs, Problem, SceneCondition, TrainedField, TrainState
from ntfields.data import PairDataset
from ntfields.envs import SphereBoxEnv
from ntfields.methods.td_ntfields import TDNTFields, TDNTFieldsConfig
from ntfields.methods.td_ntfields.loss import td_ntfields_loss
from ntfields.plan import mppi
from ntfields.speed import ClippedEased
from ntfields.train import Callback, fit
from tests.fidelity.ntrl import (
    NTRL_DEMO,
    import_ntrl,
    import_ntrl_planner,
    load_ref_checkpoint,
    ntfields_model_from_ref,
    ref_network,
    ref_train,
    requires_ntrl,
    to_batch,
    to_double,
    to_ntfields_state,
    ur5_batch,
    ur5_data,
)

UR5_CKPT = NTRL_DEMO / "Experiments" / "UR5" / "arm_08_22_18_25" / "Model_Epoch_03900_ValLoss_3.967947e-03.pt"


@requires_ntrl
def test_forward_matches_reference_random_weights():
    net_mod, _ = import_ntrl()
    torch.manual_seed(0)
    b = 0.2 * torch.randn(256, 6)
    ref = net_mod.NN("cpu", 6, b)
    ref.apply(ref.init_weights)
    with torch.no_grad():
        for gate in ref.gate:
            gate.weight.fill_(0.7)  # non-zero so the residual gates are exercised
    ref, b = to_double(ref, b)
    x = torch.rand(1000, 12, dtype=torch.float64) - 0.5
    torch.testing.assert_close(ntfields_model_from_ref(ref, b, 6)(x[:, :6], x[:, 6:], None), ref.out(x)[0][:, 0])


@requires_ntrl
def test_forward_matches_reference_ur5_checkpoint():
    """Level 1: ntrl's trained UR5 weights give the same T in ntfields."""
    ref, b = to_double(*load_ref_checkpoint(UR5_CKPT))
    x = torch.rand(2000, 12, dtype=torch.float64) - 0.5
    torch.testing.assert_close(ntfields_model_from_ref(ref, b, 6)(x[:, :6], x[:, 6:], None), ref.out(x)[0][:, 0])


@requires_ntrl
def test_loss_and_gradients_match_reference():
    """Level 2: same weights and batch give ntrl's `Function.Loss` value and parameter gradients."""
    ref, b = to_double(*load_ref_checkpoint(UR5_CKPT))
    _, function_module = import_ntrl()
    x, speed, normal = (t.double() for t in ur5_batch(2000))
    _, loss_n, _ = function_module.Function(".", "cpu", ref, 6).Loss(x, speed, normal, 1.0, 0.001, 1)
    loss_n.backward()

    model = ntfields_model_from_ref(ref, b, 6)
    out = td_ntfields_loss(model, to_batch(x, speed, normal), TDNTFieldsConfig.arm_ntrl())
    out.total.backward()

    torch.testing.assert_close(out.total, loss_n)
    ref_grads = to_ntfields_state({k: p.grad for k, p in ref.named_parameters() if p.grad is not None}, b)
    _assert_grads_close(model, ref_grads)


def _assert_grads_close(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, expected[name], msg=name)


@requires_ntrl
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the reference MPPI is CUDA-only")
def test_mppi_matches_reference_planner():
    """ntrl's `arm_plan_stat.py::MPPI` and ntfields' mppi give the same path from the same seed and weights."""
    ref, b = load_ref_checkpoint(UR5_CKPT)
    ref = ref.cuda()
    ref.B = ref.B.cuda()
    _, function_module = import_ntrl()
    plan_stat = import_ntrl_planner()
    womodel = SimpleNamespace(function=function_module.Function(".", "cuda", ref, 6))
    start, goal = torch.tensor([0.1, -0.2, 0.15, 0.0, 0.2, -0.1]), torch.tensor([-0.15, 0.1, -0.1, 0.2, -0.2, 0.1])

    torch.manual_seed(0)
    with torch.no_grad():
        ref_path, _ = plan_stat.MPPI(womodel, torch.cat([start, goal])[None].cuda())
    field = TrainedField(TDNTFields(TDNTFieldsConfig.arm_ntrl()), _problem(), ntfields_model_from_ref(ref, b, 6).cuda())
    torch.manual_seed(0)
    path = mppi(field, start.cuda(), goal.cuda()).path

    torch.testing.assert_close(path, torch.cat(ref_path))


def _problem() -> Problem:
    """A 6-D placeholder problem: planning only queries the field, never the environment."""
    return Problem(SphereBoxEnv.build(bounds=[(-0.5, 0.5)] * 6), ClippedEased())


@requires_ntrl
def test_short_training_matches_reference():
    """Level 3: 5 epochs (25 AdamW steps, with rollback and loss scaling) from the same init, data and seed.

    The reference reports the 5-batch loss sum / 4 and ntfields the mean, so losses are compared x 5/4.
    """
    epochs, seed = 5, 0
    torch.manual_seed(seed)
    b = 0.2 * torch.randn(128, 6)
    ref = ref_network(UR5_CKPT.parent, b)
    model = ntfields_model_from_ref(ref, b, 6)
    data = ur5_data()
    ref_losses = ref_train(ref, data, epochs, seed)

    x, s, normal = data[:, :12], data[:, 12:14], data[:, 14:]
    speed = s * s * (2 - s) * (2 - s)
    pairs = PairDataset(x[:, :6], x[:, 6:], speed[:, 0], speed[:, 1], normal[:, :6], normal[:, 6:])
    method = TDNTFields(TDNTFieldsConfig.arm_ntrl())
    losses: list[float] = []

    class Record(Callback):
        def on_epoch_end(self, state: TrainState) -> None:
            if not state.retry:
                losses.append(state.loss * 5 / 4)

    callbacks = [*method.callbacks(), Record()]
    generator = torch.Generator().manual_seed(seed)
    fit(method, model, pairs, SceneCondition(), Epochs(epochs), 2000, callbacks, generator, batches_per_epoch=5)

    torch.testing.assert_close(torch.tensor(losses), torch.tensor(ref_losses))
    _assert_drift_below(model, to_ntfields_state(dict(ref.state_dict()), b), 1e-4)


def _assert_drift_below(model: torch.nn.Module, expected: dict[str, torch.Tensor], bound: float) -> None:
    """Relative parameter drift ‖θ - θ_ref‖ / ‖θ_ref‖. Training amplifies rounding: nudging the reference's own init
    by 1e-7 (relative) gives 5e-5 drift after 5 epochs, while training moves the weights by 6e-2."""
    state = model.state_dict()
    drift = sum((state[n] - p).norm() ** 2 for n, p in expected.items()) ** 0.5
    scale = sum(p.norm() ** 2 for p in expected.values()) ** 0.5
    assert drift / scale < bound, f"relative drift {drift / scale:.2e}"
