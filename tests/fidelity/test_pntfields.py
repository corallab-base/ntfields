from typing import Any

import pytest
import torch

from ntfields import Epochs, Problem, SceneCondition, TrainedField, TrainState
from ntfields.base import Batch
from ntfields.data import PairDataset
from ntfields.envs import SphereBoxEnv
from ntfields.methods.pntfields import PNTFields, PNTFieldsConfig
from ntfields.methods.pntfields.loss import pntfields_loss
from ntfields.plan import PlanResult, gradient_bidirectional
from ntfields.speed import Clipped
from ntfields.train import Callback, fit
from tests.fidelity.pntf import (
    DIM,
    UR5_CKPT,
    import_pntfields,
    ntfields_model_from_ref,
    random_pairs,
    ref_model,
    ref_network,
    ref_train,
    requires_pntfields,
    to_ntfields_state,
)


def _ref_net(seed: int = 0) -> tuple[Any, torch.Tensor]:
    import_pntfields()
    torch.manual_seed(seed)
    b = 0.5 * torch.randn(128, DIM)
    return ref_network(b), b.double()


@requires_pntfields
def test_tau_gradient_and_laplacian_match_reference():
    """Level 1: τ, ∇τ and the Hessian diagonal against the reference's forward mode (`out_laplace`)."""
    ref, b = _ref_net()
    x = torch.rand(500, 2 * DIM, generator=torch.Generator().manual_seed(1), dtype=torch.float64) - 0.5
    tau_ref, dtau_ref, ltau_ref, _ = ref.out_laplace(x)

    model = ntfields_model_from_ref(ref, b)
    tau, dtau, hess = model.tau_jet(x[:, :DIM], x[:, DIM:])

    torch.testing.assert_close(tau, tau_ref[:, 0])
    torch.testing.assert_close(model.tau(x[:, :DIM], x[:, DIM:]), ref.out(x)[0][:, 0])
    torch.testing.assert_close(dtau, dtau_ref)
    torch.testing.assert_close(hess, ltau_ref)


@requires_pntfields
def test_travel_time_matches_reference():
    ref, b = _ref_net()
    x = torch.rand(500, 2 * DIM, generator=torch.Generator().manual_seed(2), dtype=torch.float64) - 0.5
    t = ntfields_model_from_ref(ref, b)(x[:, :DIM], x[:, DIM:], SceneCondition())
    torch.testing.assert_close(t, ref_model(ref).TravelTimes(x))


requires_ur5 = pytest.mark.skipif(
    not UR5_CKPT.is_file(), reason="needs the reference's pretrained UR5 model (Google Drive)"
)


def _ur5_net() -> tuple[Any, torch.Tensor]:
    module = import_pntfields()
    ckpt = torch.load(UR5_CKPT, map_location="cpu", weights_only=False)
    b = ckpt["B_state_dict"].double()
    ref = module.NN("cpu", DIM, b).double()
    ref.load_state_dict(ckpt["model_state_dict"])
    ref.B = ref.B.double()
    return ref, b


@requires_pntfields
@requires_ur5
def test_forward_matches_reference_ur5_checkpoint():
    """Level 1 with the reference's trained UR5 weights."""
    ref, b = _ur5_net()
    x = torch.rand(2000, 2 * DIM, dtype=torch.float64) - 0.5
    tau_ref, dtau_ref, ltau_ref, _ = ref.out_laplace(x)
    model = ntfields_model_from_ref(ref, b)
    torch.testing.assert_close(model.tau(x[:, :DIM], x[:, DIM:]), ref.out(x)[0][:, 0])
    for ours, theirs in zip(model.tau_jet(x[:, :DIM], x[:, DIM:]), (tau_ref[:, 0], dtau_ref, ltau_ref), strict=True):
        torch.testing.assert_close(ours, theirs)


@requires_pntfields
@requires_ur5
def test_loss_and_gradients_match_reference_ur5_checkpoint():
    """Level 2 with the trained UR5 weights, at the final α = 1.05."""
    ref, b = _ur5_net()
    _assert_loss_and_grads_match(ref, b, 9500)


@requires_pntfields
@requires_ur5
def test_bidirectional_planner_matches_reference_ur5_checkpoint():
    """test/arm_plan.py's query with its trained model: the same path, and it converges."""
    ref, b = _ur5_net()
    base = torch.tensor([0, -0.5 * torch.pi, 0.0, -0.5 * torch.pi, 0.0, 0.0], dtype=torch.float64)
    start = (torch.tensor([-2.2, 0.4, 1.1, 0.5, -0.5, 0.9], dtype=torch.float64) + base) / (2 * torch.pi)
    goal = (torch.tensor([-1.3, 0.4, 1.1, 0.5, -0.5, 0.0], dtype=torch.float64) + base) / (2 * torch.pi)
    ref_path, converged = _ref_plan(ref_model(ref), start, goal, 301)
    assert converged

    result = _plan(ref, b, start, goal, 301)
    assert result.success
    torch.testing.assert_close(result.path, ref_path)


@requires_pntfields
@pytest.mark.parametrize(("epoch", "alpha"), [(0, 0.5), (5000, None), (9500, 1.05)])
def test_loss_and_gradients_match_reference(epoch: int, alpha: float | None):
    """Level 2: same weights and batch give the reference's `Model.Loss` value and parameter gradients."""
    if alpha is not None:
        assert PNTFieldsConfig().alpha(epoch) == alpha
    _assert_loss_and_grads_match(*_ref_net(), epoch)


def _assert_loss_and_grads_match(ref: Any, b: torch.Tensor, epoch: int) -> None:
    cfg = PNTFieldsConfig()
    a = cfg.alpha(epoch)
    x, speed = random_pairs(1000, torch.Generator().manual_seed(3))
    _, loss_n, _ = ref_model(ref).Loss(x, a * speed + 1 - a, 1.0, cfg.viscosity)
    loss_n.backward()

    model = ntfields_model_from_ref(ref, b)
    batch = Batch(x[:, :DIM], x[:, DIM:], speed[:, 0], speed[:, 1], torch.zeros(1), torch.zeros(1), SceneCondition())
    state = TrainState(model=model, optimizer=torch.optim.SGD(model.parameters()), epoch=epoch)
    out = pntfields_loss(model, batch, state, cfg)
    out.total.backward()

    torch.testing.assert_close(out.total, loss_n)
    ref_grads = to_ntfields_state({k: p.grad for k, p in ref.named_parameters() if p.grad is not None}, b)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, ref_grads[name], msg=name)


@requires_pntfields
def test_short_training_matches_reference():
    """Level 3: 5 epochs (30 AdamW steps, with rollback and loss scaling) from the same init, data and seed.

    The reference reports Σ₆ batch losses / len(loader) and ntfields the mean, so losses are compared x 6/12.
    """
    epochs, seed, batch, n = 5, 0, 500, 6000
    ref, b = _ref_net(seed)
    model = ntfields_model_from_ref(ref, b)
    x, speed = random_pairs(n, torch.Generator().manual_seed(4))
    ref_losses = ref_train(ref, torch.cat([x, speed], dim=1), epochs, seed, batch)

    zeros = torch.zeros(n, DIM, dtype=torch.float64)
    pairs = PairDataset(x[:, :DIM], x[:, DIM:], speed[:, 0], speed[:, 1], zeros, zeros)
    method = PNTFields(PNTFieldsConfig(loss_scale_numerator=n / batch))
    losses: list[float] = []

    class Record(Callback):
        def on_epoch_end(self, state: TrainState) -> None:
            if not state.retry:
                losses.append(state.loss * 6 / (n / batch))

    callbacks = [*method.callbacks(), Record()]
    generator = torch.Generator().manual_seed(seed)
    fit(method, model, pairs, SceneCondition(), Epochs(epochs), batch, callbacks, generator, batches_per_epoch=6)

    torch.testing.assert_close(torch.tensor(losses, dtype=torch.float64), torch.tensor(ref_losses, dtype=torch.float64))
    expected = to_ntfields_state(dict(ref.state_dict()), b)
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, expected[name], msg=name)


@requires_pntfields
def test_bidirectional_planner_matches_reference():
    """`gradient_bidirectional` against test/arm_plan.py's loop on `Model.Gradient` (β 0.015, d_g 0.03), random
    weights, capped at 40 steps: random weights need not converge."""
    ref, b = _ref_net()
    start = torch.tensor([0.1, -0.2, 0.15, 0.0, 0.2, -0.1], dtype=torch.float64)
    goal = torch.tensor([-0.15, 0.1, -0.1, 0.2, -0.2, 0.1], dtype=torch.float64)
    ref_path, _ = _ref_plan(ref_model(ref), start, goal, 40)
    torch.testing.assert_close(_plan(ref, b, start, goal, 40).path, ref_path)


def _ref_plan(model: Any, start: torch.Tensor, goal: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, bool]:
    """test/arm_plan.py's loop: step both ends by 0.015·`Model.Gradient` until they are within 0.03."""
    xp = torch.cat([start, goal])[None]
    fwd, bwd = [xp[:, :DIM]], [xp[:, DIM:]]
    for _ in range(max_steps):
        if torch.norm(xp[:, DIM:] - xp[:, :DIM]) <= 0.03:
            break
        xp = xp + 0.015 * model.Gradient(xp.clone())
        fwd.append(xp[:, :DIM])
        bwd.append(xp[:, DIM:])
    converged = bool(torch.norm(xp[:, DIM:] - xp[:, :DIM]) <= 0.03)
    return torch.cat([*fwd, *reversed(bwd)]).detach(), converged


def _plan(ref: Any, b: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, max_steps: int) -> PlanResult:
    problem = Problem(SphereBoxEnv.build(bounds=[(-0.5, 0.5)] * DIM), Clipped())
    field = TrainedField(PNTFields(PNTFieldsConfig()), problem, ntfields_model_from_ref(ref, b))
    return gradient_bidirectional(field, start, goal, beta=0.015, goal_tol=0.03, max_steps=max_steps)
