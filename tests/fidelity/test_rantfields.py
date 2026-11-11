"""RaNTFields against the reference (nap) on the paper's own checkpoints and datasets.

NTFIELDS_NAP=../nap uv run pytest tests/fidelity/test_rantfields.py      # the sim tests need `--group sim`
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ntfields.base import Batch, SceneCondition
from ntfields.data import sample_valid
from ntfields.methods.rantfields.loss import rantfields_loss
from ntfields.methods.rantfields.model import RaNTFieldsModel
from tests.fidelity.nap_repo import NAP, RUNS, TABLE_RESULTS, TABLE_RUNS, import_nap_import, reference, requires_nap

pytestmark = requires_nap


@pytest.mark.parametrize("run", sorted(RUNS))
def test_forward_loss_and_gradients(run: str, tmp_path: Path):
    """Levels 1-2: the reference's weights give the same T, randers_5 loss and parameter gradients on its own
    training batch (its run's dataset, retargeting and flow transforms applied)."""
    path, epoch = RUNS[run]
    reference("loss", NAP / path, epoch, tmp_path / "ref.pt")
    r = torch.load(tmp_path / "ref.pt", weights_only=False)
    field = import_nap_import().load_nap_run(NAP / path, epoch)
    model = field.model
    assert isinstance(model, RaNTFieldsModel)
    z = torch.zeros_like(r["goal"])
    batch = Batch(
        r["goal"], r["start"], r["speed"], r["speed"], z, z, SceneCondition(), metric_s=r["G"], drift_s=r["flow"]
    )
    model.train()
    tau = model(batch.q_s, batch.q_g)
    torch.testing.assert_close(tau.detach(), r["T"], rtol=1e-5, atol=2e-6)
    out = rantfields_loss(model, batch, r["flow_scale"], 5.0)
    assert out.total.item() == pytest.approx(r["loss"]["total"], rel=1e-5)
    assert out.terms["randers"] == pytest.approx(r["loss"]["randers_5"], rel=1e-5)
    assert out.terms["boundary"] == pytest.approx(r["loss"]["boundary_condition"], rel=1e-5)
    out.total.backward()
    ours = {"predict_time." + k: p.grad for k, p in model.named_parameters()}
    for k, g in r["grads"].items():
        if "flow_proj" not in k:
            assert ours[k] is not None
            torch.testing.assert_close(ours[k], g, rtol=1e-3, atol=1e-4 * float(g.abs().max()))


def test_arm_problem_fields(tmp_path: Path):
    """The arm metric (training-time G and drift) and the reference's online drift."""
    import_nap_import()
    from problems import ARM_VIA_SCALE, ReferenceOnlineDrift, arm_problem  # type: ignore[import-not-found]

    reference("arm_fields", tmp_path / "ref.npz")
    r = np.load(tmp_path / "ref.npz")
    q = torch.tensor(r["q"], dtype=torch.float64)
    metric = arm_problem("rantf").metric
    assert metric is not None
    np.testing.assert_allclose(metric.G(q).numpy(), r["G"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metric.drift(q).numpy(), r["f"], atol=2e-6)
    online = ReferenceOnlineDrift(
        lambda_reg=0.1, max_norm_g_sq=0.95, via_anisotropy=True, via_anisotropy_scale=ARM_VIA_SCALE
    )
    np.testing.assert_allclose(online.drift(q).numpy(), r["f_online"], atol=1e-6)


@pytest.mark.parametrize(("run", "atol"), [("flow_constant", 1e-4), ("terrain_ridge", 1e-3)])
def test_planner_paths(run: str, atol: float, tmp_path: Path):
    """`riemannian_gradient` against the reference's G-aware descent, on the same checkpoint and pairs. The
    reference looks the drift up at the nearest dataset point, ntfields evaluates it exactly, so paths agree less
    closely where the drift varies (the terrain)."""
    path, epoch = RUNS[run]
    field = import_nap_import().load_nap_run(NAP / path, epoch)
    g = torch.Generator().manual_seed(0)
    starts, goals = sample_valid(field.problem, 10, g), sample_valid(field.problem, 10, g)
    torch.save({"starts": starts, "goals": goals}, tmp_path / "pairs.pt")
    reference("planner", NAP / path, epoch, tmp_path / "pairs.pt", tmp_path / "ref.pt")
    r = torch.load(tmp_path / "ref.pt", weights_only=False)
    for s, e, ref_path, ref_ok in zip(starts, goals, r["paths"], r["converged"], strict=True):
        result = field.plan(s, e)
        assert result.success == ref_ok
        n = min(len(result.path), len(ref_path))
        torch.testing.assert_close(result.path[:n], ref_path[:n].float(), atol=atol, rtol=0)


@pytest.mark.parametrize("model", sorted(TABLE_RUNS))
def test_closed_loop_matches_the_published_runs(model: str, tmp_path: Path):
    """`GeodesicTauController` in MuJoCo, with the reference's online drift, against the reference's own per-pair
    results behind the paper's table (2 pairs per model)."""
    pytest.importorskip("mujoco")
    import_nap_import()
    from sim.arm_table import main  # type: ignore[import-not-found]

    path, epoch = TABLE_RUNS[model]
    out = tmp_path / "table.json"
    main(["--pairs", "2", "--reference-drift", "--models", f"{model}={NAP / path}:{epoch}", "--out", str(out)])
    ours = json.loads(out.read_text())[model]
    ref = json.loads((NAP / TABLE_RESULTS[model]).read_text())["pairs"]
    for i, m in enumerate(ours):
        r = ref[str(i)]["metrics"]["nap_geodesic"]
        assert m["converged"] == r["converged"] and m["collision_free"] == r["collision_free"]
        assert m["time_to_goal"] == pytest.approx(r["time_to_goal"], abs=1e-6)
        assert m["traced_length"] == pytest.approx(r["traced_path_length"], rel=1e-5)
        assert m["peak_torque"] == pytest.approx(r["max_torque"], rel=1e-4)
