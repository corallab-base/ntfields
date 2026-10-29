"""Access to ntrl-demo (github.com/ruiqini/ntrl-demo), imported unmodified from a local checkout."""

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from ntfields import Batch, SceneCondition
from ntfields.methods.td_ntfields.model import TDMetricModel

NTRL_DEMO = Path(os.environ.get("NTFIELDS_NTRL_DEMO", Path(__file__).resolve().parents[3] / "ntrl-demo"))
UR5_DATA = NTRL_DEMO / "datasets" / "arm" / "UR5"
DIM = 6

requires_ntrl = pytest.mark.skipif(
    not (NTRL_DEMO / "models").is_dir(), reason="set NTFIELDS_NTRL_DEMO to an ntrl-demo checkout"
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def import_ntrl() -> tuple[Any, Any]:
    """(model_network_metric, model_function_metric) from models/metric_arm."""
    sys.modules.setdefault("igl", types.ModuleType("igl"))  # imported but unused by the parts we call
    if str(NTRL_DEMO) not in sys.path:
        sys.path.insert(0, str(NTRL_DEMO))
    from models.metric_arm import model_function_metric, model_network_metric  # type: ignore[import-not-found]

    return model_network_metric, model_function_metric


def ref_network(run_dir: Path, b: Tensor, dim: int = DIM) -> Any:
    """A freshly initialized reference NN, built with the network code `run_dir` saved (its width)."""
    spec = importlib.util.spec_from_file_location(
        f"ntrl_net_{run_dir.name}", run_dir / "models" / "model_network_metric.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.NN("cpu", dim, b)
    net.apply(net.init_weights)
    return net


def load_ref_checkpoint(path: Path, dim: int = DIM) -> tuple[Any, Tensor]:
    """(reference NN, B) from a run checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ref_network(path.parent, ckpt["B_state_dict"], dim)
    net.load_state_dict(ckpt["model_state_dict"])
    return net, ckpt["B_state_dict"]


def ref_train(net: Any, data: Tensor, epochs: int, seed: int, batch: int = 2000) -> list[float]:
    """model_train_metric.py::train (arm), minus prints/plots/saves; data rows are [points, raw speed, normal].

    Shuffles with a fresh randperm of the original order per attempt (the reference's loader re-permutes its already
    permuted tensors: statistically the same). Returns the per-epoch `total_diff`.
    """
    import pickle
    import random

    _, function_module = import_ntrl()
    func = function_module.Function(".", "cpu", net, DIM)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.5)
    torch.manual_seed(seed)
    random.seed(seed)

    beta, current_diff, losses = 1.0, 1000.0, []
    current_state = pickle.loads(pickle.dumps(net.state_dict()))
    current_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
    state_queue: list[Any] = []
    optimizer_queue: list[Any] = []
    for epoch in range(1, epochs + 1):
        state_queue.append(current_state)
        optimizer_queue.append(current_optimizer)
        if len(state_queue) > 5:
            state_queue.pop(0)
            optimizer_queue.pop(0)
        current_state = pickle.loads(pickle.dumps(net.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
        optimizer.param_groups[0]["lr"] = 5e-4
        prev_diff = current_diff
        while True:
            total_diff = 0.0
            shuffled = data[torch.randperm(len(data))]
            for i in range(5):
                rows = shuffled[i * batch : (i + 1) * batch]
                s = rows[:, 2 * DIM : 2 * DIM + 2]
                speed = s * s * (2 - s) * (2 - s)
                loss, loss_n, _ = func.Loss(rows[:, : 2 * DIM], speed, rows[:, 2 * DIM + 2 :], beta, 0.001, epoch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_diff += loss_n.item()
            current_diff = total_diff / 4
            if 0 < current_diff / prev_diff < 1.2:
                break
            with torch.no_grad():
                k = random.randint(0, len(state_queue) - 1)
                net.load_state_dict(state_queue[k], strict=True)
                optimizer.load_state_dict(optimizer_queue[k])
        beta = 1.0 / current_diff
        losses.append(current_diff)
    return losses


def ur5_data() -> Tensor:
    """ntrl's whole UR5 training set as rows [points (12), raw speed (2), normal (12)]."""
    parts = [torch.from_numpy(np.load(UR5_DATA / f"{k}.npy")).float() for k in ("sampled_points", "speed", "normal")]
    return torch.cat(parts, dim=1)


def to_double(ref_net: Any, b: Tensor) -> tuple[Any, Tensor]:
    """The reference NN and B in float64, so comparisons test the math rather than float32 summation order (CPU
    BLAS reductions are not bit-reproducible across runs)."""
    ref_net.double()
    ref_net.B = ref_net.B.double()  # a plain attribute, not a buffer
    return ref_net, b.double()


def import_ntrl_planner() -> Any:
    """ntrl-demo's tests/arm_plan_stat.py (for `MPPI`); its FK/visualization imports are stubbed."""
    import_ntrl()
    for name in ("pytorch_kinematics", "igl"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location("ntrl_arm_plan_stat", NTRL_DEMO / "tests" / "arm_plan_stat.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_ntfields_state(ref_state: dict[str, Tensor], b: Tensor, n_blocks: int = 2) -> dict[str, Tensor]:
    """Reference `NN` state dict (+ its Fourier matrix B) -> TDMetricModel state dict.

    Drops the reference's unused `encoder.0` and the unused gate biases.
    """
    out = {"fourier.b": b}
    for ours, theirs in (("gate_u", "pe_gate.0"), ("gate_v", "pe_gate.1"), ("final", f"encoder.{3 * n_blocks + 1}")):
        out[f"{ours}.weight"] = ref_state[f"{theirs}.weight"]
        out[f"{ours}.bias"] = ref_state[f"{theirs}.bias"]
    for i in range(n_blocks):
        for j in range(3):
            for p in ("weight", "bias"):
                out[f"blocks.{i}.lin{j + 1}.{p}"] = ref_state[f"encoder.{3 * i + j + 1}.{p}"]
        out[f"block_gates.{i}"] = ref_state[f"gate.{i}.weight"].reshape(())
    return out


def ntfields_model_from_ref(ref_net: Any, b: Tensor, dim: int) -> TDMetricModel:
    model = TDMetricModel(dim, hidden=ref_net.encoder[1].weight.shape[0]).to(b.dtype)
    model.load_state_dict(to_ntfields_state(ref_net.state_dict(), b))
    return model


def ur5_batch(n: int) -> tuple[Tensor, Tensor, Tensor]:
    """The first n rows of ntrl's UR5 training data: points (n, 12), speeds (n, 2) after the reference's
    s²(2 - s)² transform, unit normals (n, 12)."""
    x, s, normal = (
        torch.from_numpy(np.load(UR5_DATA / f"{k}.npy")[:n]).float() for k in ("sampled_points", "speed", "normal")
    )
    return x, s * s * (2 - s) * (2 - s), normal


def to_batch(x: Tensor, speed: Tensor, normal: Tensor) -> Batch:
    return Batch(x[:, :DIM], x[:, DIM:], speed[:, 0], speed[:, 1], normal[:, :DIM], normal[:, DIM:], SceneCondition())
