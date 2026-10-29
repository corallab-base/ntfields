"""Access to P-NTFields (github.com/ruiqini/P-NTFields), imported unmodified from a local checkout."""

import importlib.util
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from ntfields.methods.pntfields.model import PNTFieldsModel

PNTFIELDS = Path(os.environ.get("NTFIELDS_PNTFIELDS", Path(__file__).resolve().parents[3] / "P-NTFields"))
UR5_CKPT = PNTFIELDS / "Experiments" / "UR5" / "Model_Epoch_10000_ValLoss_3.526511e-03.pt"
DIM = 6

requires_pntfields = pytest.mark.skipif(
    not (PNTFIELDS / "models" / "model_res_sigmoid.py").is_file(), reason="set NTFIELDS_PNTFIELDS to a checkout"
)


def import_pntfields() -> Any:
    """models/model_res_sigmoid.py, loaded by path: ntrl-demo also has a top-level `models` package."""
    name = "pntfields_model_res_sigmoid"
    if name in sys.modules:
        return sys.modules[name]
    sys.modules.setdefault("pickle5", pickle)
    spec = importlib.util.spec_from_file_location(name, PNTFIELDS / "models" / "model_res_sigmoid.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `from models import data_mlp` only feeds the reference's own dataset loading, which we never call
    saved = sys.modules.get("models")
    sys.modules["models"] = types.SimpleNamespace(data_mlp=types.ModuleType("data_mlp"))  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["models"]
        else:
            sys.modules["models"] = saved
    sys.modules[name] = module
    return module


def ref_network(b: Tensor, dim: int = DIM) -> Any:
    """A freshly initialized reference NN (float64) with Fourier matrix B (n, dim)."""
    net = import_pntfields().NN("cpu", dim, b)
    net.apply(net.init_weights)
    net.double()
    net.B = net.B.double()  # a plain attribute, not a buffer
    return net


def ref_model(net: Any, dim: int = DIM) -> Any:
    """A reference `Model` wrapping `net`, for `Loss`, `TravelTimes` and `Gradient`."""
    model = import_pntfields().Model(".", ".", dim, device="cpu")
    model.network = net
    return model


def to_ntfields_state(ref_state: dict[str, Tensor], b: Tensor, encoder_blocks: int = 2) -> dict[str, Tensor]:
    """Reference `NN` state dict (+ B) -> PNTFieldsModel state dict; drops the never-called `encoder1.0`."""
    pairs = {"encoder_in": "encoder.0", "encoder_out": f"encoder.{encoder_blocks + 1}"}
    pairs |= {f"encoder_blocks.{i}.lin1": f"encoder.{i + 1}" for i in range(encoder_blocks)}
    pairs |= {f"encoder_blocks.{i}.lin2": f"encoder1.{i + 1}" for i in range(encoder_blocks)}
    n_gen = sum(1 for k in ref_state if k.startswith("generator1.") and k.endswith(".weight"))
    pairs |= {f"generator_blocks.{i}.lin1": f"generator.{i}" for i in range(n_gen)}
    pairs |= {f"generator_blocks.{i}.lin2": f"generator1.{i}" for i in range(n_gen)}
    pairs |= {"generator_hidden": f"generator.{n_gen}", "generator_out": f"generator.{n_gen + 1}"}
    out = {"fourier.b": b}
    for ours, theirs in pairs.items():
        for p in ("weight", "bias"):
            out[f"{ours}.{p}"] = ref_state[f"{theirs}.{p}"]
    return out


def ntfields_model_from_ref(net: Any, b: Tensor, dim: int = DIM) -> PNTFieldsModel:
    model = PNTFieldsModel(dim, hidden=net.encoder[0].weight.shape[0], fourier_features=b.shape[0]).to(b.dtype)
    model.load_state_dict(to_ntfields_state(net.state_dict(), b))
    return model


def ref_train(net: Any, data: Tensor, epochs: int, seed: int, batch: int) -> list[float]:
    """model_res_sigmoid.py::Model.train minus prints/plots/saves; data rows are [points (2·dim), raw speed (2)].

    Shuffles with a fresh randperm of the original order per attempt (the reference's loader re-permutes its already
    permuted tensors: statistically the same) and draws a rollback state from the whole queue (the reference indexes
    randint(0, 4), which fails while the queue is shorter). Returns the per-epoch `total_diff`.
    """
    model = ref_model(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    n_loader = -(-len(data) // batch)
    torch.manual_seed(seed)
    random.seed(seed)

    beta, current_diff, step, losses = 1.0, 1.0, -2000.0 / 4000.0, []
    current_state = pickle.loads(pickle.dumps(net.state_dict()))
    current_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
    state_queue: list[Any] = []
    optimizer_queue: list[Any] = []
    for epoch in range(1, epochs + 1):
        alpha = min(max(0.5, 0.5 + 0.5 * step), 1.05)
        step += 1.0 / 4000 / (int(epoch / 4000) + 1.0)
        state_queue.append(current_state)
        optimizer_queue.append(current_optimizer)
        if len(state_queue) > 5:
            state_queue.pop(0)
            optimizer_queue.pop(0)
        current_state = pickle.loads(pickle.dumps(net.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
        optimizer.param_groups[0]["lr"] = min(max(1e-3 * (1 - (epoch - 8000) / 1000.0), 5e-4), 1e-3)
        prev_diff = current_diff
        while True:
            total_diff = 0.0
            shuffled = data[torch.randperm(len(data))]
            for i in range(6):
                rows = shuffled[i * batch : (i + 1) * batch]
                speed = alpha * rows[:, 2 * DIM :] + 1 - alpha
                loss, loss_n, _ = model.Loss(rows[:, : 2 * DIM], speed, beta, 0.001)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_diff += loss_n.item()
            current_diff = total_diff / n_loader
            if 0 < current_diff / prev_diff < 1.2:
                break
            with torch.no_grad():
                k = random.randint(0, len(state_queue) - 1)
                net.load_state_dict(state_queue[k], strict=True)
                optimizer.load_state_dict(optimizer_queue[k])
        beta = 1.0 / current_diff
        losses.append(current_diff)
    return losses


def random_pairs(n: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Reference-style rows: points (n, 2·dim) in [-0.5, 0.5] and raw speeds (n, 2) in [0.1, 1], float64."""
    x = torch.rand(n, 2 * DIM, generator=generator, dtype=torch.float64) - 0.5
    speed = 0.1 + 0.9 * torch.rand(n, 2, generator=generator, dtype=torch.float64)
    return x, speed
