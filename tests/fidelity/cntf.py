"""Access to CNTFields (ntfields-old, a corallab-base repo), imported unmodified from a local checkout."""

import importlib
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
from torch import Tensor

CNTFIELDS = Path(os.environ.get("NTFIELDS_CNTFIELDS", Path(__file__).resolve().parents[3] / "ntfields-old"))
DIM = 6

requires_cntfields = pytest.mark.skipif(
    not (CNTFIELDS / "ntfields" / "model" / "cntfield_model.py").is_file(),
    reason="set NTFIELDS_CNTFIELDS to a checkout",
)
_loaded: Any = None


def import_cntfields() -> Any:
    """ntfields-old's `ntfields.model.cntfield_model`.

    That repo's package is also called `ntfields`, so it is mounted under the name just for this import, and robofin
    and geometrout (only used by its visualization and joint normalization) are stubbed.
    """
    global _loaded
    if _loaded is not None:
        return _loaded
    ours = {k: v for k, v in sys.modules.items() if k == "ntfields" or k.startswith("ntfields.")}
    for k in ours:
        del sys.modules[k]
    root = types.ModuleType("ntfields")
    root.__path__ = [str(CNTFIELDS / "ntfields")]  # type: ignore[attr-defined]
    sys.modules["ntfields"] = root
    stubs = {k: mock.MagicMock() for k in ("robofin", "robofin.bullet", "robofin.robots", "geometrout",
                                           "geometrout.primitive")}  # fmt: skip
    try:
        with mock.patch.dict(sys.modules, stubs):
            _loaded = importlib.import_module("ntfields.model.cntfield_model")
    finally:
        for k in [k for k in sys.modules if k == "ntfields" or k.startswith("ntfields.")]:
            del sys.modules[k]
        sys.modules.update(ours)
    return _loaded


def ref_config(dim: int = DIM) -> dict[str, Any]:
    """The fields of `cntfields_hold_cup.yaml` the model reads."""
    return {
        "name": "cntfields",
        "model": {"dim": dim, "save_path": "."},
        "data": {"train_data_path": "."},
        "loss": {"gamma": 0.002},
    }


def ref_model(dtype: torch.dtype = torch.float64, dim: int = DIM) -> tuple[Any, Tensor]:
    """A freshly initialized reference `CNTFields` and its Fourier matrix B (n, dim)."""
    model = import_cntfields().CNTFields(ref_config(dim), device="cpu")
    model.network.to(dtype)
    model.network.B = model.network.B.to(dtype)  # a plain attribute, not a buffer
    return model, model.network.B.T.clone()


def ref_train(
    model: Any,
    data: Tensor,
    epochs: int,
    seed: int,
    batch: int,
    threshold: float = 1.2,
    stats: dict[str, int] | None = None,
) -> list[float]:
    """base_model.py::NTFields.train minus prints/plots/saves; data rows are [points (2·dim), raw speed (2)].

    Shuffles with a fresh randperm of the original order per attempt (the reference's loader re-permutes its already
    permuted tensors: statistically the same). `threshold` stands in for the hard-coded 1.2; rollbacks are counted in
    `stats`. Returns the per-epoch mean loss.
    """
    net = model.network
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    torch.manual_seed(seed)
    random.seed(seed)
    dim = (data.shape[1] - 2) // 2
    normals = torch.zeros(batch, 2 * dim, dtype=data.dtype)

    curr_loss, losses = 9999.0, []
    curr_model = pickle.loads(pickle.dumps(net.state_dict()))
    curr_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
    prev_models: list[Any] = []
    prev_optimizers: list[Any] = []
    for epoch in range(epochs):
        prev_models.append(curr_model)
        prev_optimizers.append(curr_optimizer)
        if len(prev_models) > 5:
            prev_models.pop(0)
            prev_optimizers.pop(0)
        curr_model = pickle.loads(pickle.dumps(net.state_dict()))
        curr_optimizer = pickle.loads(pickle.dumps(optimizer.state_dict()))
        prev_loss = curr_loss
        alpha = min(0.5 + epoch * 1e-4, 0.95)
        while True:
            avg_loss = 0.0
            shuffled = data[torch.randperm(len(data))]
            for i in range(5):
                rows = shuffled[i * batch : (i + 1) * batch]
                speeds = rows[:, 2 * dim :]
                speeds = speeds * speeds * (2 - speeds) * (2 - speeds)
                speeds = alpha * speeds + 1 - alpha
                loss_value = model.loss(rows[:, : 2 * dim], speeds, normals)
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss_value.item()
            curr_loss = avg_loss / 5
            if curr_loss / prev_loss < threshold:
                losses.append(curr_loss)
                break
            if stats is not None:
                stats["rollbacks"] = stats.get("rollbacks", 0) + 1
            with torch.no_grad():
                k = random.randint(0, len(prev_models) - 1)
                net.load_state_dict(prev_models[k], strict=True)
                optimizer.load_state_dict(prev_optimizers[k])
    return losses
