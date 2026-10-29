"""Shared pieces of the P-NTFields UR5 benchmark: problem, pair data, and the two models to compare."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ntfields import Problem
from ntfields.data import PairDataset, Sampler
from ntfields.methods.pntfields import PNTFieldsConfig
from ntfields.methods.pntfields.model import PNTFieldsModel
from ntfields.speed import Clipped

sys.path.insert(0, str(Path(__file__).parent))
from scene import CHECKPOINT, MARGIN, OFFSET, ur5_cabinet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.fidelity.pntf import to_ntfields_state  # noqa: E402


def problem(work: Path, device: str = "cuda") -> Problem:
    return Problem(ur5_cabinet(work / "ur5_cabinet_grid.pt", device), Clipped(OFFSET, MARGIN))


def pairs(problem: Problem, n: int, seed: int, sampler: Sampler, device: str = "cuda") -> PairDataset:
    """Pairs with S* at both ends; no normals (P-NTFields does not use them)."""
    q_s, q_g = sampler.sample(problem, n, torch.Generator().manual_seed(seed), device)
    with torch.no_grad():
        s_s, s_g = (torch.cat([problem.speed_at(c.to(device)).cpu() for c in q.split(65536)]) for q in (q_s, q_g))
    zeros = torch.zeros_like(q_s)
    return PairDataset(q_s, q_g, s_s, s_g, zeros, zeros)


def cached_pairs(problem: Problem, n: int, seed: int, sampler: Sampler, path: Path) -> PairDataset:
    if not path.is_file():
        pairs(problem, n, seed, sampler).save(path)
    return PairDataset.load(path)


def reference_model() -> PNTFieldsModel:
    """The published UR5 model (Google Drive) in ntfields."""
    safe = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = PNTFieldsModel(6)
    model.load_state_dict(to_ntfields_state(ckpt["model_state_dict"], ckpt["B_state_dict"]))
    return model


def ntfields_model(path: Path) -> PNTFieldsModel:
    c = PNTFieldsConfig.arm()
    model = PNTFieldsModel(6, c.hidden, c.encoder_blocks, c.generator_blocks, c.fourier_features, c.fourier_scale)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def straight(a: Tensor, b: Tensor, step: float) -> Tensor:
    n = max(1, int((b - a).norm() / step) + 1)
    return a + torch.linspace(0, 1, n + 1, device=a.device, dtype=a.dtype)[:, None] * (b - a)
