from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import h5py
import torch
from torch import Tensor

from ntfields.base import Batch, SceneCondition


@dataclass
class PairDataset:
    q_s: Tensor  # (N, dim)
    q_g: Tensor
    speed_s: Tensor  # (N,)
    speed_g: Tensor
    normal_s: Tensor  # (N, dim)
    normal_g: Tensor

    def __len__(self) -> int:
        return self.q_s.shape[0]

    def to(self, device: torch.device | str) -> Self:
        return type(self)(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def __getitem__(self, idx: Tensor) -> Self:
        return type(self)(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def batches(
        self, batch_size: int, scene: SceneCondition, generator: torch.Generator | None = None
    ) -> Iterator[Batch]:
        """Shuffled minibatches; the last partial batch is kept."""
        perm = torch.randperm(len(self), generator=generator).to(self.q_s.device)
        for idx in perm.split(batch_size):
            yield Batch(**{f.name: getattr(self, f.name)[idx] for f in fields(self)}, scene=scene)

    def save(self, path: str | Path) -> None:
        with h5py.File(path, "w") as f:
            for field in fields(self):
                f.create_dataset(field.name, data=getattr(self, field.name).cpu().numpy())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        tensors: dict[str, Tensor] = {}
        with h5py.File(path, "r") as f:
            for field in fields(cls):
                ds = f[field.name]
                assert isinstance(ds, h5py.Dataset)
                tensors[field.name] = torch.from_numpy(ds[()])
        return cls(**tensors)
