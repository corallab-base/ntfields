from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
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
    # problem data, set when the problem has a `Metric` (see `sample_pairs`)
    metric_s: Tensor | None = field(default=None, kw_only=True)  # (N, dim, dim) G(q)
    metric_g: Tensor | None = field(default=None, kw_only=True)
    drift_s: Tensor | None = field(default=None, kw_only=True)  # (N, dim) f(q)
    drift_g: Tensor | None = field(default=None, kw_only=True)
    # per-pair tensors (N, ...) a method needs beyond these (e.g. a flow or a metric at both ends), carried through
    # shuffling, batching and saving untouched; ntfields itself never reads them
    extras: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k, v in self.extras.items():
            if v.shape[0] != len(self):
                raise ValueError(f"extra {k!r} has {v.shape[0]} rows, the dataset {len(self)}")

    def __len__(self) -> int:
        return self.q_s.shape[0]

    def _tensors(self) -> dict[str, Tensor]:
        """The standard fields that are set."""
        return {f.name: v for f in fields(self) if f.name != "extras" and (v := getattr(self, f.name)) is not None}

    def to(self, device: torch.device | str) -> Self:
        return self._map(lambda t: t.to(device))

    def __getitem__(self, idx: Tensor) -> Self:
        return self._map(lambda t: t[idx])

    def _map(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return type(self)(
            **{k: fn(v) for k, v in self._tensors().items()}, extras={k: fn(v) for k, v in self.extras.items()}
        )

    def batches(
        self, batch_size: int, scene: SceneCondition, generator: torch.Generator | None = None
    ) -> Iterator[Batch]:
        """Shuffled minibatches; the last partial batch is kept."""
        perm = torch.randperm(len(self), generator=generator).to(self.q_s.device)
        for idx in perm.split(batch_size):
            part = self[idx]
            yield Batch(**part._tensors(), scene=scene, extras=part.extras)

    def save(self, path: str | Path) -> None:
        with h5py.File(path, "w") as f:
            for name, t in self._tensors().items():
                f.create_dataset(name, data=t.cpu().numpy())
            for name, t in self.extras.items():
                f.create_dataset(f"extras/{name}", data=t.cpu().numpy())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        tensors: dict[str, Tensor] = {}
        extras: dict[str, Tensor] = {}
        with h5py.File(path, "r") as f:
            for fld in fields(cls):
                if fld.name == "extras" or fld.name not in f:  # optional fields may be absent
                    continue
                ds = f[fld.name]
                assert isinstance(ds, h5py.Dataset)
                tensors[fld.name] = torch.from_numpy(ds[()])
            group = f.get("extras")
            if isinstance(group, h5py.Group):
                for name, ds in group.items():
                    assert isinstance(ds, h5py.Dataset)
                    extras[name] = torch.from_numpy(ds[()])
        return cls(**tensors, extras=extras)
