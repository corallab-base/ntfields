from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from ntfields.base import SceneCondition
from ntfields.grid import Interpolation, interpolate


@dataclass
class GridEnv:
    """Point mass in an occupancy grid (2D or 3D), e.g. a map from pyrobosim or ROS.

    Array axis i is coordinate i, and cell `idx` covers `origin + idx * resolution` to `origin + (idx + 1) *
    resolution`, so `bounds` is the whole map. (A ROS `OccupancyGrid`'s data is row-major (y, x): transpose it.)

    `distance` interpolates a signed distance stored at the cell centres (`sdf`): the Euclidean distance transform of
    the occupancy, half a cell short, so it is 0 on the boundary between a free and an occupied cell. It is accurate
    to about half a cell. Linear interpolation, the default, never overshoots, so a wall one cell thick stays a wall;
    its gradient (the clearance normal training uses) jumps at cell edges, which "cubic" smooths.
    """

    bounds: Tensor  # (dim, 2)
    occupied: Tensor  # (R_0, ..., R_{dim-1}) bool
    sdf: Tensor  # same shape, float64, at the cell centres
    interp: Interpolation = "linear"

    @classmethod
    def from_occupancy(
        cls,
        occupancy: ArrayLike,
        resolution: float | Sequence[float],
        origin: Sequence[float] | None = None,
        interp: Interpolation = "linear",
    ) -> Self:
        """`occupancy`: nonzero = occupied (so ROS's -1, unknown, counts as occupied). `origin`: the map's lowest
        corner (default 0). Needs scipy (the `gt` extra)."""
        from scipy.ndimage import distance_transform_edt

        occupied = np.asarray(occupancy) != 0
        dim = occupied.ndim
        if dim not in (2, 3):
            raise ValueError(f"expected a 2D or 3D occupancy grid, got shape {occupied.shape}")
        cell = np.broadcast_to(np.asarray(resolution, dtype=np.float64), (dim,))
        lo = np.zeros(dim) if origin is None else np.asarray(origin, dtype=np.float64)
        bounds = np.stack([lo, lo + cell * occupied.shape], axis=-1)

        if not occupied.any():
            sdf = np.full(occupied.shape, np.inf)
        elif occupied.all():
            sdf = np.full(occupied.shape, -np.inf)
        else:
            # distance_transform_edt(a) is, at each nonzero cell of a, the distance to the nearest zero cell
            outside = distance_transform_edt(~occupied, sampling=cell)  # free cell -> nearest occupied centre
            inside = distance_transform_edt(occupied, sampling=cell)  # occupied cell -> nearest free centre
            half = 0.5 * cell.min()
            sdf = np.where(occupied, half - inside, outside - half)  # pyright: ignore[reportOperatorIssue]
        return cls(
            torch.tensor(bounds, dtype=torch.float32),
            torch.from_numpy(occupied.copy()),
            torch.from_numpy(np.asarray(sdf, dtype=np.float64)),
            interp,
        )

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    @property
    def cell(self) -> Tensor:
        """(dim,) cell side lengths."""
        lo, hi = self.bounds.double().unbind(-1)
        return (hi - lo) / torch.tensor(self.occupied.shape, dtype=torch.float64)

    def distance(self, q: Tensor) -> Tensor:
        """Signed distance to the nearest occupied cell (inf if there are none), clamped to the border outside."""
        if torch.isinf(self.sdf).all():
            return self.sdf.view(-1)[0].to(q).expand(q.shape[0]).clone()
        half = self.cell / 2
        centres = torch.stack([self.bounds[:, 0].double() + half, self.bounds[:, 1].double() - half], dim=-1)
        return interpolate(self.sdf, centres, q, self.interp)

    def in_bounds(self, q: Tensor) -> Tensor:
        b = self.bounds.to(q)
        return ((q >= b[:, 0]) & (q <= b[:, 1])).all(dim=-1)

    def is_valid(self, q: Tensor) -> Tensor:
        """In bounds and in a free cell."""
        cell = self.cell.to(q.device)
        size = torch.tensor(self.occupied.shape, device=q.device)
        idx = ((q.double() - self.bounds[:, 0].to(q.device, torch.float64)) / cell).floor().long()
        idx = idx.clamp(min=0).minimum(size - 1)  # the upper bound belongs to the last cell
        free = ~self.occupied.to(q.device)[tuple(idx.unbind(-1))]
        return self.in_bounds(q) & free

    def condition(self) -> SceneCondition:
        return SceneCondition()
