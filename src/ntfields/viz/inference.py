import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ntfields.bench.protocol import Protocol, densify, evaluate_path, is_valid
from ntfields.bench.suite import Suite
from ntfields.field import TrainedField

# a compact viridis, for travel times; the viewer stays free of plotting libraries
VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32) / 255
BLOCKED = np.array([0.15, 0.15, 0.15], np.float32)


@dataclass
class PlanView:
    """One online plan, judged as the benchmark judges it."""

    path: Tensor  # (N, dim), on the cpu
    success: bool
    converged: bool
    travel_time: float
    cost: float | None
    length: float | None
    seconds: float
    min_clearance: float
    first_invalid: float | None  # where along the path, in [0, 1]


def plan_view(
    trained: TrainedField, protocol: Protocol, start: Tensor, goal: Tensor, planner: str, rng_seed: int, device: str
) -> PlanView:
    s, g = start.to(device), goal.to(device)
    torch.manual_seed(rng_seed)
    t0 = time.perf_counter()
    result = trained.plan(s, g, planner)
    seconds = time.perf_counter() - t0
    path = result.path.detach().cpu()
    outcome = evaluate_path(trained.problem, protocol, start.cpu(), goal.cpu(), path)
    dense = densify(path.double(), protocol.resolution)
    valid = is_valid(trained.problem, dense, protocol.margin, protocol.self_collision)
    bad = (~valid).nonzero().view(-1)
    return PlanView(
        path=path,
        success=outcome.success,
        converged=bool(result.success),
        travel_time=float(trained.travel_time(s[None], g[None])[0]),
        cost=outcome.cost,
        length=outcome.length,
        seconds=seconds,
        min_clearance=float(trained.problem.env.distance(dense.float()).min()),
        first_invalid=float(bad[0]) / max(len(dense) - 1, 1) if len(bad) else None,
    )


def colormap(values: np.ndarray) -> np.ndarray:
    """(N, 3) colors, low to high over the finite values."""
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    x = np.nan_to_num((values - lo) / max(hi - lo, 1e-9), nan=1.0, posinf=1.0, neginf=0.0).clip(0, 1) * (
        len(VIRIDIS) - 1
    )
    low = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    return VIRIDIS[low] + (VIRIDIS[low + 1] - VIRIDIS[low]) * (x - low)[:, None]


def field_slice(
    trained: TrainedField, source: Tensor, z: float, resolution: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """T(source, ·) over a point scene, at height `z` if it is 3D: grid points (N, 3) and colors (N, 3); blocked cells
    are dark."""
    bounds = trained.problem.env.bounds.cpu()
    axes = [torch.linspace(float(lo), float(hi), resolution) for lo, hi in bounds[:2]]
    x, y = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()], -1)
    if len(bounds) == 3:
        grid = torch.cat([grid, torch.full((len(grid), 1), z)], 1)
    q = grid.to(device)
    t = trained.travel_time(source.to(device).expand(len(q), -1), q).cpu().numpy().astype(np.float64)
    colors = colormap(t)
    colors[(trained.problem.env.distance(q) <= 0).cpu().numpy()] = BLOCKED
    planar = grid if len(bounds) == 3 else torch.cat([grid, torch.zeros(len(grid), 1)], 1)
    return planar.numpy().astype(np.float32), colors.astype(np.float32)


class Worker:
    """Runs jobs one at a time off the GUI's thread. A job waits for the one running; a newer job replaces one still
    waiting, since only the latest request matters."""

    def __init__(self) -> None:
        self._ready = threading.Condition()
        self._next: tuple[Callable[[], Any], Callable[[Any, Exception | None], None]] | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job: Callable[[], Any], done: Callable[[Any, Exception | None], None]) -> None:
        with self._ready:
            self._next = (job, done)
            self._ready.notify()

    def _loop(self) -> None:
        while True:
            with self._ready:
                while self._next is None:
                    self._ready.wait()
                job, done = self._next
                self._next = None
            try:
                result, error = job(), None
            except Exception as e:  # reported in the GUI: a bad checkpoint must not kill the viewer
                result, error = None, e
            done(result, error)


def judged_as(suite: Suite, self_collision: bool) -> Protocol:
    p = suite.protocol
    return Protocol(p.margin, p.resolution, p.endpoint_tol, p.reference_tol, self_collision)
