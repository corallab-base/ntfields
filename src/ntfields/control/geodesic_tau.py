import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ntfields.control.dynamics import Dynamics
from ntfields.field import TrainedField
from ntfields.plan.riemannian_gradient import riemannian_direction


@dataclass(frozen=True)
class GeodesicTauConfig:
    """Defaults are the paper's UR5e runs, except `lam`: the paper's table fixes λ = 0.5."""

    cruise_speed: float = 1.15  # speed of q̇_des, in physical units per second
    decel_radius: float = 0.2  # q̇_des ramps linearly to 0 within this distance of the goal
    beta: float = 0.1  # λ = βT/(1 + βT): weight of the field's direction against the straight line to the goal
    lam: float | None = None  # fixed λ instead
    replan_every: int = 10  # re-query the field every this many control steps, holding q̇_des in between
    tracking_time_constant: float | None = None  # q̈_des = (q̇_des - q̇)/τ; None: replan_every · dt
    qddot_max: float | None = None  # optional per-joint cap on q̈_des
    goal_tol: float = 0.05
    goal_tol_vel: float = 0.1


class GeodesicTauController:
    """Online kinodynamic controller on a trained field: every `replan_every` steps it queries the field's G-aware
    descent direction at the current state, blends it with the straight line to the goal, and sets q̇_des at
    `cruise_speed`; every step it realizes q̈_des = (q̇_des - q̇)/τ through inverse dynamics, τ = ID(q, q̇, q̈_des),
    clipped to `tau_max`.

    The robot runs in physical coordinates. When the field's problem uses coordinates normalized to [-1, 1] over
    `physical_bounds` (dim, 2), states are mapped into it and directions back out of it.
    """

    def __init__(
        self,
        field: TrainedField,
        dynamics: Dynamics,
        goal: np.ndarray,
        tau_max: np.ndarray,
        dt: float,
        cfg: GeodesicTauConfig | None = None,
        physical_bounds: Sequence[Sequence[float]] | None = None,
        device: str = "cpu",
    ) -> None:
        self.field, self.dynamics, self.goal = field, dynamics, np.asarray(goal, dtype=float)
        self.tau_max, self.dt, self.cfg = tau_max, dt, cfg or GeodesicTauConfig()
        self.device = device
        b = None if physical_bounds is None else np.asarray(physical_bounds, dtype=float)
        self._lo, self._half = (None, None) if b is None else (b[:, 0], (b[:, 1] - b[:, 0]) / 2)
        self._tau_track = self.cfg.tracking_time_constant or self.cfg.replan_every * dt
        self._goal_field = self._to_field(self.goal)
        self._step = 0
        self._qdot_des = np.zeros_like(self.goal)
        self.t_current = float("nan")
        self.query_times_s: list[float] = []
        # warm up (metric set-up, kernels), so query_times_s measures steady-state queries
        riemannian_direction(self.field, self._goal_field, self._goal_field, lam=1.0)

    def _to_field(self, q: np.ndarray) -> torch.Tensor:
        x = q if self._lo is None else (q - self._lo) / self._half - 1
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def desired_velocity(self, q: np.ndarray) -> np.ndarray:
        """q̇_des (physical units), re-queried every `replan_every` calls."""
        c = self.cfg
        if self._step % c.replan_every == 0:
            start = time.perf_counter()
            direction, self.t_current = riemannian_direction(self.field, self._to_field(q), self._goal_field, lam=1.0)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            self.query_times_s.append(time.perf_counter() - start)
            d = direction.detach().cpu().numpy().astype(float)
            if self._half is not None:
                d = d * self._half  # a displacement maps back with the inverse of the normalization
            norm = np.linalg.norm(d)
            d = d / norm if norm > 1e-8 else np.zeros_like(d)
            lam = c.beta * self.t_current / (1 + c.beta * self.t_current) if c.lam is None else c.lam
            to_goal = self.goal - q
            dist = np.linalg.norm(to_goal)
            blended = lam * d + (1 - lam) * (to_goal / dist if dist > 1e-8 else np.zeros_like(to_goal))
            bnorm = np.linalg.norm(blended)
            speed = c.cruise_speed * min(1.0, dist / c.decel_radius)
            self._qdot_des = speed * blended / bnorm if bnorm > 1e-8 else np.zeros_like(q)
        self._step += 1
        return self._qdot_des

    def compute_torque(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        qddot = (self.desired_velocity(q) - qdot) / self._tau_track
        if self.cfg.qddot_max is not None:
            qddot = np.clip(qddot, -self.cfg.qddot_max, self.cfg.qddot_max)
        return np.clip(self.dynamics.inverse_dynamics(q, qdot, qddot), -self.tau_max, self.tau_max)

    def is_converged(self, q: np.ndarray, qdot: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(q - self.goal) <= self.cfg.goal_tol and np.linalg.norm(qdot) <= self.cfg.goal_tol_vel
        )
