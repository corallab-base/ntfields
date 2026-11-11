# pinocchio ships no type stubs
# pyright: reportAttributeAccessIssue=false
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class ArmDynamics:
    """Gravity drift and inertia metric of the first `dof` joints of a robot arm (the rest held at 0), in
    coordinates q = θ / θ_max (needs the `[dynamics]` extra):

    - G(q) = M(θ)/m_ref + `lambda_reg`·I, with m_ref the mean trace(M)/2 over a few reference poses;
    - f(q) = c·b(θ)/θ_max, b = -M⁻¹τ_g, with one global c so that max bᵀGb = `max_norm_g_sq` (times a 0.9 margin
      when `lambda_reg` > 0), found by local maximization from reference and random poses;
    - with `via_anisotropy`, f is replaced by `via_anisotropy_scale`·G·f (a raw gravity-torque direction), then
      scaled down where needed so that fᵀGf ≤ `clip_norm_g_sq`;
    - without `with_drift`, f = 0.

    G is applied to gradients in q unchanged, as in the reference (it is not transformed to the scaled
    coordinates).
    """

    robot: str = "ur5e"
    dof: int = 5
    lambda_reg: float = 0.0
    max_norm_g_sq: float = 0.5
    via_anisotropy: bool = False
    via_anisotropy_scale: float = 1.0
    clip_norm_g_sq: float = 0.95
    with_drift: bool = True  # False: f = 0 (a Riemannian problem)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def G(self, q: Tensor) -> Tensor:
        g, _ = self._evaluate(q)
        return g

    def drift(self, q: Tensor) -> Tensor:
        _, f = self._evaluate(q)
        return f

    def physical(self, q: Tensor) -> np.ndarray:
        """Joint angles θ (B, dof) of q."""
        return q.detach().cpu().double().numpy() * self._model()["upper"]

    def _evaluate(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """(G, f) at q; the last result is reused, since callers usually ask for G and f at the same states."""
        last = self._cache.get("last")
        if last is not None and last[0].shape == q.shape and torch.equal(last[0], q):
            return last[1], last[2]
        g, f = self._compute(q)
        self._cache["last"] = (q.detach().clone(), g, f)
        return g, f

    def _compute(self, q: Tensor) -> tuple[Tensor, Tensor]:
        m = self._model()
        theta = self.physical(q)
        gs, fs = [], []
        for t in theta:
            mass, tau = self._mass_and_gravity(t)
            g = mass / m["m_ref"] + self.lambda_reg * np.eye(self.dof)
            f = -np.linalg.solve(mass, tau) * self._drift_scale() / m["upper"]
            if self.via_anisotropy:
                f = self.via_anisotropy_scale * (g @ f)
                f = f * min(1.0, math.sqrt(self.clip_norm_g_sq / max(float(f @ g @ f), 1e-12)))
            gs.append(g)
            fs.append(f if self.with_drift else np.zeros_like(f))
        to = {"dtype": q.dtype, "device": q.device}
        return torch.tensor(np.stack(gs), **to), torch.tensor(np.stack(fs), **to)

    def _model(self) -> dict[str, Any]:
        if "model" not in self._cache:
            from ntfields.control.dynamics import load_pinocchio_model

            model = load_pinocchio_model(self.robot)
            upper = np.asarray(model.upperPositionLimit[: self.dof])
            self._cache.update(model=model, data=model.createData(), upper=upper)
            traces = [np.trace(self._mass_and_gravity(p)[0]) / 2 for p in self._reference_poses()]
            self._cache["m_ref"] = max(float(np.mean(traces)), 1e-8)
        return self._cache

    def _mass_and_gravity(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import pinocchio as pin

        model, data = self._cache["model"], self._cache["data"]
        full = np.zeros(model.nq)
        full[: self.dof] = theta
        upper = np.triu(pin.crba(model, data, full))  # crba fills only the upper triangle
        mass = (upper + upper.T - np.diag(np.diag(upper)))[: self.dof, : self.dof]
        tau = pin.computeGeneralizedGravity(model, data, full)[: self.dof]
        return mass, tau

    def _reference_poses(self) -> list[np.ndarray]:
        """The zero pose, all joints at ±π/2, and each joint alone at ±π/2 (clipped to the limits)."""
        model = self._cache["model"]
        lo, hi = model.lowerPositionLimit[: self.dof], model.upperPositionLimit[: self.dof]
        poses = [np.zeros(self.dof)] + [np.clip(np.full(self.dof, s * math.pi / 2), lo, hi) for s in (1.0, -1.0)]
        for k in range(self.dof):
            for s in (1.0, -1.0):
                p = np.zeros(self.dof)
                p[k] = np.clip(s * math.pi / 2, lo[k], hi[k])
                poses.append(p)
        return poses

    def _b_g_b(self, theta: np.ndarray) -> float:
        mass, tau = self._mass_and_gravity(theta)
        b = -np.linalg.solve(mass, tau)
        return float(b @ (mass / self._cache["m_ref"] + self.lambda_reg * np.eye(self.dof)) @ b)

    def _drift_scale(self) -> float:
        if "drift_scale" not in self._cache:
            m = self._model()
            model = m["model"]
            lo, hi = model.lowerPositionLimit[: self.dof], model.upperPositionLimit[: self.dof]
            rng = np.random.default_rng(0)
            starts = self._reference_poses()
            starts += [np.array([rng.uniform(lo[i], hi[i]) for i in range(self.dof)]) for _ in range(500)]
            peak = 1e-8
            for p in starts:
                peak = max(peak, self._b_g_b(p))
                if self.lambda_reg > 0:
                    from scipy.optimize import minimize

                    box = list(zip(lo, hi, strict=True))
                    res = minimize(lambda x: -self._b_g_b(x), p, method="L-BFGS-B", bounds=box, options={"maxiter": 50})
                    peak = max(peak, float(-res.fun))
            target = self.max_norm_g_sq * (0.9 if self.lambda_reg > 0 else 1.0)
            m["drift_scale"] = math.sqrt(target / peak)
        return self._cache["drift_scale"]
