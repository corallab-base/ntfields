"""The paper's problems (RaNTFields), as ntfields `Problem`s.

Constants that the reference computed from its own datasets are fixed here:
- `ARM_D_MAX`: the largest clearance among the arm dataset's samples;
- `ARM_VIA_SCALE`: its global rescale of the gravity-torque drift (mean fᵀGf = 0.5).
"""

import numpy as np
import torch
from torch import Tensor

from ntfields.envs import load_env
from ntfields.metric import ArmDynamics, FlowField, Terrain
from ntfields.problem import Problem
from ntfields.speed import ExpRamp

ARM_D_MAX = 0.2803711  # m
ARM_SPEED_SCALE = 5.0  # the reference's max_obstacle_distance_scale
ARM_VIA_SCALE = 14.565833569472103  # with max_norm_g_sq = 0.95 (the run's override)


def _point_mass(env: str, metric: FlowField | Terrain | None) -> Problem:
    e = load_env(env)
    return Problem(e, ExpRamp.for_env(e), metric)


PROBLEMS_2D = {
    "flow_constant": lambda: _point_mass("box2d/one_sphere", FlowField("constant", vector=(0.8, 0.0))),
    "flow_circular": lambda: _point_mass("box2d/empty", FlowField("circular", magnitude=0.7)),
    "flow_saddle": lambda: _point_mass("box2d/small_sphere", FlowField("saddle", magnitude=0.9)),
    "terrain_ridge": lambda: _point_mass("box2d/two_spheres", Terrain("smooth_ridge", height=0.4)),
    "terrain_hill": lambda: _point_mass("box2d/three_small_spheres", Terrain("single_hill")),
}


def arm_problem(model: str = "rantf") -> Problem:
    """The 5-joint UR5e among a 0.2 m sphere. `model`: "rantf" (gravity drift as a torque direction, inertia metric),
    "n_riem" (inertia metric, no drift) or "n_eik" (neither)."""
    metric = {
        "rantf": ArmDynamics(
            lambda_reg=0.1, max_norm_g_sq=0.95, via_anisotropy=True, via_anisotropy_scale=ARM_VIA_SCALE
        ),
        "n_riem": ArmDynamics(lambda_reg=0.1, with_drift=False),
        "n_eik": None,
    }[model]
    return Problem(load_env("ur5e/5j_sphere"), ExpRamp(ARM_SPEED_SCALE * ARM_D_MAX), metric)


class ReferenceOnlineDrift(ArmDynamics):
    """The drift the reference's online controller used for a `via_anisotropy` model: G·b in joint radians, without
    the scaling to q, the global rescale or the clip applied in training. Only for reproducing the published table."""

    def drift(self, q: Tensor) -> Tensor:
        g, _ = ArmDynamics._evaluate(self, q)
        if "plain" not in self._cache:  # calibrating its drift scale takes seconds: do it once
            self._cache["plain"] = ArmDynamics(self.robot, self.dof, self.lambda_reg, self.max_norm_g_sq)
        plain = self._cache["plain"]
        b_rad = plain.drift(q) * torch.as_tensor(np.asarray(plain._cache["upper"]), dtype=q.dtype, device=q.device)
        return torch.einsum("bij,bj->bi", g, b_rad)
