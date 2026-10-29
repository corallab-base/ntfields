import numpy as np
import pytest
import torch

from ntfields import Problem
from ntfields.envs import ArmEnv
from ntfields.speed import ClippedEased
from tests.fidelity.ntrl import UR5_DATA, requires_ntrl

pytest.importorskip("pytorch_kinematics")


@requires_ntrl
def test_clearance_and_normals_match_ntrl_data():
    """ntrl-demo stores clamp(d / 0.05, 0.1, 1) and the unit clearance normal for each UR5 sample."""
    keys = ("sampled_points", "speed", "normal")
    x, speed, normal = (torch.from_numpy(np.load(UR5_DATA / f"{k}.npy")[:2000]).float() for k in keys)
    problem = Problem(ArmEnv.ur5e_ntrl_scene(), ClippedEased(0.005, 0.05))
    for side in (0, 1):
        q = x[:, 6 * side : 6 * side + 6]
        d = problem.env.distance(q)
        torch.testing.assert_close((d / 0.05).clamp(0.1, 1), speed[:, side], atol=1e-5, rtol=0)
        _, n = problem.speed_and_normal(q)
        cos = (n * normal[:, 6 * side : 6 * side + 6]).sum(dim=-1)
        # where two arm spheres are equally close (to ~1e-8 m) rounding may pick the other one
        assert (cos > 0.999).float().mean() > 0.999
