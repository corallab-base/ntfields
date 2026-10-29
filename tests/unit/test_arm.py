import math
from pathlib import Path

import pytest
import torch

import ntfields
from ntfields import Problem, TrainConfig
from ntfields.data import UR5eSurfacePairs
from ntfields.envs import ArmEnv
from ntfields.envs.arm import ntrl_scenarios
from ntfields.eval import plan_succeeded
from ntfields.methods.td_ntfields import TDNTFieldsConfig
from ntfields.speed import ClippedEased

pytest.importorskip("pytorch_kinematics")

OBSTACLE = torch.tensor([[0.4, 0.2, 0.5, 0.1]])  # (x, y, z, r)


def test_sphere_and_point_distances_match_brute_force():
    torch.manual_seed(0)
    points = torch.rand(300, 3) - 0.5
    env = ArmEnv.ur5e(points=points, spheres=OBSTACLE, clearance_offset=0.01)
    q = torch.rand(16, 6) - 0.5
    c, r = env.arm_spheres(q)
    to_sphere = (c - OBSTACLE[0, :3]).norm(dim=-1) - OBSTACLE[0, 3]
    to_points = torch.cdist(c, points[None].expand(16, -1, -1)).amin(dim=-1)
    expected = (torch.minimum(to_sphere, to_points) - r).amin(dim=-1) - 0.01
    torch.testing.assert_close(env.distance(q), expected)


def test_base_joint_rotates_the_whole_arm():
    env = ArmEnv.ur5e()
    q = torch.rand(4, 6, dtype=torch.float64) - 0.5
    turned = q.clone()
    turned[:, 0] += 0.1  # θ += 0.2π about the base z axis
    a, s = math.cos(0.2 * math.pi), math.sin(0.2 * math.pi)
    rz = torch.tensor([[a, -s, 0.0], [s, a, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(env.arm_spheres(turned)[0], env.arm_spheres(q)[0] @ rz.T)
    assert env.arm_spheres(q)[0].shape == (4, 29, 3)


def test_normal_matches_finite_differences():
    problem = Problem(ArmEnv.ur5e(spheres=OBSTACLE), ClippedEased())
    q = torch.tensor([[0.05, -0.2, 0.1, -0.15, 0.1, 0.0]], dtype=torch.float64)
    _, normal = problem.speed_and_normal(q)
    d, eps = problem.env.distance, 1e-6
    fd = torch.stack([(d(q + eps * e) - d(q - eps * e)) / (2 * eps) for e in torch.eye(6, dtype=torch.float64)], dim=-1)
    torch.testing.assert_close(normal, torch.nn.functional.normalize(fd, dim=-1))


def test_obstacle_on_the_arm_is_a_collision():
    q = torch.zeros(1, 6)
    center = ArmEnv.ur5e().arm_spheres(q)[0][0, 10]
    env = ArmEnv.ur5e(spheres=torch.cat([center, torch.tensor([0.05])])[None])
    assert not env.is_valid(q).item()


def test_train_and_plan_on_the_arm(tmp_path: Path):
    problem = Problem(ArmEnv.ur5e_ntrl_scene(), ClippedEased(0.005, 0.05))
    cfg = TDNTFieldsConfig.arm()
    cfg.hidden = 32
    cfg.sampler = UR5eSurfacePairs(batch=100)
    tc = TrainConfig(n_pairs=64, batch_size=32, device="cpu", out_dir=tmp_path)
    field = ntfields.train("td_ntfields", problem, ntfields.Epochs(1), cfg=cfg, train_cfg=tc)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    starts, goals = ntrl_scenarios()
    assert isinstance(loaded.problem.env, ArmEnv) and loaded.problem.env.in_bounds(torch.cat([starts, goals])).all()
    assert isinstance(plan_succeeded(loaded.problem, loaded.plan(starts[0], goals[0])), bool)
