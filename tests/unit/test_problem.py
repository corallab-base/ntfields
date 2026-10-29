import torch

from ntfields import Problem
from ntfields.envs import load_env
from ntfields.speed import Smoothstep


def test_normal_points_away_from_sphere():
    problem = Problem(load_env("box2d/one_sphere"), Smoothstep(dmax=0.5))
    q = torch.tensor([[0.45, 0.0], [0.0, -0.6], [0.3, 0.4]])
    s, n = problem.speed_and_normal(q)
    torch.testing.assert_close(n, torch.nn.functional.normalize(q, dim=-1))
    torch.testing.assert_close(s, problem.speed_at(q))


def test_normal_without_obstacles():
    problem = Problem(load_env("box2d/empty"), Smoothstep())
    s, n = problem.speed_and_normal(torch.zeros(4, 2))
    assert (s == 1).all() and (n == 0).all()
