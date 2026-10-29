import torch

from ntfields.envs import SphereBoxEnv, load_env


def test_sphere_distance():
    env = load_env("box2d/one_sphere")
    q = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, -1.0]])
    torch.testing.assert_close(env.distance(q), torch.tensor([-0.3, 0.2, 0.7]))


def test_box_distance():
    env = SphereBoxEnv.build(bounds=[(-1, 1)] * 2, boxes=[((0.0, 0.0), (0.4, 0.2))])
    q = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.5, 0.4]])
    expected = torch.tensor([-0.1, 0.3, (0.3**2 + 0.3**2) ** 0.5])
    torch.testing.assert_close(env.distance(q), expected)


def test_is_valid():
    env = load_env("box2d/one_sphere")
    q = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.5, 0.0]])
    assert env.is_valid(q).tolist() == [False, True, False]


def test_empty_env_is_inf():
    env = load_env("box2d/empty")
    assert torch.isinf(env.distance(torch.zeros(3, 2))).all()
