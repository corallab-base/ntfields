import numpy as np
import pytest
import torch

from ntfields import Problem
from ntfields.envs import GridEnv, SphereBoxEnv, load_env
from ntfields.grid import Interpolation
from ntfields.speed import Clipped


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


def _disc_grid(cell: float = 0.01, radius: float = 0.3, interp: Interpolation = "linear") -> GridEnv:
    """[-1, 1]^2 with the cells whose centres lie within `radius` of the origin occupied."""
    n = round(2 / cell)
    c = torch.linspace(-1 + cell / 2, 1 - cell / 2, n)
    x, y = torch.meshgrid(c, c, indexing="ij")
    return GridEnv.from_occupancy((x**2 + y**2 <= radius**2).numpy(), cell, origin=(-1, -1), interp=interp)


def test_grid_bounds_and_cells():
    env = GridEnv.from_occupancy(np.zeros((40, 20)), 0.05, origin=(1.0, -2.0))
    torch.testing.assert_close(env.bounds, torch.tensor([[1.0, 3.0], [-2.0, -1.0]]))
    torch.testing.assert_close(env.cell, torch.tensor([0.05, 0.05], dtype=torch.float64))
    assert env.dim == 2


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_grid_distance_matches_a_disc(interp: Interpolation):
    env = _disc_grid(interp=interp)
    q = torch.rand(2000, 2, generator=torch.Generator().manual_seed(0)) * 1.9 - 0.95
    exact = q.norm(dim=-1) - 0.3
    assert float((env.distance(q) - exact).abs().max()) < 0.01  # one cell


def test_grid_normals_point_away_from_the_disc():
    problem = Problem(_disc_grid(interp="cubic"), Clipped())
    q = torch.tensor([[0.5, 0.0], [0.0, -0.45], [-0.3, 0.3]])
    _, normal = problem.speed_and_normal(q)
    torch.testing.assert_close(normal, torch.nn.functional.normalize(q), atol=0.05, rtol=0)


def test_grid_is_valid_looks_up_the_cell():
    occupancy = np.zeros((4, 3), dtype=bool)
    occupancy[1, 2] = True  # covers [0.5, 1.0] x [1.0, 1.5]
    env = GridEnv.from_occupancy(occupancy, 0.5)
    q = torch.tensor([[0.75, 1.25], [0.49, 1.25], [0.75, 0.99], [2.0, 1.5], [2.01, 0.5], [-0.01, 0.5]])
    assert env.is_valid(q).tolist() == [False, True, True, True, False, False]


def test_grid_sdf_is_zero_on_cell_boundaries():
    occupancy = np.zeros((10, 10), dtype=bool)
    occupancy[5:, :] = True  # x >= 0.5 occupied
    env = GridEnv.from_occupancy(occupancy, 0.1)
    q = torch.tensor([[0.5, 0.42], [0.3, 0.5], [0.8, 0.5]])
    torch.testing.assert_close(env.distance(q), torch.tensor([0.0, 0.2, -0.3]), atol=1e-6, rtol=0)


def test_grid_without_obstacles_is_inf():
    env = GridEnv.from_occupancy(np.zeros((8, 8)), 0.25)
    assert torch.isinf(env.distance(torch.zeros(3, 2))).all()
    assert env.is_valid(torch.ones(3, 2)).all()


def test_grid_3d_box():
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[8:12, 8:12, 8:12] = True  # [0.8, 1.2]^3
    env = GridEnv.from_occupancy(occupancy, 0.1)
    q = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.5], [1.6, 1.0, 1.0]])
    # half a cell short at the centre: its nearest cell centres are 0.15 deep, not 0.2
    torch.testing.assert_close(env.distance(q), torch.tensor([-0.2, 0.3, 0.4]), atol=0.05 + 1e-6, rtol=0)
    assert env.is_valid(q).tolist() == [False, True, True]


def test_grid_rejects_1d():
    with pytest.raises(ValueError, match="2D or 3D"):
        GridEnv.from_occupancy(np.zeros(10), 0.1)
