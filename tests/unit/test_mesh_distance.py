import torch

from ntfields.envs.mesh_distance import MeshDistance, point_triangle_distance


def _unit_square() -> torch.Tensor:
    """The square [0, 1]² at z = 0 as two triangles."""
    v = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    return v[torch.tensor([[0, 1, 2], [0, 2, 3]])]


def test_point_triangle_distance_regions():
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float64)
    p = torch.tensor([[0.2, 0.2, 0.5], [-1, -1, 0], [2, 0, 1], [0.5, 0.5, 0], [0.6, 0.6, 0.1], [0.5, -2, 0]])
    expected = [0.5, 2**0.5, 2**0.5, 0.0, (0.1**2 * 2 + 0.01) ** 0.5, 2.0]
    torch.testing.assert_close(point_triangle_distance(p.double(), tri), torch.tensor(expected, dtype=torch.float64))


def test_grid_matches_exact_distance():
    tri = _unit_square()
    mesh = MeshDistance.build(tri, spacing=0.02, cap=0.2, exact_below=0.05)
    p = torch.rand(500, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 1.6 - 0.3
    exact = point_triangle_distance(p, tri).clamp_max(0.2)
    torch.testing.assert_close(mesh(p), exact, atol=0.02 / 4, rtol=0)  # trilinear error near edges
    near = exact < 0.05
    torch.testing.assert_close(mesh(p)[near], exact[near])  # recomputed exactly near the surface
    assert mesh(torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)).item() == 0.2  # beyond the grid: the cap
