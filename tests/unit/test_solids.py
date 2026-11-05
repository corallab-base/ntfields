import math

import pytest
import torch

from ntfields.bench.decomposition import evaluate_boxes
from ntfields.bench.protocol import Protocol, evaluate_path, is_valid
from ntfields.bench.rrt import bidirectional_rrt
from ntfields.bench.suite import SuiteProblem
from ntfields.envs import ArmEnv
from ntfields.envs.solids import SolidScene, quat_matrix, winding_number

IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _box(center: list[float], size: list[float], quat: list[float] = IDENTITY) -> dict:
    return {"id": "b", "type": "box", "center": center, "size": size, "quat": quat}


def _cylinder(center: list[float], height: float, radius: float) -> dict:
    return {"id": "c", "type": "cylinder", "center": center, "quat": IDENTITY, "height": height, "radius": radius}


def test_quat_matrix_turns_about_z():
    half = math.sqrt(0.5)
    rotation = quat_matrix(torch.tensor([0.0, 0.0, half, half]))
    torch.testing.assert_close(rotation @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6, rtol=0)


def test_box_distance_is_signed():
    scene = SolidScene.from_json([_box([0.0, 0, 0], [2.0, 2, 2])])
    p = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [3.0, 0, 0], [2.0, 2, 1]])
    torch.testing.assert_close(scene(p), torch.tensor([-1.0, -0.5, 2.0, 2**0.5]))


def test_rotated_box_distance():
    half = math.sqrt(0.5)
    scene = SolidScene.from_json([_box([0.0, 0, 0], [2.0, 0.2, 0.2], [0.0, 0.0, half, half])])  # long axis along y
    torch.testing.assert_close(scene(torch.tensor([[0.0, 2, 0], [2.0, 0, 0]])), torch.tensor([1.0, 1.9]))


def test_cylinder_distance_is_signed():
    scene = SolidScene.from_json([_cylinder([0.0, 0, 0], 2.0, 1.0)])
    p = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0.0, 0, 3.0], [4.0, 0, 5.0]])
    torch.testing.assert_close(scene(p), torch.tensor([-1.0, 1.0, 2.0, 5.0]))


def test_triangles_are_closed_with_the_winding_number():
    box = SolidScene.from_json([_box([0.5, 0, 0], [1.0, 1, 1])]).triangles()
    p = torch.tensor([[0.5, 0, 0], [2.0, 0, 0]])
    torch.testing.assert_close(winding_number(p, box).abs(), torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0)
    cylinder = SolidScene.from_json([_cylinder([0.0, 0, 0], 1.0, 0.5)]).triangles()
    torch.testing.assert_close(winding_number(p[:1] * 0, cylinder).abs(), torch.tensor([1.0]), atol=1e-5, rtol=0)


def test_mesh_distance_matches_the_box_it_came_from_and_is_negative_inside():
    objects = [_box([0.0, 0, 0], [1.0, 1, 1])]
    mesh = SolidScene.from_json(
        [{"id": "m", "type": "mesh", "triangles": SolidScene.from_json(objects).triangles().tolist()}]
    )
    p = torch.tensor([[0.1, 0.1, 0.1], [0.0, 0, 0], [2.0, 0, 0], [0.8, 0.8, 0]])
    torch.testing.assert_close(mesh(p), SolidScene.from_json(objects)(p), atol=1e-5, rtol=0)


def test_rrt_goes_around_a_wall():
    def free(q: torch.Tensor) -> torch.Tensor:
        return ~((q[:, 0].abs() < 0.1) & (q[:, 1] < 0.6))

    start, goal = torch.tensor([-0.8, 0.0]), torch.tensor([0.8, 0.0])
    lo, hi = torch.tensor([-1.0, -1]), torch.tensor([1.0, 1])
    generator = torch.Generator().manual_seed(0)
    path = bidirectional_rrt(free, start, goal, lo, hi, generator, resolution=0.01, step=0.2)
    assert path is not None
    torch.testing.assert_close(path[0], start)
    torch.testing.assert_close(path[-1], goal)
    assert path[:, 1].amax() >= 0.6
    assert free(path).all()


def test_rrt_reports_a_sealed_goal():
    def free(q: torch.Tensor) -> torch.Tensor:
        return q[:, 0].abs() > 0.1

    lo, hi = torch.tensor([-1.0, -1]), torch.tensor([1.0, 1])
    generator = torch.Generator().manual_seed(0)
    path = bidirectional_rrt(
        free, torch.tensor([-0.8, 0.0]), torch.tensor([0.8, 0.0]), lo, hi, generator, 0.01, rounds=20
    )
    assert path is None


def test_box_fit_scores():
    pytest.importorskip("pytorch_kinematics")
    objects = [_box([0.6, 0, 0.2], [0.2, 0.2, 0.2])]
    generator = torch.Generator().manual_seed(0)
    exact = evaluate_boxes(objects, objects, 0.0, generator, n_points=20_000, n_configs=200)
    assert exact.iou == 1 and exact.coverage == 1 and exact.free_blocked == 0
    half = _box([0.55, 0, 0.2], [0.1, 0.2, 0.2])
    fit = evaluate_boxes(objects, [half], 0.0, generator, n_points=20_000, n_configs=200)
    assert fit.coverage == pytest.approx(0.5, abs=0.05)


def test_self_collision_is_an_option():
    pytest.importorskip("pytorch_kinematics")
    scene = {"kind": "arm", "robot": "ur5e_2f85", "objects": [_box([5.0, 0, 0], [0.1, 0.1, 0.1])]}
    problem = SuiteProblem("p", scene, {"kind": "clipped", "dmin": 0.005, "dmax": 0.05}, {}).problem()
    env = problem.env
    assert isinstance(env, ArmEnv)
    torch.manual_seed(0)
    q = torch.rand(3000, 6) - 0.5
    plain, strict = is_valid(problem, q, 0.0), is_valid(problem, q, 0.0, self_collision=True)
    assert plain.all()  # the only obstacle is out of reach
    assert (~strict).any() and strict.any()
    bad, good = q[~strict][0], q[strict][0]
    off = Protocol(0.0, 0.0025)
    on = Protocol(0.0, 0.0025, self_collision=True)
    assert evaluate_path(problem, off, bad, bad, torch.stack([bad, bad])).success
    assert not evaluate_path(problem, on, bad, bad, torch.stack([bad, bad])).success
    assert evaluate_path(problem, on, good, good, torch.stack([good, good])).success


def test_self_collision_needs_an_arm():
    problem = SuiteProblem(
        "p", {"kind": "sphere_box", "bounds": [[-1, 1]] * 2}, {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}, {}
    ).problem()
    with pytest.raises(ValueError, match="needs an arm"):
        is_valid(problem, torch.zeros(1, 2), 0.0, self_collision=True)
