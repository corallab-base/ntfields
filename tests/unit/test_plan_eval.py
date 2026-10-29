import torch

from ntfields import Problem, TrainedField
from ntfields.envs import load_env
from ntfields.eval import is_collision_free, path_cost, path_length, travel_time_error
from ntfields.gt import fmm_travel_time
from ntfields.methods.euclidean import Euclidean, EuclideanConfig
from ntfields.plan import mppi
from ntfields.speed import Clipped


def _empty() -> Problem:
    return Problem(load_env("box2d/empty"), Clipped())


def _euclidean_field(problem: Problem) -> TrainedField:
    method = Euclidean(EuclideanConfig())
    return TrainedField(method, problem, method.build_model(problem))


def test_fmm_free_space_is_euclidean():
    problem = _empty()
    gt = fmm_travel_time(problem, torch.tensor([0.2, -0.3]), resolution=101)
    q = torch.tensor([[0.9, 0.9], [-0.7, 0.1], [0.25, -0.3]], dtype=torch.float64)
    torch.testing.assert_close(gt.value(q), (q - gt.goal.double()).norm(dim=-1), atol=0.01, rtol=0)
    assert travel_time_error(_euclidean_field(problem), gt, q.float()) < 0.01


def test_fmm_detours_around_obstacle(problem: Problem):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=101)
    q = torch.tensor([[-0.6, 0.0]])
    assert gt.value(q) > 1.2 * 1.2  # longer than the straight line, even at full speed


def test_path_metrics(problem: Problem):
    line = torch.stack([torch.linspace(-0.9, 0.9, 10), torch.full((10,), 0.8)], dim=-1)
    assert abs(path_length(line) - 1.8) < 1e-5
    assert abs(path_cost(_empty(), line) - 1.8) < 1e-5
    assert is_collision_free(problem, line)
    assert not is_collision_free(problem, line - torch.tensor([0.0, 0.8]))


def test_planner_meets_in_free_space():
    field = _euclidean_field(_empty())
    start, goal = torch.tensor([-0.8, -0.5]), torch.tensor([0.7, 0.6])
    result = field.plan(start, goal)
    assert result.success
    assert torch.equal(result.path[0], start) and torch.equal(result.path[-1], goal)
    assert path_length(result.path) < 1.01 * float((goal - start).norm()) + 0.06


def test_mppi_reaches_goal_in_free_space():
    field = _euclidean_field(_empty())
    start, goal = torch.tensor([-0.6, -0.3]), torch.tensor([0.5, 0.4])
    result = mppi(field, start, goal, generator=torch.Generator().manual_seed(0))
    assert result.success
    assert torch.equal(result.path[0], start) and torch.equal(result.path[-1], goal)
    assert path_length(result.path) < 1.1 * float((goal - start).norm())
