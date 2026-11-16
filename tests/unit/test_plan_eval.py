import torch

from ntfields import Problem, TrainedField
from ntfields.envs import load_env
from ntfields.eval import is_collision_free, path_cost, path_length, travel_time_error
from ntfields.field import GoalField
from ntfields.gt import GridField, fmm_travel_time
from ntfields.methods.euclidean import Euclidean, EuclideanConfig
from ntfields.plan import mppi
from ntfields.registry import create
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
    torch.testing.assert_close(gt.travel_time(q), (q - gt.goal.double()).norm(dim=-1), atol=0.01, rtol=0)
    assert travel_time_error(_euclidean_field(problem), gt, q.float()) < 0.01


def test_grid_field_is_a_goal_field():
    gt: GoalField = fmm_travel_time(_empty(), torch.tensor([0.2, -0.3]), resolution=101)
    assert isinstance(gt, GridField)
    q = torch.tensor([[0.9, 0.9], [-0.7, 0.1]])
    assert gt.travel_time(q).dtype == q.dtype


def test_grid_field_gradient_in_free_space():
    """In free space at full speed, ∇T is the unit vector away from the goal."""
    goal = torch.tensor([0.2, -0.3])
    gt = fmm_travel_time(_empty(), goal, resolution=201)
    q = torch.tensor([[0.9, 0.9], [-0.7, 0.1], [0.6, -0.8]], dtype=torch.float64)
    away = torch.nn.functional.normalize(q - goal.double(), dim=-1)
    torch.testing.assert_close(gt.gradient(q), away, atol=0.03, rtol=0)
    assert not gt.gradient(q).requires_grad


def test_grid_field_gradient_matches_finite_differences(problem: Problem):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=101)
    q = torch.tensor([[-0.613, 0.057], [0.031, 0.607]], dtype=torch.float64)  # off grid lines, where it is smooth
    h = 1e-6
    fd = torch.stack(
        [(gt.travel_time(q + h * e) - gt.travel_time(q - h * e)) / (2 * h) for e in torch.eye(2, dtype=q.dtype)], dim=-1
    )
    torch.testing.assert_close(gt.gradient(q), fd, atol=1e-5, rtol=1e-5)


def test_fmm_detours_around_obstacle(problem: Problem):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=101)
    q = torch.tensor([[-0.6, 0.0]])
    assert gt.travel_time(q) > 1.2 * 1.2  # longer than the straight line, even at full speed


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


def test_towards_matches_pair_queries():
    """In float64: in float32, embedding the goal alone rather than in a batch changes the rounding."""
    torch.manual_seed(0)
    problem = _empty()
    for name in ("euclidean", "td_ntfields", "pntfields"):
        method = create(name)
        field = TrainedField(method, problem, method.build_model(problem).double().eval())
        q = torch.rand(16, 2, dtype=torch.float64) - 0.5
        goal = torch.tensor([0.3, -0.2], dtype=torch.float64)
        target = field.towards(goal)
        torch.testing.assert_close(target.travel_time(q), field.travel_time(q, goal.expand_as(q)), msg=name)
        torch.testing.assert_close(target.gradient(q), field.gradient(q, goal.expand_as(q))[0], msg=name)


def test_mppi_reaches_goal_in_free_space():
    field = _euclidean_field(_empty())
    start, goal = torch.tensor([-0.6, -0.3]), torch.tensor([0.5, 0.4])
    result = mppi(field, start, goal, generator=torch.Generator().manual_seed(0))
    assert result.success
    assert torch.equal(result.path[0], start) and torch.equal(result.path[-1], goal)
    assert path_length(result.path) < 1.1 * float((goal - start).norm())
