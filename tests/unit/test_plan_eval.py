import pytest
import torch

from ntfields import Problem, TrainedField
from ntfields.envs import GridEnv, load_env
from ntfields.eval import is_collision_free, path_cost, path_length, plan_succeeded, travel_time_error
from ntfields.field import Field, GoalField
from ntfields.grid import Interpolation
from ntfields.gt import FMMField, GridField, fmm_travel_time
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
    assert travel_time_error(_euclidean_field(problem), gt, gt.goal, q.float()) < 0.01


def test_grid_field_is_a_goal_field():
    gt: GoalField = fmm_travel_time(_empty(), torch.tensor([0.2, -0.3]), resolution=101)
    assert isinstance(gt, GridField)
    q = torch.tensor([[0.9, 0.9], [-0.7, 0.1]])
    assert gt.travel_time(q).dtype == q.dtype


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_grid_field_gradient_in_free_space(interp: Interpolation):
    """In free space at full speed, ∇T is the unit vector away from the goal."""
    goal = torch.tensor([0.2, -0.3])
    gt = fmm_travel_time(_empty(), goal, resolution=201, interp=interp)
    q = torch.tensor([[0.9, 0.9], [-0.7, 0.1], [0.6, -0.8]], dtype=torch.float64)
    away = torch.nn.functional.normalize(q - goal.double(), dim=-1)
    torch.testing.assert_close(gt.gradient(q), away, atol=0.03, rtol=0)
    assert not gt.gradient(q).requires_grad


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_grid_field_gradient_matches_finite_differences(problem: Problem, interp: Interpolation):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=101, interp=interp)
    q = torch.tensor([[-0.613, 0.057], [0.031, 0.607]], dtype=torch.float64)  # off grid lines, where it is smooth
    h = 1e-6
    fd = torch.stack(
        [(gt.travel_time(q + h * e) - gt.travel_time(q - h * e)) / (2 * h) for e in torch.eye(2, dtype=q.dtype)], dim=-1
    )
    torch.testing.assert_close(gt.gradient(q), fd, atol=1e-5, rtol=1e-5)


def test_cubic_passes_through_nodes(problem: Problem):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=51)
    nodes = torch.tensor([[-0.6, 0.12], [0.36, -0.8], [-1.0, 1.0]], dtype=torch.float64)  # on the 0.04 grid
    ij = ((nodes + 1) / 0.04).round().long()
    torch.testing.assert_close(gt.travel_time(nodes), gt.values[ij[:, 0], ij[:, 1]])


def test_cubic_gradient_is_continuous_across_cell_edges(problem: Problem):
    """Linear interpolation's gradient jumps at a grid line; Catmull-Rom's doesn't."""
    edge = torch.tensor([[-0.6, 0.12]], dtype=torch.float64)  # x = -0.6 is a grid line at resolution 51
    across = torch.tensor([[1e-7, 0.0]], dtype=torch.float64)
    jump = {}
    for interp in ("linear", "cubic"):
        gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=51, interp=interp)
        jump[interp] = float((gt.gradient(edge + across) - gt.gradient(edge - across)).norm())
    assert jump["linear"] > 1e-2
    assert jump["cubic"] < 1e-5


def test_cubic_reproduces_linear_functions_in_3d():
    axes = [torch.linspace(-1, 1, 11, dtype=torch.float64)] * 3
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    a = torch.tensor([0.5, -2.0, 1.5], dtype=torch.float64)
    gt = GridField(a[0] * x + a[1] * y + a[2] * z, torch.tensor([[-1.0, 1.0]] * 3), torch.zeros(3))
    q = torch.rand(32, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 1.6 - 0.8  # interior
    torch.testing.assert_close(gt.travel_time(q), q @ a)
    torch.testing.assert_close(gt.gradient(q), a.expand_as(q))


def test_fmm_detours_around_obstacle(problem: Problem):
    gt = fmm_travel_time(problem, torch.tensor([0.6, 0.0]), resolution=101)
    q = torch.tensor([[-0.6, 0.0]])
    assert gt.travel_time(q) > 1.2 * 1.2  # longer than the straight line, even at full speed


def test_fmm_field_pair_queries_match_towards(problem: Problem):
    field: Field = FMMField(problem, resolution=101)
    goals = torch.tensor([[0.6, 0.0], [-0.5, 0.5]])
    q_s = torch.tensor([[-0.6, 0.0], [0.7, 0.7], [0.1, -0.8], [-0.2, 0.6]])
    q_g = goals[[0, 1, 0, 1]]
    expected = torch.stack([field.towards(g).travel_time(q[None])[0] for q, g in zip(q_s, q_g, strict=True)])
    torch.testing.assert_close(field.travel_time(q_s, q_g), expected)
    g_s, _ = field.gradient(q_s, q_g)
    torch.testing.assert_close(
        g_s, torch.cat([field.towards(g).gradient(q[None]) for q, g in zip(q_s, q_g, strict=True)])
    )


def test_fmm_field_caches_per_goal(problem: Problem):
    field = FMMField(problem, resolution=51, cache_size=2)
    a, b, c = torch.tensor([0.6, 0.0]), torch.tensor([-0.5, 0.5]), torch.tensor([0.0, -0.7])
    first = field.towards(a)
    assert field.towards(a.clone()) is first
    field.towards(b)
    field.towards(c)
    assert field.towards(a) is not first  # evicted


def test_fmm_field_is_symmetric_and_goal_gradient_points_away_from_start():
    field = FMMField(_empty(), resolution=201)
    q_s, q_g = torch.tensor([[-0.6, -0.3]]), torch.tensor([[0.5, 0.4]])
    torch.testing.assert_close(field.travel_time(q_s, q_g), field.travel_time(q_g, q_s), atol=0.01, rtol=0)
    g_s, g_g = field.gradient(q_s, q_g)
    torch.testing.assert_close(g_s, torch.nn.functional.normalize(q_s - q_g), atol=0.03, rtol=0)
    torch.testing.assert_close(g_g, torch.nn.functional.normalize(q_g - q_s), atol=0.03, rtol=0)


@pytest.mark.parametrize("planner", ["mppi", "gradient_bidirectional"])
def test_fmm_field_plans_around_obstacle(problem: Problem, planner: str):
    field = FMMField(problem, resolution=101)
    result = field.plan(torch.tensor([-0.6, 0.1]), torch.tensor([0.6, -0.1]), planner)
    assert plan_succeeded(problem, result)


def test_fmm_on_a_grid_env_matches_the_sphere_env(problem: Problem):
    """The one_sphere scene rasterized at 0.01: FMM on it agrees with FMM on the analytic scene, and plans in it."""
    c = torch.linspace(-0.995, 0.995, 200)
    x, y = torch.meshgrid(c, c, indexing="ij")
    grid = Problem(GridEnv.from_occupancy((x**2 + y**2 <= 0.09).numpy(), 0.01, origin=(-1, -1)), problem.speed)
    goal = torch.tensor([0.6, -0.1])
    q = torch.tensor([[-0.6, 0.35], [-0.8, -0.6], [0.0, 0.7]])
    t_grid = fmm_travel_time(grid, goal, resolution=101).travel_time(q)
    torch.testing.assert_close(t_grid, fmm_travel_time(problem, goal, resolution=101).travel_time(q), rtol=0.02, atol=0)
    assert plan_succeeded(grid, FMMField(grid, resolution=101).plan(q[0], goal))


def test_travel_time_error_takes_any_goal_field(problem: Problem):
    goal = torch.tensor([0.6, 0.0])
    fmm = FMMField(problem, resolution=101)
    starts = torch.tensor([[-0.6, 0.0], [0.0, 0.7], [0.5, -0.5]])
    assert travel_time_error(fmm, fmm.towards(goal), goal, starts) == 0.0
    assert travel_time_error(_euclidean_field(problem), fmm.towards(goal), goal, starts) > 0.1  # ignores the obstacle


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
