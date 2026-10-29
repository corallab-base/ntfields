import pytest

from ntfields import Problem
from ntfields.envs import load_env
from ntfields.speed import Clipped


@pytest.fixture
def problem() -> Problem:
    return Problem(load_env("box2d/one_sphere"), Clipped(dmin=0.01, dmax=0.1))
