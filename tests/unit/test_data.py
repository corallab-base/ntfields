from pathlib import Path

import torch

from ntfields import Problem, SceneCondition
from ntfields.data import NearSurface, PairDataset, SurfaceMix, sample_pairs


def test_samples_are_valid_and_deterministic(problem: Problem):
    data = sample_pairs(problem, 500, seed=1)
    assert len(data) == 500
    assert problem.env.is_valid(data.q_s).all() and problem.env.is_valid(data.q_g).all()
    torch.testing.assert_close(sample_pairs(problem, 500, seed=1).q_s, data.q_s)


def test_batches_cover_dataset(problem: Problem):
    data = sample_pairs(problem, 100)
    batches = list(data.batches(32, SceneCondition()))
    assert [b.q_s.shape[0] for b in batches] == [32, 32, 32, 4]
    assert torch.cat([b.speed_s for b in batches]).sort().values.equal(data.speed_s.sort().values)


def test_h5_round_trip(problem: Problem, tmp_path: Path):
    data = sample_pairs(problem, 50)
    data.save(tmp_path / "pairs.h5")
    loaded = PairDataset.load(tmp_path / "pairs.h5")
    torch.testing.assert_close(loaded.normal_g, data.normal_g)


def test_near_surface_pairs(problem: Problem):
    data = sample_pairs(problem, 300, NearSurface(margin=0.1, max_step=0.5))
    d = problem.env.distance(data.q_s)
    assert ((d > 0) & (d < 0.1)).all()
    assert ((data.q_g - data.q_s).norm(dim=-1) <= 0.5 + 1e-6).all()
    assert (data.q_g.abs() <= 1).all()


def test_surface_mix_pairs(problem: Problem):
    data = sample_pairs(problem, 400, SurfaceMix(surface_fraction=0.5, shell=0.2, perturb=0.5))
    assert problem.env.is_valid(data.q_s).all() and problem.env.is_valid(data.q_g).all()
    surface = data.q_s[200:], data.q_g[200:]
    near = torch.minimum(problem.env.distance(surface[0]), problem.env.distance(surface[1])) < 0.2
    assert near.all() and ((surface[0] - surface[1]).abs() <= 0.5 + 1e-6).all()
