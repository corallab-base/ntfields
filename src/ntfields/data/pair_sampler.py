from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from ntfields.data.pair_dataset import PairDataset
from ntfields.problem import Problem


class Sampler(Protocol):
    """Draws n (start, goal) pairs; `device` is where the environment may be evaluated (results come back on CPU)."""

    def sample(self, problem: Problem, n: int, generator: torch.Generator, device: str) -> tuple[Tensor, Tensor]: ...


@dataclass
class Uniform:
    """Starts and goals uniform over the free space."""

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        return sample_valid(problem, n, generator), sample_valid(problem, n, generator)


@dataclass
class NearSurface:
    """TD-NTFields: starts with min_clearance < d < margin (margin should match the speed's dmax), goals a uniform
    distance up to `max_step` away in a random direction, inside the bounds."""

    margin: float = 0.1
    min_clearance: float = 0.0
    goal_min_clearance: float | None = None  # None: goal clearance unchecked (the arm reference)
    max_step: float | None = 0.5  # None: the diagonal of the bounds

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        bounds = problem.env.bounds
        max_step = self.max_step or float((bounds[:, 1] - bounds[:, 0]).norm())

        def draw(m: int) -> Tensor:
            q_s = _uniform(bounds, m, generator)
            d = problem.env.distance(q_s.to(device)).cpu()
            q_s = q_s[(d > self.min_clearance) & (d < self.margin)]
            direction = torch.nn.functional.normalize(torch.rand(q_s.shape, generator=generator) - 0.5, dim=-1)
            q_g = q_s + direction * max_step * torch.rand(q_s.shape[0], 1, generator=generator)
            keep = _in_bounds(bounds, q_g)
            if self.goal_min_clearance is not None:
                keep &= problem.env.distance(q_g.to(device)).cpu() > self.goal_min_clearance
            return torch.cat([q_s, q_g], dim=-1)[keep]

        pairs = _rejection_sample(draw, n)
        return pairs[:, : problem.dim], pairs[:, problem.dim :]


@dataclass
class SurfaceMix:
    """Mixed pairs: `1 - surface_fraction` uniform free pairs; the rest have one end within `shell` of an
    obstacle and the other a uniform offset in [-perturb, perturb]^dim from it (clamped to the bounds, free), the
    surface end on the start or goal side with equal probability.

    Surface ends come from `anchor`: "project" moves free points along the clearance gradient to a clearance
    uniform in (0, shell) (needs clearance in configuration units, e.g. a
    point mass); "reject" keeps uniform configurations with 0 < d < shell, as its arm sampler does.
    """

    surface_fraction: float = 0.5
    shell: float = 0.2
    perturb: float = 0.5
    anchor: Literal["project", "reject"] = "project"

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        bounds = problem.env.bounds
        n_surface = round(n * self.surface_fraction)

        def draw(m: int) -> Tensor:
            if self.anchor == "project":
                q = sample_valid(problem, m, generator)
                _, normal = problem.speed_and_normal(q)
                target = self.shell * torch.rand(m, 1, generator=generator)
                a = q - (problem.env.distance(q)[:, None] - target) * normal
                a = a[problem.env.is_valid(a)]
            else:
                a = _uniform(bounds, m, generator)
                d = problem.env.distance(a)
                a = a[(d > 0) & (d < self.shell)]
            b = (a + self.perturb * (2 * torch.rand(a.shape, generator=generator) - 1)).clamp(
                bounds[:, 0], bounds[:, 1]
            )
            return torch.cat([a, b], dim=-1)[problem.env.is_valid(b)]

        pairs = _rejection_sample(draw, n_surface)
        swap = torch.rand(n_surface, 1, generator=generator) < 0.5
        a, b = pairs[:, : problem.dim], pairs[:, problem.dim :]
        u_s, u_g = Uniform().sample(problem, n - n_surface, generator, device)
        return torch.cat([u_s, torch.where(swap, b, a)]), torch.cat([u_g, torch.where(swap, a, b)])


def sample_valid(problem: Problem, n: int, generator: torch.Generator | None = None) -> Tensor:
    """n uniform samples inside the bounds with positive clearance."""

    def draw(m: int) -> Tensor:
        q = _uniform(problem.env.bounds, m, generator)
        return q[problem.env.is_valid(q)]

    return _rejection_sample(draw, n)


def sample_pairs(
    problem: Problem, n: int, sampler: Sampler | None = None, seed: int = 0, device: str = "cpu"
) -> PairDataset:
    """n start/goal pairs with S* and the clearance normal at both ends, evaluated on `device`; returned on CPU."""
    generator = torch.Generator().manual_seed(seed)
    q_s, q_g = (sampler or Uniform()).sample(problem, n, generator, device)
    speed_s, normal_s = problem.speed_and_normal(q_s.to(device))
    speed_g, normal_g = problem.speed_and_normal(q_g.to(device))
    return PairDataset(q_s, q_g, speed_s, speed_g, normal_s, normal_g).to("cpu")


def _uniform(bounds: Tensor, n: int, generator: torch.Generator | None) -> Tensor:
    lo, hi = bounds.unbind(-1)
    return lo + (hi - lo) * torch.rand(n, bounds.shape[0], generator=generator)


def _in_bounds(bounds: Tensor, q: Tensor) -> Tensor:
    return ((q >= bounds[:, 0]) & (q <= bounds[:, 1])).all(dim=-1)


def _rejection_sample(draw: Callable[[int], Tensor], n: int, max_rounds: int = 100) -> Tensor:
    chunks: list[Tensor] = []
    count = 0
    for _ in range(max_rounds):
        chunks.append(draw(2 * n))
        count += chunks[-1].shape[0]
        if count >= n:
            return torch.cat(chunks)[:n]
    raise RuntimeError(f"only found {count}/{n} samples; is the sampled region empty?")
