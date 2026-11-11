import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.data import sample_valid
from ntfields.problem import Problem


@dataclass
class JointSpaceMix:
    """The reference's arm pairs: `uniform_fraction` independent valid starts and goals; the rest anchor a goal with
    clearance in (`shift`, `shift` + `margin`) and put the start a uniform distance up to `max_step` away in a random
    direction of joint angles (θ = q · env.scale), in bounds and with clearance above `shift`."""

    uniform_fraction: float = 0.5
    margin: float = 0.05
    shift: float = 0.01
    max_step: float = math.pi  # radians

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        n_uniform = round(n * self.uniform_fraction)
        u_s, u_g = sample_valid(problem, n_uniform, generator), sample_valid(problem, n_uniform, generator)
        env, bounds = problem.env, problem.env.bounds
        scale = env.scale(bounds) if hasattr(env, "scale") else torch.ones(())  # type: ignore[attr-defined]
        lo, hi = bounds.unbind(-1)
        anchors, partners, count = [], [], 0
        for _ in range(1000):
            if count >= n - n_uniform:
                break
            m = 4 * (n - n_uniform - count) + 64
            x0 = lo + (hi - lo) * torch.rand(m, len(lo), generator=generator)
            direction = torch.nn.functional.normalize(torch.randn(m, len(lo), generator=generator), dim=-1)
            x1 = x0 + direction * self.max_step * torch.rand(m, 1, generator=generator) / scale
            keep = ((x1 >= lo) & (x1 <= hi)).all(-1)
            x0, x1 = x0[keep], x1[keep]
            d0 = problem.env.distance(x0.to(device)).cpu() - self.shift
            d1 = problem.env.distance(x1.to(device)).cpu() - self.shift
            keep = (d0 > 0) & (d0 < self.margin) & (d1 > 0)
            anchors.append(x0[keep])
            partners.append(x1[keep])
            count += int(keep.sum())
        else:
            raise RuntimeError(f"only {count}/{n - n_uniform} near-surface pairs; is the band empty?")
        k = n - n_uniform
        return torch.cat([u_s, torch.cat(partners)[:k]]), torch.cat([u_g, torch.cat(anchors)[:k]])
