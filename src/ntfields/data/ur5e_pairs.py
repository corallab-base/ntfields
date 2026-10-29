import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ntfields.envs.ur5e_ik import rpy_pose, solve_ur5e_ik
from ntfields.problem import Problem

Box = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
# the cabinet box: the ntrl scene's bounding box intersected with the UR5e's reach (m, base frame)
CABINET_BOX: Box = ((-0.898296594619751, -0.431555), (-0.872116, 0.808328), (-0.126332, 1.0845240354537964))


@dataclass
class UR5eSurfacePairs:
    """UR5e surface pairs, following ntrl-demo's `arm_append_list` with the goal also checked for collisions.

    Starts: every IK branch of random wrist poses (position uniform in `box`, roll/pitch/yaw within
    `orientation_spread` of `orientation_center`), kept if 0 < d < margin. Goals: the start plus a random direction
    times U(0, perturb), kept inside the bounds and collision-free. Needs a UR5e `ArmEnv` (q = θ / 2π).
    """

    box: Box = CABINET_BOX
    orientation_center: tuple[float, float, float] = (-math.pi / 2, 0.0, math.pi / 2)
    orientation_spread: float = 0.3 * math.pi
    margin: float = 0.05
    perturb: float = 0.5  # q units: π rad
    batch: int = 2000  # wrist poses per round
    max_rounds: int = 1000

    def sample(
        self, problem: Problem, n: int, generator: torch.Generator, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        env, bounds = problem.env, problem.env.bounds
        starts: list[Tensor] = []
        goals: list[Tensor] = []
        count = 0
        for _ in range(self.max_rounds):
            if count >= n:
                break
            q_s = self._ik_starts(generator)
            q_s = q_s[_in_bounds(bounds, q_s)]
            direction = torch.nn.functional.normalize(
                torch.randn(q_s.shape, generator=generator, dtype=torch.float64), dim=-1
            )
            q_g = q_s + direction * self.perturb * torch.rand(len(q_s), 1, generator=generator, dtype=torch.float64)
            keep = _in_bounds(bounds, q_g)
            q_s, q_g = q_s[keep].float(), q_g[keep].float()
            d_s, d_g = env.distance(q_s.to(device)).cpu(), env.distance(q_g.to(device)).cpu()
            keep = (d_s > 0) & (d_s < self.margin) & (d_g > 0)
            starts.append(q_s[keep])
            goals.append(q_g[keep])
            count += int(keep.sum())
        else:
            raise RuntimeError(f"only {count}/{n} pairs after {self.max_rounds} rounds; is an obstacle near the box?")
        return torch.cat(starts)[:n], torch.cat(goals)[:n]

    def _ik_starts(self, generator: torch.Generator) -> Tensor:
        """(8 · batch, 6) IK solutions in q units (float64)."""
        lo, hi = torch.tensor(self.box, dtype=torch.float64).unbind(-1)
        position = lo + (hi - lo) * torch.rand(self.batch, 3, generator=generator, dtype=torch.float64)
        center = torch.tensor(self.orientation_center, dtype=torch.float64)
        angles = center + self.orientation_spread * (
            2 * torch.rand(self.batch, 3, generator=generator, dtype=torch.float64) - 1
        )
        poses = rpy_pose(position, angles[:, 0], angles[:, 1], angles[:, 2])
        return solve_ur5e_ik(poses).reshape(-1, 6) / (2 * math.pi)


def _in_bounds(bounds: Tensor, q: Tensor) -> Tensor:
    b = bounds.to(q)
    return ((q >= b[:, 0]) & (q <= b[:, 1])).all(dim=-1)
