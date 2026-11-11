import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_kinematics.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from torch import Tensor

from ntfields.envs.arm import ArmEnv
from ntfields.envs.constraint import ConstrainedEnv
from ntfields.envs.solids import SolidScene
from ntfields.envs.ur5e_ik import solve_ur5e_ik
from ntfields.problem import Problem
from ntfields.speed import GaussianBand

CUP_SCENE = Path(__file__).parents[1] / "assets" / "cup" / "table_boxes.json"
IK_SOLVERS = {"ur5e": solve_ur5e_ik}


@dataclass
class TSRConstraint:
    """A task space region (Berenson et al.): the end-effector pose stays within a box in a world frame.

    Displacements are (x, y, z, roll, pitch, yaw) with intrinsic XYZ angles, of the pose T_o_w⁻¹ · T_o_e · T_w_e⁻¹.
    d_M is the norm of the displacement's excess over `bounds`, in mixed metres and radians, as in ntfields-old's
    `TSRSampler.distance`. Samples are IK solutions of poses drawn in `bounds` widened by `offsets`, and `self_clear`
    drops self-colliding ones.

    `branches` are the IK branches to sample from, and `joints` optionally narrows the samples to a box in q. Analytic
    branches fix the sign of θ5, so one branch jumps by π in θ4 and θ6 where a path crosses the wrist singularity
    θ5 = 0: a connected sheet of solutions needs the branches either side of it, with `joints` choosing the sign of θ6.
    Different sheets do not connect within the band, so a travel time between two of them is meaningless.
    """

    arm: ArmEnv
    bounds: Tensor  # (6, 2), lower and upper
    offsets: Tensor  # (6,)
    T_o_w: Tensor  # (4, 4) world (TSR) frame in the arm's base frame
    T_w_e: Tensor  # (4, 4) end link in the frame being constrained
    ik: str = "ur5e"
    branches: tuple[int, ...] = (0,)
    joints: Tensor | None = None  # (dof, 2) in q, lower and upper
    self_clear: bool = True
    batch: int = 4096  # poses per round of `sample`
    max_rounds: int = 2000

    def displacement(self, q: Tensor) -> Tensor:
        """(B, 6) pose displacements."""
        return _displacement(self.arm.end_pose(q), self.T_o_w.to(q), self.T_w_e.to(q))

    def violation(self, q: Tensor) -> Tensor:
        return box_excess(self.displacement(q), self.bounds.to(q)).norm(dim=-1)

    def sample(self, n: int, generator: torch.Generator, device: str = "cpu") -> Tensor:
        lo, hi = (self.bounds[:, 0] - self.offsets).double(), (self.bounds[:, 1] + self.offsets).double()
        branches = torch.tensor(self.branches)
        out: list[Tensor] = []
        count = 0
        for _ in range(self.max_rounds):
            if count >= n:
                return torch.cat(out)[:n]
            d = lo + (hi - lo) * torch.rand(self.batch, 6, generator=generator, dtype=torch.float64)
            poses = self._poses(d.to(device))
            q = IK_SOLVERS[self.ik](poses)[:, branches.to(device)]  # (batch, k, dof), radians
            pick = torch.randint(len(self.branches), (len(poses),), generator=generator).to(device)
            q = q[torch.arange(len(q), device=device), pick]
            q = q / self.arm.scale(q)
            fk = self.arm.end_pose(q)
            good = q.isfinite().all(-1) & self.arm.in_bounds(q) & _pose_close(fk, poses)
            if self.joints is not None:
                joints = self.joints.to(q)
                good &= ((q >= joints[:, 0]) & (q <= joints[:, 1])).all(-1)
            q = q[good].float()
            if self.self_clear and len(q):
                q = q[self.arm.self_distance(q) > 0]
            out.append(q.cpu())
            count += out[-1].shape[0]
        raise RuntimeError(f"only {count}/{n} samples after {self.max_rounds} rounds; is the TSR reachable?")

    def tangent(self, q: Tensor, v: Tensor) -> Tensor:
        """v minus its part along the gradients of the equality dimensions of `bounds` (signed: the hinge in
        `violation` has no gradient on the manifold)."""
        equal = (self.bounds[:, 0] == self.bounds[:, 1]).nonzero()[:, 0].to(q.device)
        with torch.enable_grad():
            x = q.detach().requires_grad_(True)
            r = self.displacement(x)[:, equal].sum(0)
            jac = torch.stack([torch.autograd.grad(ri, x, retain_graph=True)[0] for ri in r], 1)  # (B, k, dof)
        jjt = jac @ jac.transpose(1, 2) + 1e-8 * torch.eye(len(equal)).to(q)
        return v - (jac.transpose(1, 2) @ torch.linalg.solve(jjt, jac @ v[..., None]))[..., 0]

    def _poses(self, d: Tensor) -> Tensor:
        t = torch.eye(4).to(d).repeat(len(d), 1, 1)
        t[:, :3, 3] = d[:, :3]
        t[:, :3, :3] = euler_angles_to_matrix(d[:, 3:], "XYZ")
        return self.T_o_w.to(d) @ t @ self.T_w_e.to(d)


def box_excess(x: Tensor, bounds: Tensor) -> Tensor:
    """How far each coordinate of x lies outside (lower, upper), signed, 0 inside."""
    return (x - bounds[:, 0]).clamp(max=0) + (x - bounds[:, 1]).clamp(min=0)


def _displacement(T_o_s: Tensor, T_o_w: Tensor, T_w_e: Tensor) -> Tensor:
    T_w_s = torch.inverse(T_o_w) @ T_o_s @ torch.inverse(T_w_e)
    angles = matrix_to_euler_angles(T_w_s[:, :3, :3], "XYZ")
    return torch.cat([T_w_s[:, :3, 3], torch.remainder(angles + math.pi, 2 * math.pi) - math.pi], dim=1)


def _pose_close(a: Tensor, b: Tensor, tol: float = 1e-3) -> Tensor:
    """IK branches outside the reach come back with clamped angles: keep those that reach the pose."""
    return (a.to(b) - b).flatten(1).abs().amax(dim=1) < tol


def ur5e_cup_env(
    table: bool = False, epsilon: float = 0.7, tolerance: float = 0.1, clearance: float = 0.05
) -> ConstrainedEnv:
    """A UR5e with a 2F-85 carrying a cup upright (roll = pitch = 0) in front of it.

    The grasp is the reference's: the tool 0.15 m behind the cup, pointing horizontally along the world x axis of the
    cup. Position and yaw are confined to where the arm has one connected manifold: see the method README. `table` adds
    ntfields-old's `hold_cup` table and boxes, lowered so that the table top is level with the arm's base.
    """
    T_w_e = torch.eye(4)
    T_w_e[:3, 3] = torch.tensor([-0.15, 0.0, 0.0])
    T_w_e[:3, :3] = euler_angles_to_matrix(torch.tensor([0.0, math.pi / 2, 0.0]), "XYZ")
    bounds = torch.tensor([[0.3, 0.7], [-0.4, 0.4], [0.15, 0.65], [0.0, 0.0], [0.0, 0.0], [-math.pi / 3, math.pi / 3]])
    offsets = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.0])
    solids = SolidScene.from_json(json.loads(CUP_SCENE.read_text())) if table else None
    arm = ArmEnv.ur5e_2f85(solids)
    joints = torch.tensor([[-0.5, 0.5]] * 5 + [[-0.5, 0.0]])  # θ6 < 0: the sheet with the most poses
    constraint = TSRConstraint(arm, bounds, offsets, torch.eye(4), T_w_e, branches=(0, 2), joints=joints)
    return ConstrainedEnv(arm, constraint, epsilon, tolerance, clearance)


def ur5e_cup_problem(table: bool = False, epsilon: float = 0.7, lam: float = 0.1, eased: bool = True) -> Problem:
    """The cup task with the speed around its manifold, sharing ε."""
    return Problem(ur5e_cup_env(table, epsilon), GaussianBand(epsilon, lam, eased))
