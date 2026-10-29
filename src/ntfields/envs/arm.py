import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from ntfields.base import SceneCondition

UR5E = Path(__file__).parents[1] / "resources" / "ur5e"
UR5E_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")


def ntrl_scenarios() -> tuple[Tensor, Tensor]:
    """ntrl-demo's 5 UR5 planning scenarios (`tests/arm_plan_stat.py`): starts and goals (5, 6), q = θ / 2π."""
    base = [0.0, -0.5 * math.pi, 0.0, -0.5 * math.pi, 0.0, 0.0]
    half = 0.5 * math.pi
    scenarios = torch.tensor([
        [0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.5, -1.2, half, half, 0.0],
        [-0.2, -0.5, -0.35, 0.2 * math.pi, half, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, -0.5, -0.35, 0.3 * math.pi, half, 0.0, 0.2, -0.7, -0.9, 0.2 * math.pi, 0.7 * math.pi, 0.0],
        [0.2, -0.5, -1.2, half, half, 0.0, -0.2, -0.5, -0.35, 0.2 * math.pi, half, 0.0],
        [0.2, -0.7, -1.0, half, half, 0.0, -0.2, -0.5, -0.35, 0.2 * math.pi, half, 0.0],
    ])  # fmt: skip
    q = (scenarios + torch.tensor(base * 2)) / (2 * math.pi)
    return q[:, :6], q[:, 6:]


def ntrl_random_pairs() -> tuple[Tensor, Tensor]:
    """50 random evaluation pairs (collision-free IK configurations in the cabinet box, pair seed 0):
    starts and goals (50, 6), q = θ / 2π."""
    pairs = torch.from_numpy(np.load(UR5E / "eval_random_pairs.npy"))
    return pairs[:, 0], pairs[:, 1]


@dataclass
class ArmEnv:
    """Serial arm with collision spheres on its links, among point-cloud and sphere obstacles.

    Configurations are q = θ / joint_scale. The clearance is the smallest gap between an arm sphere and an
    obstacle, minus `clearance_offset`.
    """

    urdf: str  # URDF text; only the kinematic chain is used
    end_link: str
    link_spheres: dict[str, Tensor]  # link -> (k, 4) rows (x, y, z, r) in the link frame
    bounds: Tensor  # (dof, 2)
    points: Tensor  # (P, 3) obstacle point cloud (may be empty)
    sphere_centers: Tensor  # (K, 3) obstacle spheres (may be empty)
    sphere_radii: Tensor  # (K,)
    joint_scale: float = 2 * math.pi
    clearance_offset: float = 0.0
    chunk: int = 512  # configurations per distance chunk (memory: chunk x spheres x points)
    _chains: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def ur5e(cls, points: Tensor | None = None, spheres: Tensor | None = None, clearance_offset: float = 0.0) -> Self:
        """UR5e with ntrl-demo's collision spheres; q = θ / 2π within the URDF limits (±1, elbow ±0.5).
        `spheres` rows are (x, y, z, r)."""
        import pytorch_kinematics as pk  # the [arm] extra

        urdf = (UR5E / "ur5e.urdf").read_text()
        lower, upper = pk.build_serial_chain_from_urdf(urdf, "wrist_3_link").get_joint_limits()
        spheres = torch.zeros(0, 4) if spheres is None else spheres
        return cls(
            urdf=urdf,
            end_link="wrist_3_link",
            link_spheres={k: torch.from_numpy(np.load(UR5E / "spheres" / f"{k}.npy")).float() for k in UR5E_LINKS},
            bounds=torch.tensor([lower, upper], dtype=torch.float32).T / (2 * math.pi),
            points=torch.zeros(0, 3) if points is None else points,
            sphere_centers=spheres[:, :3],
            sphere_radii=spheres[:, 3],
            clearance_offset=clearance_offset,
        )

    @classmethod
    def ur5e_ntrl_scene(cls) -> Self:
        """ntrl-demo's UR5 setup: its scanned lab scene and a 0.01 m clearance offset."""
        points = torch.from_numpy(np.load(UR5E / "ntrl_scene_points.npy"))
        return cls.ur5e(points=points, clearance_offset=0.01)

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    def arm_spheres(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """World-frame arm sphere centres (B, S, 3) and radii (S,)."""
        frames = self._chain(q).forward_kinematics(q * self.joint_scale, end_only=False)
        centers, radii = [], []
        for link, spheres in self.link_spheres.items():
            m = frames[link].get_matrix()  # (B, 4, 4)
            s = spheres.to(q)
            centers.append(s[:, :3] @ m[:, :3, :3].transpose(1, 2) + m[:, None, :3, 3])
            radii.append(s[:, 3])
        return torch.cat(centers, dim=1), torch.cat(radii)

    def distance(self, q: Tensor) -> Tensor:
        return torch.cat([self._distance(c) for c in q.split(self.chunk)]) - self.clearance_offset

    def _distance(self, q: Tensor) -> Tensor:
        centers, radii = self.arm_spheres(q)
        gaps = [torch.full(centers.shape[:2], torch.inf, dtype=q.dtype, device=q.device)]
        if len(self.points):
            points = self.points.to(q)
            with torch.no_grad():  # the min's gradient only flows through the nearest point
                nearest = torch.cdist(centers, points[None].expand(len(q), -1, -1)).argmin(dim=-1)
            gaps.append((centers - points[nearest]).norm(dim=-1))
        if len(self.sphere_centers):
            d = (centers[:, :, None] - self.sphere_centers.to(q)).norm(dim=-1) - self.sphere_radii.to(q)
            gaps.append(d.amin(dim=-1))
        return (torch.stack(gaps).amin(dim=0) - radii).amin(dim=-1)

    def in_bounds(self, q: Tensor) -> Tensor:
        b = self.bounds.to(q)
        return ((q >= b[:, 0]) & (q <= b[:, 1])).all(dim=-1)

    def is_valid(self, q: Tensor) -> Tensor:
        return self.in_bounds(q) & (self.distance(q) > 0)

    def condition(self) -> SceneCondition:
        return SceneCondition(points=self.points)

    def _chain(self, q: Tensor) -> Any:
        key = f"{q.device}/{q.dtype}"
        if key not in self._chains:
            import pytorch_kinematics as pk  # the [arm] extra

            chain = pk.build_serial_chain_from_urdf(self.urdf, self.end_link)
            self._chains[key] = chain.to(dtype=q.dtype, device=q.device)
        return self._chains[key]
