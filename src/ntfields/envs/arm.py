import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from ntfields.base import SceneCondition
from ntfields.envs.mesh_distance import MeshDistance
from ntfields.envs.solids import SolidScene

UR5E = Path(__file__).parents[1] / "assets" / "ur5e"
UR5E_2F85 = Path(__file__).parents[1] / "assets" / "ur5e_2f85"
UR5E_SPHERIZED = Path(__file__).parents[1] / "assets" / "ur5e_spherized"
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
    """Serial arm with collision spheres on its links, among point-cloud, sphere and triangle-mesh obstacles.

    Configurations are q = θ / joint_scale (one scale, or one per joint); joints past `dim` are held at `locked`
    (default 0). The clearance is the smallest gap between an arm sphere and an
    obstacle, minus `clearance_offset`.
    """

    urdf: str  # URDF text; only the kinematic chain is used
    end_link: str
    link_spheres: dict[str, Tensor]  # link -> (k, 4) rows (x, y, z, r) in the link frame
    bounds: Tensor  # (dof, 2)
    points: Tensor  # (P, 3) obstacle point cloud (may be empty)
    sphere_centers: Tensor  # (K, 3) obstacle spheres (may be empty)
    sphere_radii: Tensor  # (K,)
    joint_scale: float | tuple[float, ...] = 2 * math.pi
    locked: tuple[float, ...] = ()  # angles of the joints after the first `dim` (default: 0)
    clearance_offset: float = 0.0
    chunk: int = 512  # configurations per distance chunk (memory: chunk x spheres x points)
    mesh: MeshDistance | None = None  # unsigned distance to a triangle mesh
    solids: SolidScene | None = None  # signed distance to solid primitives and closed meshes
    self_ignore: tuple[tuple[str, str], ...] = ()  # link pairs left out of `self_distance`
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
    def ur5e_spherized(cls, dof: int = 6, spheres: Tensor | None = None) -> Self:
        """UR5e with the collision spheres of `assets/ur5e_spherized/ur5e.urdf` (RaNTFields' arm), the first `dof`
        joints movable and the rest at 0; q = θ / θ_max per joint, so q in [-1, 1]^dof. `spheres` rows are
        (x, y, z, r)."""
        import xml.etree.ElementTree as ET

        import pytorch_kinematics as pk  # the [arm] extra

        urdf = (UR5E_SPHERIZED / "ur5e.urdf").read_text()
        link_spheres: dict[str, Tensor] = {}
        for link in ET.fromstring(urdf).iter("link"):
            rows = []
            for c in link.iter("collision"):
                sphere, origin = c.find("geometry/sphere"), c.find("origin")
                if sphere is not None and origin is not None:
                    rows.append([*map(float, origin.get("xyz", "0 0 0").split()), float(sphere.get("radius", "0"))])
            if rows:
                link_spheres[link.get("name", "")] = torch.tensor(rows, dtype=torch.float32)
        _, upper = pk.build_serial_chain_from_urdf(urdf, "wrist_3_link").get_joint_limits()
        spheres = torch.zeros(0, 4) if spheres is None else spheres
        return cls(
            urdf=urdf,
            end_link="wrist_3_link",
            link_spheres=link_spheres,
            bounds=torch.tensor([[-1.0, 1.0]] * dof),
            points=torch.zeros(0, 3),
            sphere_centers=spheres[:, :3],
            sphere_radii=spheres[:, 3],
            joint_scale=tuple(float(u) for u in upper[:dof]),
        )

    @classmethod
    def ur5e_ntrl_scene(cls) -> Self:
        """ntrl-demo's UR5 setup: its scanned lab scene and a 0.01 m clearance offset."""
        points = torch.from_numpy(np.load(UR5E / "ntrl_scene_points.npy"))
        return cls.ur5e(points=points, clearance_offset=0.01)

    @classmethod
    def ur5e_2f85(cls, solids: SolidScene | None = None) -> Self:
        """UR5e with a rigid, open Robotiq 2F-85 on wrist_3_link, q = θ / 2π in [-0.5, 0.5]^6 (all joints ±π)."""
        pairs = json.loads((UR5E_2F85 / "self_ignore.json").read_text())
        spheres = {k: torch.from_numpy(np.load(UR5E_2F85 / "spheres" / f"{k}.npy")).float() for k in UR5E_LINKS}
        return cls(
            urdf=(UR5E / "ur5e.urdf").read_text(),
            end_link="wrist_3_link",
            link_spheres=spheres,
            bounds=torch.tensor([[-0.5, 0.5]] * 6),
            points=torch.zeros(0, 3),
            sphere_centers=torch.zeros(0, 3),
            sphere_radii=torch.zeros(0),
            solids=solids,
            self_ignore=tuple((a, b) for a, b in pairs),
            chunk=4096,
        )

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    def arm_spheres(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """World-frame arm sphere centres (B, S, 3) and radii (S,)."""
        frames = self._chain(q).forward_kinematics(self._angles(q), end_only=False)
        centers, radii = [], []
        for link, spheres in self.link_spheres.items():
            m = frames[link].get_matrix()  # (B, 4, 4)
            s = spheres.to(q)
            centers.append(s[:, :3] @ m[:, :3, :3].transpose(1, 2) + m[:, None, :3, 3])
            radii.append(s[:, 3])
        return torch.cat(centers, dim=1), torch.cat(radii)

    def end_pose(self, q: Tensor) -> Tensor:
        """(B, 4, 4) poses of `end_link` in the base frame."""
        return self._chain(q).forward_kinematics(self._angles(q)).get_matrix()

    def scale(self, like: Tensor) -> Tensor:
        """`joint_scale` as a tensor (scalar or (dim,)) with `like`'s dtype and device: θ = q · scale."""
        return torch.as_tensor(self.joint_scale, dtype=like.dtype, device=like.device)

    def _angles(self, q: Tensor) -> Tensor:
        """Joint angles of the whole chain (B, n) from q (B, dim)."""
        theta = q * self.scale(q)
        n = len(self._chain(q).get_joint_parameter_names())
        if n > theta.shape[-1]:
            tail = torch.zeros(n - theta.shape[-1], dtype=q.dtype, device=q.device)
            if self.locked:
                tail = torch.tensor(self.locked, dtype=q.dtype, device=q.device)
            theta = torch.cat([theta, tail.expand(len(q), -1)], dim=-1)
        return theta

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
        if self.mesh is not None:
            gaps.append(self.mesh(centers.reshape(-1, 3)).view(centers.shape[:2]))
        if self.solids is not None:
            gaps.append(self.solids(centers.reshape(-1, 3)).view(centers.shape[:2]))
        return (torch.stack(gaps).amin(dim=0) - radii).amin(dim=-1)

    def self_distance(self, q: Tensor) -> Tensor:
        """The smallest gap (B,) between spheres on different links, leaving out `self_ignore`."""
        ignored = {frozenset(p) for p in self.self_ignore}
        owner = [k for k, s in self.link_spheres.items() for _ in range(len(s))]
        i, j = torch.triu_indices(len(owner), len(owner), offset=1).tolist()
        keep = [(a, b) for a, b in zip(i, j, strict=True)
                if owner[a] != owner[b] and frozenset((owner[a], owner[b])) not in ignored]  # fmt: skip
        first, second = torch.tensor(keep, device=q.device).T
        out = []
        for c in q.split(self.chunk):
            centers, radii = self.arm_spheres(c)
            gap = (centers[:, first] - centers[:, second]).norm(dim=-1) - radii[first] - radii[second]
            out.append(gap.amin(dim=-1))
        return torch.cat(out)

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
