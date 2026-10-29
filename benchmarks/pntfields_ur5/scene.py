"""P-NTFields' UR5 cabinet scene, built from a P-NTFields checkout (its assets are under a non-commercial license, so
ntfields does not ship them)."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ntfields.envs import ArmEnv
from ntfields.envs.mesh_distance import MeshDistance

PNTFIELDS = Path(os.environ.get("NTFIELDS_PNTFIELDS", Path(__file__).resolve().parents[3] / "P-NTFields"))
UR5 = PNTFIELDS / "datasets" / "arm" / "UR5"
CHECKPOINT = PNTFIELDS / "Experiments" / "UR5" / "Model_Epoch_10000_ValLoss_3.526511e-03.pt"
# speed_sampling_gpu.py::arm_obstacle_distance skips the first two FK frames (world, base_link)
LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
MARGIN = 0.5 / 12
OFFSET = MARGIN / 10
# test/arm_plan.py: its query, in θ around the base pose
BASE = [0.0, -0.5 * math.pi, 0.0, -0.5 * math.pi, 0.0, 0.0]
QUERY_START = [-2.2, 0.4, 1.1, 0.5, -0.5, 0.9]
QUERY_GOAL = [-1.3, 0.4, 1.1, 0.5, -0.5, 0.0]


def _vertices(path: Path) -> Tensor:
    return torch.tensor([[float(x) for x in line.split()[1:4]] for line in path.open() if line.startswith("v ")])


def cabinet_triangles() -> Tensor:
    lines = (UR5 / "untitled_scaled.off").read_text().split("\n")
    nv, nf, _ = map(int, lines[1].split())
    v = torch.tensor([[float(x) for x in line.split()] for line in lines[2 : 2 + nv]])
    f = torch.tensor([[int(x) for x in line.split()[1:]] for line in lines[2 + nv : 2 + nv + nf]])
    return v[f]


def ur5_cabinet(cache: Path, device: str = "cuda") -> ArmEnv:
    """The UR5 among the cabinet: clearance is the unsigned distance from the links' collision-mesh vertices to the
    cabinet mesh, q = θ / 2π in [-0.5, 0.5]^6, as in the reference's data generation."""
    if cache.is_file():
        mesh = MeshDistance(**torch.load(cache, weights_only=True))
    else:
        mesh = MeshDistance.build(cabinet_triangles(), spacing=0.005, cap=0.1, exact_below=0.01, device=device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vars(mesh), cache)
    link_points = {k: torch.cat([p := _vertices(UR5 / "meshes" / "collision" / f"{k}.obj"), torch.zeros(len(p), 1)], 1)
                   for k in LINKS}  # fmt: skip
    return ArmEnv(
        urdf=(UR5 / "UR5.urdf").read_text(),
        end_link="wrist_3_link",
        link_spheres=link_points,
        bounds=torch.tensor([[-0.5, 0.5]] * 6),
        points=torch.zeros(0, 3),
        sphere_centers=torch.zeros(0, 3),
        sphere_radii=torch.zeros(0),
        chunk=4096,
        mesh=mesh.to(device),
    )


def query() -> tuple[Tensor, Tensor]:
    base = torch.tensor(BASE)
    return (torch.tensor(QUERY_START) + base) / (2 * math.pi), (torch.tensor(QUERY_GOAL) + base) / (2 * math.pi)


# speed_sampling_gpu.py::arm_obstacle_obb: hard-coded obstacle box, as (max x, y, z, min x, y, z)
OBB_OBSTACLE = (0.7, -0.5, 1.0, -0.6, -0.7, -0.7)
OBB_LINKS = ("upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")


def _corners(bbox: Tensor) -> Tensor:
    """bbox_corner: (4, 8) homogeneous corners of a (max xyz, min xyz) box."""
    pts = [[bbox[i * 3], bbox[1 + j * 3], bbox[2 + k * 3], 1.0] for i in range(2) for j in range(2) for k in range(2)]
    return torch.tensor(pts, dtype=bbox.dtype).T


def _cross_axes(row: Tensor) -> Tensor:
    """cross_matrix: three unit vectors from one rotation row (a, b, c), as the reference builds them."""
    a, b, c = row.unbind(-1)
    l0 = (c**2 + b**2).sqrt().clamp_min(1e-6)
    l1 = (c**2 + a**2).sqrt().clamp_min(1e-6)
    l2 = (b**2 + a**2).sqrt().clamp_min(1e-6)
    z = torch.zeros_like(a)
    return torch.stack([
        torch.stack([z, c / l0, -b / l0], -1),
        torch.stack([-c / l1, z, a / l1], -1),
        torch.stack([b / l2, -a / l2, z], -1),
    ], dim=1)  # fmt: skip


def wrist_obb_filter(env: ArmEnv, q: Tensor, margin: float = MARGIN) -> Tensor:
    """arm_obstacle_obb as it behaves: True where wrist_3's box, grown by `margin`, is not separated from
    OBB_OBSTACLE along the reference's 15 test axes. (The reference returns only its last link's test.)"""
    bbox = torch.from_numpy(np.load(UR5 / "meshes" / "collision" / "wrist_3bbox.npy")).to(q)
    bbox = torch.cat([bbox[:3] + margin, bbox[3:] - margin])
    m = env._chain(q).forward_kinematics(q * env.joint_scale, end_only=False)["wrist_3_link"].get_matrix()
    rot = m[:, :3, :3]
    axis = torch.cat([torch.eye(3).to(q).expand(len(q), 3, 3), rot, *(_cross_axes(rot[:, i]) for i in range(3))], 1)
    rob = axis @ (m @ _corners(bbox).to(q))[:, :3]  # (B, 15, 8)
    obs = axis @ _corners(torch.tensor(OBB_OBSTACLE)).to(q)[:3]
    separated = (rob.amax(-1) < obs.amin(-1)) | (obs.amax(-1) < rob.amin(-1))
    return ~separated.any(dim=-1)


@dataclass
class ReferenceArmPairs:
    """speed_sampling_gpu.py::arm_append_list: starts uniform in [-0.5, 0.5]^6 that pass `wrist_obb_filter` and have
    offset ≤ d ≤ margin; goals = start + a random unit direction · U(0, √3), kept if inside the bounds; goal
    clearance unchecked. Draws `round_size` starts at a time until n pairs are found."""

    margin: float = MARGIN
    offset: float = OFFSET
    max_step: float = math.sqrt(3)
    round_size: int = 1_000_000

    def sample(self, problem, n: int, generator: torch.Generator, device: str = "cpu") -> tuple[Tensor, Tensor]:  # noqa: ANN001
        env = problem.env
        found: list[Tensor] = []
        count = 0
        while count < n:
            p = torch.rand(self.round_size, 6, generator=generator) - 0.5
            dp = torch.rand(self.round_size, 6, generator=generator) - 0.5
            r = torch.rand(self.round_size, 1, generator=generator) * self.max_step
            goal = p + torch.nn.functional.normalize(dp, dim=1) * r
            keep = ((goal <= 0.5) & (goal >= -0.5)).all(dim=1)
            p, goal = p[keep], goal[keep]
            keep = torch.cat([wrist_obb_filter(env, c.to(device)).cpu() for c in p.split(65536)])
            p, goal = p[keep], goal[keep]
            d = torch.cat([env.distance(c.to(device)).cpu() for c in p.split(65536)])
            keep = (d >= self.offset) & (d <= self.margin)
            found.append(torch.cat([p[keep], goal[keep]], dim=1))
            count += int(keep.sum())
        pairs = torch.cat(found)[:n]
        return pairs[:, :6], pairs[:, 6:]
