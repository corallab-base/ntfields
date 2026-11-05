import math
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import Tensor

from ntfields.envs.mesh_distance import point_triangle_distance


def quat_matrix(quat: Tensor) -> Tensor:
    """(..., 3, 3) rotations from (..., 4) unit quaternions (x, y, z, w)."""
    x, y, z, w = (quat / quat.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*quat.shape[:-1], 3, 3)  # fmt: skip


def winding_number(p: Tensor, triangles: Tensor, chunk: int = 2048) -> Tensor:
    """Generalised winding number (N,) of points p (N, 3): ±1 inside a closed mesh, 0 outside, also where the mesh
    has non-manifold edges or inverted normals."""
    out = []
    corners = triangles.to(p)
    for q in p.split(chunk):
        a, b, c = (corners[None] - q[:, None, None]).unbind(2)  # (n, T, 3) each
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.linalg.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        out.append(torch.atan2(det, den).sum(-1) / (2 * math.pi))
    return torch.cat(out)


def _box_triangles(center: Tensor, half: Tensor, rot: Tensor) -> Tensor:
    signs = torch.tensor([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]).to(center)
    v = (signs * half) @ rot.T + center
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
             [0, 6, 4], [1, 5, 7], [1, 7, 3]]  # fmt: skip
    return v[torch.tensor(faces)]


def _cylinder_triangles(center: Tensor, rot: Tensor, radius: Tensor, half: Tensor, sides: int = 32) -> Tensor:
    t = torch.arange(sides).to(center) * 2 * math.pi / sides
    ring = torch.stack([radius * t.cos(), radius * t.sin()], -1)
    top = torch.cat([ring, half.expand(sides, 1)], 1)
    low = torch.cat([ring, -half.expand(sides, 1)], 1)
    i, j = torch.arange(sides), (torch.arange(sides) + 1) % sides
    apex_top, apex_low = torch.zeros(sides, 3).to(center), torch.zeros(sides, 3).to(center)
    apex_top[:, 2], apex_low[:, 2] = half, -half
    tri = torch.cat([
        torch.stack([low[i], low[j], top[j]], 1), torch.stack([low[i], top[j], top[i]], 1),
        torch.stack([apex_top, top[i], top[j]], 1), torch.stack([apex_low, low[j], low[i]], 1),
    ])  # fmt: skip
    return tri @ rot.T + center


@dataclass
class SolidScene:
    """Solid boxes, cylinders and closed meshes with a signed distance: negative inside, so that a sphere deep in a
    thick solid does not read as clear, as it would with an unsigned distance."""

    box_center: Tensor  # (B, 3)
    box_half: Tensor  # (B, 3)
    box_rot: Tensor  # (B, 3, 3), local to world
    cyl_center: Tensor  # (C, 3)
    cyl_rot: Tensor  # (C, 3, 3); the axis is local z
    cyl_radius: Tensor  # (C,)
    cyl_half: Tensor  # (C,) half height
    meshes: list[Tensor]  # each (T, 3, 3), one closed object

    @classmethod
    def from_json(cls, objects: list[dict[str, Any]]) -> Self:
        boxes = [o for o in objects if o["type"] == "box"]
        cylinders = [o for o in objects if o["type"] == "cylinder"]

        def column(items: list[dict[str, Any]], key: str, width: int) -> Tensor:
            return torch.tensor([o[key] for o in items], dtype=torch.float32).reshape(-1, width)

        return cls(
            column(boxes, "center", 3),
            column(boxes, "size", 3) / 2,
            quat_matrix(column(boxes, "quat", 4)),
            column(cylinders, "center", 3),
            quat_matrix(column(cylinders, "quat", 4)),
            column(cylinders, "radius", 1).view(-1),
            column(cylinders, "height", 1).view(-1) / 2,
            [torch.tensor(o["triangles"], dtype=torch.float32) for o in objects if o["type"] == "mesh"],
        )

    def __call__(self, p: Tensor, chunk: int = 65536) -> Tensor:
        """Signed distances (N,) for points p (N, 3)."""
        return torch.cat([self._distance(c) for c in p.split(chunk)])

    def _distance(self, p: Tensor) -> Tensor:
        parts = [torch.full(p.shape[:1], torch.inf).to(p)]
        if len(self.box_center):
            local = (p[:, None] - self.box_center.to(p))[:, :, None] @ self.box_rot.to(p)  # (N, B, 1, 3)
            q = local[:, :, 0].abs() - self.box_half.to(p)
            parts.append((q.clamp_min(0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0)).amin(dim=1))
        if len(self.cyl_center):
            local = (p[:, None] - self.cyl_center.to(p))[:, :, None] @ self.cyl_rot.to(p)
            xy, z = local[:, :, 0, :2], local[:, :, 0, 2]
            q = torch.stack([xy.norm(dim=-1) - self.cyl_radius.to(p), z.abs() - self.cyl_half.to(p)], -1)
            parts.append((q.clamp_min(0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0)).amin(dim=1))
        for triangles in self.meshes:
            d = point_triangle_distance(p, triangles)
            parts.append(torch.where(winding_number(p, triangles).abs() > 0.5, -d, d))
        return torch.stack(parts).amin(dim=0)

    def triangles(self) -> Tensor:
        """The surface of every solid as triangles (T, 3, 3). Cylinders are 32-sided."""
        parts = [_box_triangles(c, h, r) for c, h, r in zip(self.box_center, self.box_half, self.box_rot, strict=True)]
        parts += [_cylinder_triangles(c, r, ra, h) for c, r, ra, h in
                  zip(self.cyl_center, self.cyl_rot, self.cyl_radius, self.cyl_half, strict=True)]  # fmt: skip
        return torch.cat([*parts, *self.meshes])
