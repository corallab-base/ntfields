"""Read MotionBenchMaker's (MBM) scene, query and camera files into the arm frame of a UR5e, without ROS.

MBM's frames: YAML scenes are in `base_link`, on top of a 0.9144 m pedestal, and URDF scenes and cameras are in `world`,
at the pedestal's foot. MBM's sampler moves the scene by `base_offset`. So in our frame, whose origin is the arm's base:

    p = p_yaml + base_offset + lift    p = p_urdf + base_offset - pedestal + lift    p = p_camera - pedestal + lift

`lift` keeps the shoulder at MBM's height relative to the scene: the UR5e's shoulder is above the UR5's.
"""

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

MBM_COMMIT = "f50542777f4d1ae013b0de430bafbf51fdbdc40e"
PEDESTAL = 0.9144
PEDESTAL_WIDTH = 0.4
LIFT = 0.1625 - 0.089159  # shoulder height of the UR5e less that of the UR5
FLOOR_SIZE = 4.0
FLOOR_THICKNESS = 0.1


@dataclass
class Target:
    """A grasp pose from MBM's queries, in our frame. `quat` (x, y, z, w) is MBM's end-effector orientation."""

    position: list[float]
    quat: list[float]


def _read(root: Path, url: str) -> dict[str, Any]:
    return yaml.safe_load(_path(root, url).read_text())


def _path(root: Path, url: str) -> Path:
    return root / url.removeprefix("package://motion_bench_maker/")


def problem_config(root: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "problems" / f"{name}_ur5.yaml").read_text())


def _shift(config: dict[str, Any], scene_is_urdf: bool) -> list[float]:
    x, y, z = config["base_offset"]["position"]
    return [x, y, z + LIFT - (PEDESTAL if scene_is_urdf else 0.0)]


def _stl(path: Path) -> Tensor:
    """(T, 3, 3) triangles of a binary STL."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    rows = [struct.unpack("<9f", data[84 + 50 * i + 12 : 84 + 50 * i + 48]) for i in range(count)]
    return torch.tensor(rows, dtype=torch.float64).view(-1, 3, 3)


def _rotation(axis: list[float], angle: float) -> Tensor:
    x, y, z = axis
    k = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def yaml_objects(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """The collision objects of a YAML scene in our frame, at their nominal poses."""
    shift = _shift(config, False)
    out = []
    for item in _read(root, config["scene"])["world"]["collision_objects"]:
        prim, pose = item["primitives"][0], item["primitive_poses"][0]
        center = [float(a) + b for a, b in zip(pose["position"], shift, strict=True)]
        base = {"id": item["id"].strip(), "center": center, "quat": [float(x) for x in pose["orientation"]]}
        if prim["type"] == "box":
            out.append({**base, "type": "box", "size": [float(x) for x in prim["dimensions"]]})
        else:
            assert prim["type"] == "cylinder"
            height, radius = prim["dimensions"]  # as in MoveIt, about local z
            out.append({**base, "type": "cylinder", "height": float(height), "radius": float(radius)})
    return out


def urdf_objects(root: Path, config: dict[str, Any], angles: dict[str, float]) -> list[dict[str, Any]]:
    """The links of a URDF scene as closed meshes in our frame, with revolute joints at `angles` (default 0)."""
    urdf = ET.parse(_path(root, config["scene"])).getroot()
    shift = torch.tensor(_shift(config, True), dtype=torch.float64)
    joints = {j.findtext("child") or j.find("child").get("link"): j for j in urdf.findall("joint")}  # pyright: ignore[reportOptionalMemberAccess]
    out = []
    for link in urdf.findall("link"):
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            continue
        name = link.get("name")
        assert name is not None
        triangles = _stl(_path(root, mesh.get("filename", "")))
        joint = joints[name]
        origin = joint.find("origin")
        assert origin is not None
        xyz = torch.tensor([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=torch.float64)
        assert origin.get("rpy", "0 0 0").split() == ["0"] * 3, "scene joints carry no rotation"
        if joint.get("type") == "revolute":
            element = joint.find("axis")
            assert element is not None
            axis = [float(v) for v in element.get("xyz", "").split()]
            triangles = triangles @ _rotation(axis, angles.get(joint.get("name", ""), 0.0)).T
        out.append({"id": name, "type": "mesh", "triangles": (triangles + xyz + shift).tolist()})
    return out


def mount() -> list[dict[str, Any]]:
    """MBM's pedestal and a floor, which MBM's robot model holds and its scenes do not."""
    height = PEDESTAL - LIFT
    identity = [0.0, 0.0, 0.0, 1.0]
    return [
        {"id": "pedestal", "type": "box", "role": "mount", "center": [0.0, 0.0, -height / 2], "quat": identity,
         "size": [PEDESTAL_WIDTH, PEDESTAL_WIDTH, height]},
        {"id": "floor", "type": "box", "role": "mount", "center": [0.0, 0.0, -height - FLOOR_THICKNESS / 2],
         "quat": identity, "size": [FLOOR_SIZE, FLOOR_SIZE, FLOOR_THICKNESS]},
    ]  # fmt: skip


def _anchor(config: dict[str, Any], objects: list[dict[str, Any]], name: str) -> list[float]:
    """An object's position in our frame. MBM's query offsets are in this frame's axes: object poses are unrotated."""
    for o in objects:
        if o["id"] == name:
            return o["center"] if o["type"] != "mesh" else _shift(config, True)
    raise KeyError(name)


def targets(config: dict[str, Any], objects: list[dict[str, Any]], root: Path, url: str, key: str) -> list[Target]:
    """The grasp poses under `key` (start_queries or goal_queries) of a query file."""
    out = []
    for query in _read(root, url).get(key, []):
        for name in query["objects"]:
            offset = query["offset"]
            center = _anchor(config, objects, name)
            position = [a + float(b) for a, b in zip(center, offset["position"], strict=True)]
            out.append(Target(position, [float(x) for x in offset["orientation"]]))
    return out


def cameras(root: Path, config: dict[str, Any], objects: list[dict[str, Any]]) -> tuple[Tensor, dict[str, float]]:
    """(V, 2, 3) camera (position, look-at point) pairs in our frame, one per look object and camera point, and MBM's
    intrinsics."""
    sensors = _read(root, config["sensors"])
    drop = LIFT - PEDESTAL
    views = []
    for look in sensors["look_objects"]:
        center = _anchor(config, objects, look["name"].strip())
        eye = [a + float(b) for a, b in zip(center, look["position"], strict=True)]
        for x, y, z in sensors["cam_points"]:
            views.append([[float(x), float(y), float(z) + drop], eye])
    return torch.tensor(views, dtype=torch.float64), _read(root, sensors["camera_config"])


def view_rays(view: Tensor, camera: dict[str, float], stride: int = 2) -> tuple[Tensor, Tensor]:
    """Ray origin (3,) and unit directions (R, 3) of one pinhole view, every `stride`-th pixel, up along +z as in MBM's
    `lookat`."""
    origin, eye = view
    forward = (eye - origin) / (eye - origin).norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0]).to(view))
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    width, height = int(camera["width"]), int(camera["height"])
    u, v = torch.meshgrid(
        torch.arange(0, width, stride).to(view), torch.arange(0, height, stride).to(view), indexing="ij"
    )
    x, y = (u - width / 2) / camera["focal_length_x"], (v - height / 2) / camera["focal_length_y"]
    rays = forward + x.reshape(-1, 1) * right - y.reshape(-1, 1) * up
    return origin, rays / rays.norm(dim=-1, keepdim=True)


def first_hit(origin: Tensor, rays: Tensor, triangles: Tensor, chunk: int = 16384) -> Tensor:
    """Hit points of rays on the nearest triangle (Möller-Trumbore), for rays that hit something: (H, 3)."""
    a, b, c = triangles.to(rays).unbind(1)
    e1, e2 = b - a, c - a
    hits = []
    for r in rays.split(chunk):
        h = torch.linalg.cross(r[:, None], e2[None].expand(len(r), -1, -1))
        det = (e1 * h).sum(-1)
        f = 1 / det.where(det.abs() > 1e-12, torch.full_like(det, 1e-12))
        s = origin - a
        u = f * (s * h).sum(-1)
        q = torch.linalg.cross(s[None].expand_as(h), e1[None].expand_as(h))
        v = f * (r[:, None] * q).sum(-1)
        t = f * (q * e2).sum(-1)
        ok = (det.abs() > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0.01)
        nearest = t.where(ok, torch.full_like(t, torch.inf)).amin(dim=1)
        seen = nearest.isfinite()
        hits.append(origin + r[seen] * nearest[seen, None])
    return torch.cat(hits)


def surface_points(triangles: Tensor, n: int, generator: torch.Generator) -> Tensor:
    """n points uniform by area on the triangles: (n, 3)."""
    a, b, c = triangles.unbind(1)
    area = torch.linalg.cross(b - a, c - a).norm(dim=-1)
    pick = torch.multinomial(area, n, replacement=True, generator=generator)
    u, v = torch.rand(n, 1, generator=generator), torch.rand(n, 1, generator=generator)
    flip = (u + v) > 1
    u, v = u.where(~flip, 1 - u), v.where(~flip, 1 - v)
    return a[pick] + u * (b - a)[pick] + v * (c - a)[pick]


def voxel_downsample(points: Tensor, size: float) -> Tensor:
    """One point per occupied voxel of edge `size`: the first found, so the points stay on the surfaces."""
    keys = (points / size).floor().long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), len(points), device=points.device)
    first = first.scatter_reduce(0, inverse, torch.arange(len(points), device=points.device), "amin")
    return points[first]
