import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ntfields.bench.suite import SuiteProblem
from ntfields.envs.solids import SolidScene

FLAT = 0.01  # half thickness of a 2D scene's obstacles when drawn


@dataclass
class Layer:
    """A way of drawing a scene: a triangle soup, a point cloud, or both empty."""

    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


def _sphere(center: list[float], radius: float, rings: int = 16, sides: int = 32) -> Tensor:
    theta = torch.linspace(0, math.pi, rings + 1)
    phi = torch.linspace(0, 2 * math.pi, sides + 1)
    t, p = torch.meshgrid(theta, phi, indexing="ij")
    v = torch.stack([t.sin() * p.cos(), t.sin() * p.sin(), t.cos()], -1) * radius + torch.tensor(center)
    a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]
    return torch.cat([torch.stack([a, c, b], 2).reshape(-1, 3, 3), torch.stack([a, d, c], 2).reshape(-1, 3, 3)])


def _point_scene_triangles(scene: dict[str, Any]) -> np.ndarray:
    """A sphere_box scene as triangles; a 2D one lies flat in z = 0."""
    flat = len(scene["bounds"]) == 2
    identity = [0.0, 0.0, 0.0, 1.0]
    objects, spheres = [], []
    for b in scene.get("boxes", []):
        center, size = (b["center"] + [0.0], b["size"] + [2 * FLAT]) if flat else (b["center"], b["size"])
        objects.append({"id": "b", "type": "box", "center": center, "size": size, "quat": identity})
    for s in scene.get("spheres", []):
        if flat:
            objects.append({"id": "s", "type": "cylinder", "center": s["center"] + [0.0], "quat": identity,
                            "height": 2 * FLAT, "radius": s["radius"]})  # fmt: skip
        else:
            spheres.append(_sphere(s["center"], s["radius"]))
    parts = [*spheres]
    if objects:
        parts.append(SolidScene.from_json(objects).triangles())
    return torch.cat(parts).numpy() if parts else Layer().triangles


def layers(problem: SuiteProblem, suite_dir: Path) -> dict[str, Layer]:
    """The ways this scene can be drawn, by name. The mount is drawn on its own: see `mount_layer`."""
    scene = problem.scene
    if scene["kind"] == "sphere_box":
        return {"true geometry": Layer(_point_scene_triangles(scene))}
    objects = [o for o in scene["objects"] if o.get("role") != "mount"]
    out = {"true geometry": Layer(SolidScene.from_json(objects).triangles().numpy())}
    oracle = suite_dir / "oracle" / f"{problem.name}.json"
    if oracle.exists():
        boxes = [{**b, "type": "box"} for b in json.loads(oracle.read_text())["boxes"]]
        out["oracle boxes"] = Layer(SolidScene.from_json(boxes).triangles().numpy())
    for kind, label in (("full", "full point cloud"), ("sensed", "sensed point cloud")):
        path = suite_dir / "points" / f"{problem.name}_{kind}.npy"
        if path.exists():
            out[label] = Layer(points=np.load(path))
    return out


def mount_layer(problem: SuiteProblem) -> Layer:
    mount = [o for o in problem.scene.get("objects", []) if o.get("role") == "mount"]
    return Layer(SolidScene.from_json(mount).triangles().numpy()) if mount else Layer()


def soup(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of a triangle soup, as viser takes them."""
    return triangles.reshape(-1, 3), np.arange(len(triangles) * 3, dtype=np.uint32).reshape(-1, 3)
