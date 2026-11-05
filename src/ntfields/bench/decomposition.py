from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ntfields.envs import ArmEnv
from ntfields.envs.solids import SolidScene


@dataclass
class BoxFit:
    """How well boxes stand in for a scene's objects, leaving the mount out of the volumes."""

    n_boxes: int
    iou: float  # intersection over union of the boxes' volume and the objects'
    coverage: float  # share of the objects' volume inside the boxes; what is missed can be driven through
    missed_blocked: float  # share of configurations in collision that the boxes call free
    free_blocked: float  # share of free configurations that the boxes call blocked


def _inside(scene: SolidScene, p: Tensor) -> Tensor:
    return scene(p) < 0


def evaluate_boxes(
    objects: list[dict[str, Any]],
    boxes: list[dict[str, Any]],
    margin: float,
    generator: torch.Generator,
    n_points: int = 2_000_000,
    n_configs: int = 20_000,
    device: str = "cpu",
) -> BoxFit:
    """Compare boxes (dicts with center, size, quat) with the true objects of a scene, by sampling."""
    box_rows = [{**b, "type": "box"} for b in boxes]
    solid = [o for o in objects if o.get("role") != "mount"]
    mount = [o for o in objects if o.get("role") == "mount"]
    true_scene, box_scene = SolidScene.from_json(solid), SolidScene.from_json(box_rows)
    corners = torch.cat([true_scene.triangles().reshape(-1, 3), box_scene.triangles().reshape(-1, 3)])
    lo, hi = corners.amin(0), corners.amax(0)
    p = (lo + (hi - lo) * torch.rand(n_points, 3, generator=generator)).to(device)
    truth, fit = _inside(true_scene, p), _inside(box_scene, p)
    covered = int((truth & fit).sum())
    union = int((truth | fit).sum())

    true_env, box_env = (
        ArmEnv.ur5e_2f85(SolidScene.from_json(objects)),
        ArmEnv.ur5e_2f85(SolidScene.from_json(box_rows + mount)),
    )
    q = (torch.rand(n_configs, 6, generator=generator) - 0.5).to(device)
    free_true, free_box = true_env.distance(q) > margin, box_env.distance(q) > margin
    return BoxFit(
        n_boxes=len(boxes),
        iou=covered / max(union, 1),
        coverage=covered / max(int(truth.sum()), 1),
        missed_blocked=float((free_box & ~free_true).sum()) / max(int((~free_true).sum()), 1),
        free_blocked=float((~free_box & free_true).sum()) / max(int(free_true.sum()), 1),
    )
