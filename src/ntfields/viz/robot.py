from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from viser import SceneApi, transforms

from ntfields.base import Environment
from ntfields.envs import ArmEnv

VISUAL = Path(__file__).parents[1] / "assets" / "ur5e_2f85" / "visual.npz"


def link_poses(env: ArmEnv, q: Tensor) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """link -> (wxyz, position) at one configuration."""
    frames = env._chain(q).forward_kinematics(q[None] * env.joint_scale, end_only=False)
    out = {}
    for link in env.link_spheres:
        m = frames[link].get_matrix()[0].detach().cpu().numpy().astype(np.float64)
        out[link] = (transforms.SO3.from_matrix(m[:3, :3]).wxyz, m[:3, 3])
    return out


def tool_positions(env: ArmEnv | Environment, q: Tensor) -> np.ndarray:
    """(N, 3): a configuration as a point in the scene: the tool for an arm, q itself for a point, at z = 0 in 2D."""
    if isinstance(env, ArmEnv):
        m = env._chain(q).forward_kinematics(q * env.joint_scale).get_matrix()
        return m[:, :3, 3].detach().cpu().numpy()
    points = q.detach().cpu().numpy()
    return points if points.shape[1] == 3 else np.c_[points, np.zeros(len(points))]


class ArmView:
    """An arm in one pose, as its collision spheres or its meshes."""

    def __init__(self, scene: SceneApi, name: str, env: ArmEnv, color: tuple[int, int, int], opacity: float) -> None:
        self.env = env
        self.spheres: list[tuple[str, Any, np.ndarray]] = []
        for link, rows in env.link_spheres.items():
            for i, row in enumerate(rows.numpy()):
                handle = scene.add_icosphere(
                    f"{name}/spheres/{link}/{i}", radius=float(row[3]), color=color, opacity=opacity, subdivisions=2
                )
                self.spheres.append((link, handle, row[:3]))
        self.meshes: dict[str, Any] = {}
        if VISUAL.exists():
            data = np.load(VISUAL)
            for link in env.link_spheres:
                self.meshes[link] = scene.add_mesh_simple(
                    f"{name}/mesh/{link}",
                    data[f"{link}/vertices"],
                    data[f"{link}/faces"],
                    color=color,
                    opacity=opacity,
                    side="double",
                )

    def set_style(self, meshes: bool, visible: bool = True) -> None:
        for _, handle, _ in self.spheres:
            handle.visible = visible and not meshes
        for handle in self.meshes.values():
            handle.visible = visible and meshes

    def set(self, q: Tensor) -> None:
        poses = link_poses(self.env, q.detach().cpu())
        for link, handle, local in self.spheres:
            wxyz, position = poses[link]
            handle.position = position + transforms.SO3(wxyz).apply(local)
        for link, handle in self.meshes.items():
            handle.wxyz, handle.position = poses[link]

    def remove(self) -> None:
        for _, handle, _ in self.spheres:
            handle.remove()
        for handle in self.meshes.values():
            handle.remove()


class PointView:
    """A point-mass configuration, as a marker."""

    def __init__(
        self, scene: SceneApi, name: str, env: Environment, color: tuple[int, int, int], opacity: float
    ) -> None:
        self.env = env
        radius = 0.02 * float((env.bounds[:, 1] - env.bounds[:, 0]).max())
        self.handle = scene.add_icosphere(f"{name}/marker", radius=radius, color=color, opacity=opacity)

    def set_style(self, meshes: bool, visible: bool = True) -> None:
        self.handle.visible = visible

    def set(self, q: Tensor) -> None:
        self.handle.position = tool_positions(self.env, q[None])[0]

    def remove(self) -> None:
        self.handle.remove()


def robot_view(scene: SceneApi, name: str, env: Environment, color: tuple[int, int, int], opacity: float) -> Any:
    kind = ArmView if isinstance(env, ArmEnv) else PointView
    return kind(scene, name, env, color, opacity)  # pyright: ignore[reportArgumentType]


def path_points(env: Environment, path: Tensor) -> np.ndarray:
    """A path as a polyline in the scene: (N, 3)."""
    with torch.no_grad():
        return tool_positions(env, path.float())
