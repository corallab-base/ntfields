"""Render the scenes of `ur5e-mbm-v1` to PNGs, to check their placement by eye.

    uv run python benchmarks/suites/render_ur5e_mbm.py out_dir [goal]

With `goal`, the arm is drawn at a task goal from the scene's dev queries, and at a free configuration otherwise.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ntfields.envs import ArmEnv
from ntfields.envs.solids import SolidScene

SUITE = Path(__file__).parent / "ur5e-mbm-v1"
VIEWS = ((20, -60), (20, 30), (70, -90))


def arm_config(name: str, env: ArmEnv, use_goal: bool) -> torch.Tensor:
    path = SUITE / "queries" / f"{name}.json"
    if use_goal and path.exists():
        dev = json.loads(path.read_text())["dev"]
        return torch.tensor(dev["goal"][dev["kind"].index("task")])[None]
    torch.manual_seed(0)
    q = torch.rand(4096, 6) - 0.5
    return q[env.distance(q) > 0.02][:1]


def render(name: str, out: Path, use_goal: bool) -> None:
    objects = json.loads((SUITE / "scenes" / f"{name}.json").read_text())["objects"]
    scene = [o for o in objects if not (o.get("role") == "mount" and o["id"] == "floor")]
    env = ArmEnv.ur5e_2f85(SolidScene.from_json(objects))
    triangles = SolidScene.from_json(scene).triangles().numpy()
    centers, radii = env.arm_spheres(arm_config(name, env, use_goal))
    arm = np.concatenate([centers[0].detach().numpy(), radii.numpy()[:, None]], 1)
    figure = plt.figure(figsize=(15, 5))
    for i, (elevation, azimuth) in enumerate(VIEWS):
        axes = figure.add_subplot(1, 3, i + 1, projection="3d")
        axes.add_collection3d(
            Poly3DCollection(triangles, alpha=0.25, facecolor="tab:green", edgecolor="0.5", linewidth=0.1)
        )
        axes.scatter(*arm[:, :3].T, s=(arm[:, 3] * 300) ** 2, c="tab:red", alpha=0.8)
        axes.set(xlim=(-0.8, 1.6), ylim=(-1.2, 1.2), zlim=(-0.9, 1.5), xlabel="x", ylabel="y", zlabel="z")
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elevation, azimuth)
    figure.suptitle(name)
    figure.tight_layout()
    figure.savefig(out / f"{name}.png", dpi=80)
    plt.close(figure)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted((SUITE / "scenes").glob("*.json")):
        render(path.stem, out, len(sys.argv) > 2)


if __name__ == "__main__":
    main()
