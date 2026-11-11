from collections.abc import Callable

import torch

from ntfields.base import Environment
from ntfields.envs.arm import ArmEnv
from ntfields.envs.constraint import ConstrainedEnv, Constraint, SphereConstraint
from ntfields.envs.sphere_box import SphereBoxEnv

PRESETS: dict[str, Callable[[], Environment]] = {
    "box2d/empty": lambda: SphereBoxEnv.build(bounds=[(-1, 1)] * 2),
    "box2d/one_sphere": lambda: SphereBoxEnv.build(bounds=[(-1, 1)] * 2, spheres=[((0.0, 0.0), 0.3)]),
    # RaNTFields' 2D scenes
    "box2d/small_sphere": lambda: SphereBoxEnv.build(bounds=[(-1, 1)] * 2, spheres=[((0.0, 0.0), 0.15)]),
    "box2d/two_spheres": lambda: SphereBoxEnv.build(
        bounds=[(-1, 1)] * 2, spheres=[((-0.4, 0.0), 0.2), ((0.4, 0.0), 0.2)]
    ),
    "box2d/three_small_spheres": lambda: SphereBoxEnv.build(
        bounds=[(-1, 1)] * 2, spheres=[((0.0, 0.55), 0.15), ((-0.476, -0.275), 0.12), ((0.476, -0.275), 0.1)]
    ),
    "box3d/four_spheres": lambda: SphereBoxEnv.build(
        bounds=[(-1, 1)] * 3,
        spheres=[((x, y, 0.0), 0.25) for x in (-0.5, 0.5) for y in (-0.5, 0.5)],
    ),
    "ur5e/ntrl_scene": ArmEnv.ur5e_ntrl_scene,  # needs the [arm] extra
    # RaNTFields' arm: 5 joints (wrist_3 at 0) and a 0.2 m sphere in the cabinet region
    "ur5e/5j_sphere": lambda: ArmEnv.ur5e_spherized(5, torch.tensor([[-0.55, 0.0, 0.2, 0.2]])),
    "ur5e/cup": lambda: _ur5e_cup_env(),  # needs the [arm] extra; use with speed.GaussianBand
    "ur5e/cup_table": lambda: _ur5e_cup_env(table=True),
}


def _ur5e_cup_env(table: bool = False) -> Environment:
    from ntfields.envs.tsr import ur5e_cup_env

    return ur5e_cup_env(table)


def load_env(name: str) -> Environment:
    if name not in PRESETS:
        raise KeyError(f"unknown env {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()


__all__ = [
    "PRESETS",
    "ArmEnv",
    "ConstrainedEnv",
    "Constraint",
    "SphereBoxEnv",
    "SphereConstraint",
    "load_env",
]
