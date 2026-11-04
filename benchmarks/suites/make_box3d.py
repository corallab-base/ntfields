"""Build the 3D suite (`box3d-v1`) into benchmarks/suites/. Needs the `gt` extra.

    uv run python benchmarks/suites/make_box3d.py

A released suite never changes: the script skips suites that already exist. To change one, add a new version.
"""

import logging
from pathlib import Path
from typing import Any

from ntfields.bench.build import build_suite
from ntfields.bench.protocol import Protocol

OUT = Path(__file__).parent
BOUNDS = [[-1.0, 1.0]] * 3
SPEED = {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}
PROTOCOL = Protocol(margin=0.01, resolution=0.005)

Vec = tuple[float, float, float]


def scene(spheres: tuple[tuple[Vec, float], ...] = (), boxes: tuple[tuple[Vec, Vec], ...] = ()) -> dict[str, Any]:
    """Spheres as (center, radius), boxes as (center, size)."""
    return {
        "kind": "sphere_box",
        "bounds": BOUNDS,
        "spheres": [{"center": list(c), "radius": r} for c, r in spheres],
        "boxes": [{"center": list(c), "size": list(s)} for c, s in boxes],
    }


# S* is 0.1 inside obstacles, so every obstacle is at least 0.24 thick: passing through one is then dearer than going
# around it (the builder reports the share of queries where it is not; see `walls` in box2d-v1)
CLUTTER_SPHERES = (((-0.5, 0.4, 0.2), 0.25), ((0.3, -0.4, -0.3), 0.3), ((0.45, 0.5, 0.45), 0.2),
                   ((-0.35, -0.5, 0.5), 0.2), ((0.6, 0.0, -0.6), 0.18), ((-0.6, 0.6, -0.5), 0.2))  # fmt: skip
CLUTTER_BOXES = (((0.0, 0.0, 0.0), (0.3, 0.8, 0.3)), ((-0.5, -0.05, -0.5), (0.6, 0.25, 0.25)),
                 ((0.55, -0.6, 0.45), (0.25, 0.25, 0.6)))  # fmt: skip
# a plate across x = 0 with one 0.4 x 0.4 hole at (y, z) in [0.25, 0.65]^2, from four boxes
WINDOW = (((0.0, -0.375, 0.0), (0.3, 1.25, 2.0)), ((0.0, 0.825, 0.0), (0.3, 0.35, 2.0)),
          ((0.0, 0.45, -0.375), (0.3, 0.4, 1.25)), ((0.0, 0.45, 0.825), (0.3, 0.4, 0.35)))  # fmt: skip

SCENES = {
    "empty": scene(),
    "four_spheres": scene(spheres=tuple(((x, y, 0.0), 0.25) for x in (-0.5, 0.5) for y in (-0.5, 0.5))),
    "clutter": scene(spheres=CLUTTER_SPHERES, boxes=CLUTTER_BOXES),
    "window": scene(boxes=WINDOW),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if (OUT / "box3d-v1" / "suite.json").exists():
        logging.info("box3d-v1 exists; skipping")
        return
    build_suite(OUT, "box3d", 1, "3D point mass in [-1, 1]^3: empty, four spheres, clutter, a window", SCENES, SPEED,
                PROTOCOL, {"dev": 50, "test": 200}, min_distance=0.5, fmm_resolution=201, seed=0)  # fmt: skip


if __name__ == "__main__":
    main()
