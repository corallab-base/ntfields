"""Build the 2D suites (`smoke-v1`, `box2d-v1`) into benchmarks/suites/. Needs the `gt` extra.

    uv run python benchmarks/suites/make_box2d.py

A released suite never changes: the script skips suites that already exist. To change one, add a new version.
"""

import logging
from pathlib import Path
from typing import Any

from ntfields.bench.build import build_suite
from ntfields.bench.protocol import Protocol

OUT = Path(__file__).parent
BOUNDS = [[-1.0, 1.0], [-1.0, 1.0]]
SPEED = {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}
PROTOCOL = Protocol(margin=0.01, resolution=0.005)


def scene(
    spheres: tuple[tuple[float, float, float], ...] = (), boxes: tuple[tuple[float, float, float, float], ...] = ()
) -> dict[str, Any]:
    """Spheres as (x, y, radius), boxes as (x, y, width, height)."""
    return {
        "kind": "sphere_box",
        "bounds": BOUNDS,
        "spheres": [{"center": [x, y], "radius": r} for x, y, r in spheres],
        "boxes": [{"center": [x, y], "size": [w, h]} for x, y, w, h in boxes],
    }


CLUTTER_SPHERES = ((-0.55, 0.45, 0.18), (0.1, 0.6, 0.15), (0.6, 0.25, 0.2), (-0.2, -0.05, 0.17), (0.35, -0.35, 0.15),
                   (-0.6, -0.55, 0.2), (0.75, -0.75, 0.12))  # fmt: skip
CLUTTER_BOXES = ((-0.15, 0.3, 0.5, 0.08), (0.25, 0.05, 0.08, 0.4), (-0.15, -0.6, 0.4, 0.1))

SCENES = {
    "empty": scene(),
    "one_sphere": scene(spheres=((0.0, 0.0, 0.3),)),
    "clutter": scene(spheres=CLUTTER_SPHERES, boxes=CLUTTER_BOXES),
    # two walls with 0.25-wide gaps at opposite ends: crossing both takes an S-shaped detour
    "walls": scene(boxes=((-0.35, -0.125, 0.1, 1.75), (0.35, 0.125, 0.1, 1.75))),
    # the same walls, 0.3 thick: S* is 0.1 inside obstacles, so crossing a thin wall is often cheaper than the
    # detour and a faithful field plans through it; thick walls make the detour the optimum for most queries
    "thick_walls": scene(boxes=((-0.35, -0.125, 0.3, 1.75), (0.35, 0.125, 0.3, 1.75))),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    smoke = {"one_sphere": SCENES["one_sphere"]}
    suites = [
        dict(name="smoke", description="CI smoke test: one sphere, 4 queries per split", scenes=smoke,
             n_queries={"dev": 4, "test": 4}, leaderboard=False, min_seeds=1),
        dict(name="box2d", description="2D point mass in [-1, 1]^2: empty, one sphere, clutter, walls, thick walls",
             scenes=SCENES, n_queries={"dev": 50, "test": 200}),
    ]  # fmt: skip
    for spec in suites:
        if (OUT / f"{spec['name']}-v1" / "suite.json").exists():
            logging.info("%s-v1 exists; skipping", spec["name"])
            continue
        build_suite(OUT, version=1, speed=SPEED, protocol=PROTOCOL, min_distance=0.5, fmm_resolution=401, seed=0,
                    **spec)  # fmt: skip


if __name__ == "__main__":
    main()
