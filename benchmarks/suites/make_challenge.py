"""Add the detour-only challenge pairs to `box2d-v1` and `box3d-v1`. Needs the `gt` extra.

    uv run python benchmarks/suites/make_challenge.py

`empty` has no obstacle to detour around. suite.json is not touched, so existing records stay valid.
"""

import logging
from pathlib import Path

from ntfields.bench.build import build_challenge

OUT = Path(__file__).parent
N = 200
SEED = 2000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for suite in ("box2d-v1", "box3d-v1"):
        build_challenge(OUT / suite, N, SEED, skip=("empty",))


if __name__ == "__main__":
    main()
