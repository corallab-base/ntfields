import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ntfields.bench.protocol import Protocol, evaluate_path, is_valid
from ntfields.bench.suite import FORMAT_VERSION, SPLITS, Queries, SuiteProblem, sha256
from ntfields.problem import Problem

log = logging.getLogger(__name__)


def build_suite(
    out: Path,
    name: str,
    version: int,
    description: str,
    scenes: dict[str, dict[str, Any]],
    speed: dict[str, Any],
    protocol: Protocol,
    n_queries: dict[str, int],
    min_distance: float,
    fmm_resolution: int,
    seed: int,
    leaderboard: bool = True,
    min_seeds: int = 3,
) -> None:
    """Write a 2D/3D suite to `out/<name>-v<version>/`: scenes, dev and test queries with FMM references, and
    suite.json with every file's checksum. Needs the `gt` extra. Refuses to overwrite a released suite."""
    base = out / f"{name}-v{version}"
    if (base / "suite.json").exists():
        raise FileExistsError(f"{base} exists; a released suite never changes, so bump the version")
    (base / "scenes").mkdir(parents=True, exist_ok=True)
    (base / "queries").mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    files: dict[str, str] = {}
    problems = []
    for scene_name, scene in scenes.items():
        problem = SuiteProblem(scene_name, scene, speed, {}).problem()
        queries = {
            split: reference_queries(problem, protocol, n, min_distance, fmm_resolution, generator)
            for split, n in n_queries.items()
        }
        for rel, data in ((f"scenes/{scene_name}.json", scene), (f"queries/{scene_name}.json",
                          {s: q.to_json() for s, q in queries.items()})):  # fmt: skip
            (base / rel).write_text(json.dumps(data) + "\n")
            files[rel] = sha256(base / rel)
        problems.append({"name": scene_name, "scene": f"scenes/{scene_name}.json",
                         "queries": f"queries/{scene_name}.json", "speed": speed})  # fmt: skip
        log.info("%s/%s: %s", base.name, scene_name, {s: len(q) for s, q in queries.items()})
    meta = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "version": version,
        "description": description,
        "leaderboard": leaderboard,
        "min_seeds": min_seeds,
        "protocol": vars(protocol),
        "reference": {"kind": "fmm", "resolution": fmm_resolution, "seed": seed, "min_distance": min_distance},
        "problems": problems,
        "files": files,
    }
    assert set(n_queries) == set(SPLITS)
    (base / "suite.json").write_text(json.dumps(meta, indent=2) + "\n")


def reference_queries(
    problem: Problem,
    protocol: Protocol,
    n: int,
    min_distance: float,
    fmm_resolution: int,
    generator: torch.Generator,
    max_tries: int = 100,
) -> Queries:
    """n uniform (start, goal) pairs, both valid and at least `min_distance` apart, each kept only if the FMM path
    between them passes the protocol; c* is that path's cost and t* the FMM travel time."""
    from ntfields.gt import fmm_travel_time

    bounds = problem.env.bounds
    lo, hi = bounds.unbind(-1)
    step = float((hi - lo).max()) / (fmm_resolution - 1) / 2
    rows: list[tuple[Tensor, Tensor, float, float, bool]] = []
    for _ in range(max_tries * n):
        if len(rows) == n:
            break
        s, g = lo + (hi - lo) * torch.rand(2, problem.dim, generator=generator)
        if (s - g).norm() < min_distance or not bool(is_valid(problem, torch.stack([s, g]), protocol.margin).all()):
            continue
        gt = fmm_travel_time(problem, g, fmm_resolution, margin=protocol.margin)
        t_star = float(gt.value(s[None])[0])
        path = gt.descend(s, step) if t_star < float("inf") else None
        if path is None:
            continue
        outcome = evaluate_path(problem, protocol, s, g, path)
        if not outcome.success or outcome.cost is None:
            continue
        straight = evaluate_path(problem, protocol, s, g, torch.stack([s, g])).success
        rows.append((s, g, outcome.cost, t_star, straight))
    if len(rows) < n:
        raise RuntimeError(f"found only {len(rows)} of {n} queries with a valid reference path")
    return Queries(
        torch.stack([r[0] for r in rows]),
        torch.stack([r[1] for r in rows]),
        torch.tensor([r[2] for r in rows], dtype=torch.float64),
        torch.tensor([r[3] for r in rows], dtype=torch.float64),
        torch.tensor([r[4] for r in rows]),
    )
