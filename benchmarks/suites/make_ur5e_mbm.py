"""Build `ur5e-mbm-v1`, five MotionBenchMaker scenes for a UR5e with a 2F-85 gripper, into benchmarks/suites/.

    NTFIELDS_MBM=/path/to/motion_bench_maker uv run python benchmarks/suites/make_ur5e_mbm.py [scenes|queries|all]

Trains nothing. `scenes` writes the scenes, point clouds and oracle boxes; `queries` also generates the queries, each
solved by a bidirectional RRT, then writes suite.json. Finished parts are kept, so an interrupted run resumes. A
released suite never changes: the script stops if suite.json exists.
"""

import json
import logging
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mbm
import numpy as np
import torch
from torch import Tensor

from ntfields.bench.decomposition import evaluate_boxes
from ntfields.bench.protocol import Protocol, evaluate_path
from ntfields.bench.rrt import bidirectional_rrt
from ntfields.bench.suite import FORMAT_VERSION, Queries, SuiteProblem, sha256
from ntfields.envs import ArmEnv
from ntfields.envs.solids import SolidScene, quat_matrix
from ntfields.envs.ur5e_ik import solve_ur5e_ik

OUT = Path(__file__).parent / "ur5e-mbm-v1"
SPEED = {"kind": "clipped", "dmin": 0.005, "dmax": 0.05}
PROTOCOL = Protocol(margin=0.005, resolution=0.0025)
N_QUERIES = {"dev": 50, "test": 200}
MIN_DISTANCE = 0.25  # in q; about 90° of joint motion
JITTER = 0.03  # m, around each of MBM's grasp positions
TILT = math.radians(10)
FULL_POINTS = 100_000
VOXEL = 0.01
SEED = 0
QUERY_SEED = 1000
MAX_ATTEMPTS = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MBM problem, query file, and the frozen joint angles of URDF scenes: the left cupboard door and the dishwasher are
# open because MBM's kitchen queries reach into them; the right cupboard door stays shut
SCENES: dict[str, tuple[str, str, dict[str, float]]] = {
    "cage": ("cage", "cage/queries_cage.yaml", {}),
    "bookshelf_tall": ("bookshelf_tall", "bookshelf/all_queries_tall.yaml", {}),
    "box": ("box", "box/all_queries_box.yaml", {}),
    "table_under_pick": ("table_under_pick", "table/queries_under_pick_table.yaml", {}),
    "kitchen": ("kitchen", "kitchen/hard_queries.yaml",
                {"cupboard_door_l_joint": -math.pi / 2, "dishwasher_door_joint": -math.pi / 2}),
}  # fmt: skip


def query_url(file: str) -> str:
    return f"package://motion_bench_maker/configs/scenes/{file}"


def load_objects(root: Path, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problem, _, angles = SCENES[name]
    config = mbm.problem_config(root, problem)
    is_urdf = config["scene"].endswith(".urdf")
    objects = mbm.urdf_objects(root, config, angles) if is_urdf else mbm.yaml_objects(root, config)
    return config, objects


def scene_json(name: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "arm", "robot": "ur5e_2f85", "source": f"motion_bench_maker@{mbm.MBM_COMMIT}", "problem": name,
            "objects": objects + mbm.mount()}  # fmt: skip


def fit_voxels(objects: list[dict[str, Any]], size: float = 0.01) -> list[dict[str, Any]]:
    """Axis-aligned boxes from a greedy merge of an occupancy grid, for scenes that have no primitives to start from."""
    scene = SolidScene.from_json(objects)
    corners = scene.triangles().reshape(-1, 3)
    lo = corners.amin(0) - size
    shape = (((corners.amax(0) + size - lo) / size).ceil().long()).tolist()
    axes = [lo[i] + size * (torch.arange(shape[i], dtype=torch.float32) + 0.5) for i in range(3)]
    p = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occupied = torch.cat([scene(c.to(DEVICE)).cpu() < 0 for c in p.split(200_000)]).view(shape).numpy().copy()
    boxes = []
    while occupied.any():
        i, j, k = (int(v[0]) for v in np.nonzero(occupied))
        ends = [i + 1, j + 1, k + 1]
        grew = True
        while grew:
            grew = False
            for axis in range(3):
                span = [slice(a, b) for a, b in zip((i, j, k), ends, strict=True)]
                span[axis] = slice(ends[axis], ends[axis] + 1)
                if ends[axis] < shape[axis] and occupied[tuple(span)].all():
                    ends[axis] += 1
                    grew = True
        occupied[i : ends[0], j : ends[1], k : ends[2]] = False
        low = lo + size * torch.tensor([i, j, k], dtype=torch.float32)
        extent = size * (torch.tensor(ends, dtype=torch.float32) - torch.tensor([i, j, k], dtype=torch.float32))
        boxes.append({"center": (low + extent / 2).tolist(), "size": extent.tolist(), "quat": [0.0, 0.0, 0.0, 1.0]})
    return boxes


def oracle_boxes(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A best-known box fit: boxes as they are, the box around each cylinder, and voxel-merged boxes for meshes."""
    out = []
    for o in objects:
        if o["type"] == "box":
            out.append({"center": o["center"], "size": o["size"], "quat": o["quat"]})
        elif o["type"] == "cylinder":
            out.append(
                {"center": o["center"], "size": [2 * o["radius"], 2 * o["radius"], o["height"]], "quat": o["quat"]}
            )
    meshes = [o for o in objects if o["type"] == "mesh"]
    return out + (fit_voxels(meshes) if meshes else [])


def write_scene(root: Path, name: str, generator: torch.Generator) -> None:
    path = OUT / "scenes" / f"{name}.json"
    if path.exists():
        return
    config, objects = load_objects(root, name)
    scene = SolidScene.from_json(objects)
    triangles = scene.triangles().to(DEVICE)
    full = mbm.surface_points(triangles.cpu(), FULL_POINTS, generator)
    views, camera = mbm.cameras(root, config, objects)
    hits = []
    for view in views.to(DEVICE, torch.float32):
        origin, rays = mbm.view_rays(view, camera)
        hits.append(mbm.first_hit(origin, rays, triangles))
    sensed = mbm.voxel_downsample(torch.cat(hits), VOXEL).cpu()
    boxes = oracle_boxes(objects)
    fit = evaluate_boxes(scene_json(name, objects)["objects"], boxes, PROTOCOL.margin, generator, device=DEVICE)
    logging.info("%s: %d objects, %d sensed points from %d views, oracle %s", name, len(objects), len(sensed),
                 len(views), asdict(fit))  # fmt: skip
    for folder in ("scenes", "points", "oracle"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    np.save(OUT / "points" / f"{name}_full.npy", full.numpy().astype(np.float32))
    np.save(OUT / "points" / f"{name}_sensed.npy", sensed.numpy().astype(np.float32))
    (OUT / "oracle" / f"{name}.json").write_text(json.dumps({"boxes": boxes, "fit": asdict(fit)}) + "\n")
    path.write_text(json.dumps(scene_json(name, objects)) + "\n")  # last, so its presence means the rest is there


def ik_configs(env: ArmEnv, targets: list[mbm.Target], n: int, generator: torch.Generator) -> Tensor:
    """Valid configurations (K, 6) reaching around MBM's grasp poses: position jittered, tilted a little, and any roll
    about the tool axis. MBM's end effector looks along x and ours along wrist_3's z."""
    pick = torch.randint(len(targets), (n,), generator=generator)
    position = torch.tensor([targets[i].position for i in pick], dtype=torch.float64)
    position += (2 * torch.rand(n, 3, generator=generator, dtype=torch.float64) - 1) * JITTER
    rotation = quat_matrix(torch.tensor([targets[i].quat for i in pick], dtype=torch.float64))
    half = math.sqrt(0.5)
    to_tool = quat_matrix(torch.tensor([0.0, half, 0.0, half], dtype=torch.float64))
    roll = (2 * torch.rand(n, generator=generator, dtype=torch.float64) - 1) * math.pi
    tilt = (2 * torch.rand(n, 2, generator=generator, dtype=torch.float64) - 1) * TILT
    zero = torch.zeros(n, dtype=torch.float64)
    spin = torch.stack([zero, zero, (roll / 2).sin(), (roll / 2).cos()], -1)
    tilt_x = torch.stack([(tilt[:, 0] / 2).sin(), zero, zero, (tilt[:, 0] / 2).cos()], -1)
    tilt_y = torch.stack([zero, (tilt[:, 1] / 2).sin(), zero, (tilt[:, 1] / 2).cos()], -1)
    poses = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    poses[:, :3, :3] = rotation @ to_tool @ quat_matrix(tilt_x) @ quat_matrix(tilt_y) @ quat_matrix(spin)
    poses[:, :3, 3] = position
    q = (solve_ur5e_ik(poses).reshape(-1, 6) / (2 * math.pi)).float().to(DEVICE)
    reached = env._chain(q).forward_kinematics(q * env.joint_scale).get_matrix()[:, :3, 3]
    wanted = poses[:, :3, 3].repeat_interleave(8, dim=0).to(reached)
    return q[(reached - wanted).norm(dim=-1) < 1e-3]


def free_fn(env: ArmEnv, margin: float) -> Any:
    lo, hi = env.bounds[:, 0].to(DEVICE), env.bounds[:, 1].to(DEVICE)

    def free(q: Tensor) -> Tensor:
        ok = ((q >= lo) & (q <= hi)).all(dim=-1)
        ok[ok.clone()] = env.distance(q[ok]) > margin
        if ok.any():
            ok[ok.clone()] = env.self_distance(q[ok]) > margin
        return ok

    return free


def pool(free: Any, q: Tensor) -> Tensor:
    return q[torch.cat([free(c) for c in q.split(16384)])] if len(q) else q


def random_configs(free: Any, n: int, generator: torch.Generator) -> Tensor:
    return pool(free, (torch.rand(n, 6, generator=generator) - 0.5).to(DEVICE))


def make_queries(root: Path, name: str, generator: torch.Generator) -> dict[str, Any]:
    """Half the pairs of each split are tasks, from MBM's start grasp (or anywhere, if it names none) to its goal grasp;
    half are uniform. Every pair has an obstacle on its straight line, with or without self-collision, and is solved by
    RRT without self-collision, so both settings can use the same queries."""
    config, objects = load_objects(root, name)
    url = query_url(SCENES[name][1])
    goals, starts = (mbm.targets(config, objects, root, url, k) for k in ("goal_queries", "start_queries"))
    scene = json.loads((OUT / "scenes" / f"{name}.json").read_text())
    problem = SuiteProblem(name, scene, SPEED, {}).problem()
    env = problem.env
    assert isinstance(env, ArmEnv)
    free = free_fn(env, PROTOCOL.margin)
    strict = Protocol(PROTOCOL.margin, PROTOCOL.resolution, self_collision=True)
    goal_pool = pool(free, ik_configs(env, goals, 20_000, generator))
    start_pool = pool(free, ik_configs(env, starts, 20_000, generator)) if starts else None
    logging.info("%s: %d valid IK goals of %d targets, %s valid IK starts", name, len(goal_pool), len(goals),
                 None if start_pool is None else len(start_pool))  # fmt: skip
    if not len(goal_pool) or (start_pool is not None and not len(start_pool)):
        raise RuntimeError(f"{name}: MBM's targets have no valid IK solution on this robot")
    lo, hi = env.bounds[:, 0].to(DEVICE), env.bounds[:, 1].to(DEVICE)
    out: dict[str, Any] = {}
    tried = solved = skipped = 0
    for split, n in N_QUERIES.items():
        rows: list[tuple[Tensor, Tensor, str]] = []
        for kind, want in (("task", n // 2), ("uniform", n - n // 2)):
            found = 0
            while found < want:
                uniform = random_configs(free, 4096, generator)
                if kind == "task":
                    g = goal_pool[torch.randint(len(goal_pool), (1,), generator=generator)[0]]
                    s = uniform[0]
                    if start_pool is not None:
                        s = start_pool[torch.randint(len(start_pool), (1,), generator=generator)[0]]
                else:
                    s, g = uniform[0], uniform[1]
                if (s - g).norm() < MIN_DISTANCE:
                    continue
                if evaluate_path(problem, PROTOCOL, s.cpu(), g.cpu(), torch.stack([s, g]).cpu()).success:
                    skipped += 1  # the straight line is free of obstacles, so the query tests nothing
                    continue
                tried += 1
                if tried > MAX_ATTEMPTS:
                    raise RuntimeError(f"{name}: only {solved} of {tried} pairs were solved")
                path = bidirectional_rrt(free, s, g, lo, hi, generator, PROTOCOL.resolution)
                if path is None or not evaluate_path(problem, strict, s.cpu(), g.cpu(), path.cpu()).success:
                    continue
                solved += 1
                found += 1
                rows.append((s.cpu(), g.cpu(), kind))
            logging.info(
                "%s %s: %d %s pairs; %d of %d tried so far were solved", name, split, found, kind, solved, tried
            )
        queries = Queries(torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows]),
                          straight_free=torch.zeros(len(rows), dtype=torch.bool))  # fmt: skip
        out[split] = {**queries.to_json(), "kind": [r[2] for r in rows]}
    return {**out, "attempts": {"tried": tried, "solved": solved, "straight_skipped": skipped}}


def write_queries(root: Path, name: str, generator: torch.Generator) -> None:
    path = OUT / "queries" / f"{name}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = make_queries(root, name, generator)
        path.write_text(json.dumps(data) + "\n")


def write_suite() -> None:
    files = sorted(p for folder in ("scenes", "queries", "points", "oracle") for p in (OUT / folder).iterdir())
    meta = {
        "format_version": FORMAT_VERSION,
        "name": "ur5e-mbm",
        "version": 1,
        "description": "UR5e + 2F-85 in five nominal MotionBenchMaker scenes; RRT-solved queries, no optimal reference",
        "leaderboard": True,
        "min_seeds": 3,
        "protocol": asdict(PROTOCOL),
        "reference": {"kind": "rrt", "seed": SEED, "query_seed": QUERY_SEED, "min_distance": MIN_DISTANCE,
                      "mbm_commit": mbm.MBM_COMMIT},
        "problems": [{"name": n, "scene": f"scenes/{n}.json", "queries": f"queries/{n}.json", "speed": SPEED}
                     for n in SCENES],
        "files": {str(p.relative_to(OUT)): sha256(p) for p in files},
    }  # fmt: skip
    (OUT / "suite.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if (OUT / "suite.json").exists():
        logging.info("%s exists; skipping", OUT.name)
        return
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    root = Path(os.environ["NTFIELDS_MBM"])
    for index, name in enumerate(SCENES):
        write_scene(root, name, torch.Generator().manual_seed(SEED + index))
        if stage != "scenes":
            write_queries(root, name, torch.Generator().manual_seed(QUERY_SEED + index))
    if stage != "scenes":
        write_suite()


if __name__ == "__main__":
    main()
