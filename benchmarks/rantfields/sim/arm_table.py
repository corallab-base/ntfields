"""The paper's "PDE Model Comparison" on the 5-joint UR5e: RaNTF, N-Eik. and N-Riem. driven by `GeodesicTauController`
in MuJoCo from rest, on the paper's 100 detour pairs; means over the pairs every model completes collision-free.

    uv sync --group sim
    uv run python benchmarks/rantfields/sim/arm_table.py \\
        --models rantf=../nap/outputs/2026-09-15/02-52-19:100 n_eik=../nap/outputs/2026-09-14/21-37-49:100 \\
                 n_riem=../nap/outputs/2026-09-14/22-18-41:100

A model is `name=<reference run dir>:<epoch>` or `name=<ntfields ckpt.pt>`. `--reference-drift` gives the RaNTF
model the drift the reference's controller used online (not the one it was trained with), which is how the published
numbers were made; leave it off for the consistent version.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from sim.run import simulate, summarize
from sim.ur5e import UR5E_JOINTS, MuJoCoDynamics, torque_model

import ntfields
from ntfields.control import GeodesicTauConfig, GeodesicTauController
from ntfields.field import TrainedField
from ntfields.metric import ArmDynamics
from ntfields.problem import Problem

ACTIVE = UR5E_JOINTS[:5]
# the table's settings (reference results.json of planner_comparison_20260915_035701)
DT, MAX_TIME_S = 0.002, 30.0
CONTROL = GeodesicTauConfig(cruise_speed=1.15, lam=0.5, goal_tol=0.05)


def load(spec: str, device: str) -> TrainedField:
    if spec.endswith(".pt"):
        return ntfields.load(spec, device=device)
    from nap_import import load_nap_run

    run, epoch = spec.rsplit(":", 1)
    return load_nap_run(run, int(epoch), device)


def with_reference_drift(field: TrainedField) -> TrainedField:
    from problems import ReferenceOnlineDrift

    m = field.problem.metric
    if not (isinstance(m, ArmDynamics) and m.via_anisotropy):
        return field
    metric = ReferenceOnlineDrift(**{f.name: getattr(m, f.name) for f in dataclasses.fields(m) if f.init})
    return dataclasses.replace(field, problem=Problem(field.problem.env, field.problem.speed, metric))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="name=<run dir>:<epoch> or name=<ckpt.pt>")
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--reference-drift", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu answers single queries faster than a GPU for this network")
    p.add_argument("--out", type=Path, default=None, help="per-pair metrics as JSON")
    args = p.parse_args(argv)

    pairs = json.loads((Path(__file__).parent / "table_pairs.json").read_text())["pairs"][: args.pairs]
    model = torque_model(ACTIVE)
    dynamics = MuJoCoDynamics(model, ACTIVE)
    results: dict[str, list[dict]] = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        field = load(path, args.device)
        if args.reference_drift:
            field = with_reference_drift(field)
        env = field.problem.env
        scale = np.asarray(env.joint_scale)
        bounds = np.stack([-scale, scale], axis=1)
        results[name] = []
        for i, pair in enumerate(pairs):
            start, goal = np.asarray(pair["start"]) * scale, np.asarray(pair["goal"]) * scale
            tau_max = np.full(5, np.inf)  # the simulator clips to the actuator limits
            controller = GeodesicTauController(field, dynamics, goal, tau_max, DT, CONTROL, bounds, args.device)
            m = summarize(simulate(model, ACTIVE, controller, start, DT, MAX_TIME_S), env)
            results[name].append(m)
            print(f"{name} pair {i}: {m}", flush=True)

    ok = [all(r[i]["converged"] and r[i]["collision_free"] for r in results.values()) for i in range(len(pairs))]
    print(f"\n{sum(ok)}/{len(pairs)} pairs completed collision-free by every model; means over those:")
    print("| model | success % | time to goal (s) | traced length (rad) | peak torque (N·m) | control rate (Hz) |")
    print("|---|---|---|---|---|---|")
    for name, rs in results.items():
        success = 100 * np.mean([r["converged"] and r["collision_free"] for r in rs])
        sel = [r for r, k in zip(rs, ok, strict=True) if k]
        mean = {k: np.mean([r[k] for r in sel]) if sel else float("nan")
                for k in ("time_to_goal", "traced_length", "peak_torque", "control_rate_hz")}  # fmt: skip
        print(f"| {name} | {success:.0f} | {mean['time_to_goal']:.2f} | {mean['traced_length']:.2f} | "
              f"{mean['peak_torque']:.1f} | {mean['control_rate_hz']:.0f} |")  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
