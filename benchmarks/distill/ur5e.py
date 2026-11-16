"""Distill nap's TD-NTFields UR5e cabinet model (256x2) into smaller students on the UR5e benchmark.

The teacher is nap's `outputs/2026-10-01/02-19-55` run (`+preset=ntrl_demo`: ntrl-demo's network, trained on nap's
cabinet pairs) at epoch 5000, whose `PirateNetMetric` head is TDMetricModel's network under other parameter names; it
is converted to an ntfields checkpoint once. Students are fit on pairs uniform over the free space and scored with
`benchmark_ur5e` (MPPI, 5 scenarios + 50 random pairs, 3 seeds), against the teacher on random free pairs, and by
accelerated MPPI time per plan.

    uv run python benchmarks/distill/ur5e.py
"""

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import h5py
import torch
import tyro

import ntfields
from ntfields import DistillConfig, Epochs, Problem, TrainConfig, TrainedField
from ntfields.checkpoint import save
from ntfields.data import Mixture, PairDataset, Sampler, Uniform, UR5eSurfacePairs, sample_pairs, sample_valid
from ntfields.envs import load_env
from ntfields.envs.arm import ntrl_random_pairs
from ntfields.eval.ur5e import benchmark_ur5e
from ntfields.methods.td_ntfields import TDNTFields, TDNTFieldsConfig
from ntfields.methods.td_ntfields.model import TDMetricModel
from ntfields.plan import mppi
from ntfields.speed import ClippedEased
from ntfields.train import LRScheduler

NAP = Path(os.environ.get("NTFIELDS_NAP", Path(__file__).resolve().parents[3] / "nap"))
NAP_CKPT = NAP / "outputs/2026-10-01/02-19-55/checkpoints/model_epoch_5000.pth"
NAP_DATA = NAP / "data/continuous/dim6/ur5e_arm_6j_cabinet_pairs/no_flow/default/2026-10-01_02-15-24/dataset.h5"


@dataclass
class Args:
    sizes: tuple[str, ...] = ("128x2", "128x1", "64x1")  # hidden x n_blocks
    epochs: int = 2000
    n_pairs: int = 200_000
    w_grad: float = 0.1
    weight_decay: float = 0.0
    lr: float = 5e-4
    # uniform: free space; surface: TD-NTFields' arm pairs (UR5eSurfacePairs); mix: half each; nap: the teacher's own
    # 500k cabinet training pairs; nap_mix: those plus as many uniform free-space pairs (n_pairs is ignored for both)
    sampler: Literal["uniform", "surface", "mix", "nap", "nap_mix"] = "uniform"
    near_weight: float = 0.0  # DistillConfig.near_weight
    w_physics: float = 0.0  # weight of TD-NTFields' own loss (arm_ntrl recipe) on the same pairs
    cosine: bool = False  # cosine LR decay to lr / 100 over the epochs
    seed: int = 0  # the student's init and shuffling
    tag: str = ""  # appended to result keys and file names
    teacher_eval: bool = True
    out: Path = Path("runs/distill/ur5e")


SAMPLERS: dict[str, Sampler] = {
    "uniform": Uniform(),
    "surface": UR5eSurfacePairs(),
    "mix": Mixture((UR5eSurfacePairs(), Uniform()), (0.5, 0.5)),
}


def problem() -> Problem:
    return Problem(load_env("ur5e/ntrl_scene"), ClippedEased(dmin=0.005, dmax=0.05))


def teacher_checkpoint(out: Path) -> Path:
    """nap's PirateNetMetric checkpoint as an ntfields td_ntfields_arm_ntrl checkpoint."""
    path = out / "teacher_nap_2026-10-01_02-19-55_epoch5000.pt"
    if path.exists():
        return path
    ckpt = torch.load(NAP_CKPT, map_location="cpu", weights_only=True)
    state = {k.removeprefix("predict_time."): v for k, v in ckpt["model_state_dict"].items()}
    ours = {"fourier.b": state["fourier_b"]}
    for name, theirs in (("gate_u", "pe_gate_u"), ("gate_v", "pe_gate_v"), ("final", "final_linear")):
        for p in ("weight", "bias"):
            ours[f"{name}.{p}"] = state[f"{theirs}.linear.{p}"]
    for i in range(2):
        for j in range(3):
            for p in ("weight", "bias"):
                ours[f"blocks.{i}.lin{j + 1}.{p}"] = state[f"blocks.{i}.{j}.linear.{p}"]
        ours[f"block_gates.{i}"] = state[f"block_gates.{i}"].reshape(())
    method = TDNTFields(TDNTFieldsConfig.arm_ntrl())
    model = method.build_model(problem())
    assert isinstance(model, TDMetricModel)
    model.load_state_dict(ours)
    save(path, method, problem(), model, int(ckpt["epoch"]))
    return path


def nap_pairs(problem: Problem, out: Path) -> PairDataset:
    """The teacher's training pairs (radians -> q = θ / 2π), with S* and normals from ntfields' environment."""
    path = out / "nap_pairs.h5"
    if path.exists():
        return PairDataset.load(path)
    with h5py.File(NAP_DATA, "r") as f:
        q_s, q_g = (_read(f, k) / (2 * math.pi) for k in ("start_points", "goal_points"))
    data = _with_speed(problem, q_s, q_g)
    data.save(path)
    return data


def _read(f: h5py.File, key: str) -> torch.Tensor:
    ds = f[key]
    assert isinstance(ds, h5py.Dataset)
    return torch.from_numpy(ds[()])


def uniform_pairs(problem: Problem, n: int, out: Path) -> PairDataset:
    path = out / f"uniform_pairs_{n}.h5"
    if path.exists():
        return PairDataset.load(path)
    data = sample_pairs(problem, n, Uniform(), seed=0, device="cuda")
    data.save(path)
    return data


def _with_speed(problem: Problem, q_s: torch.Tensor, q_g: torch.Tensor) -> PairDataset:
    chunks = [problem.speed_and_normal(q.cuda()) for q in torch.cat([q_s, q_g]).split(65536)]
    speed = torch.cat([c[0] for c in chunks]).cpu()
    normal = torch.cat([c[1] for c in chunks]).cpu()
    n = len(q_s)
    return PairDataset(q_s, q_g, speed[:n], speed[n:], normal[:n], normal[n:])


def cat(*parts: PairDataset) -> PairDataset:
    fields = ("q_s", "q_g", "speed_s", "speed_g", "normal_s", "normal_g")
    return PairDataset(**{f: torch.cat([getattr(p, f) for p in parts]) for f in fields})


def pairs_for(args: "Args", problem: Problem) -> PairDataset:
    if args.sampler == "nap":
        return nap_pairs(problem, args.out)
    if args.sampler == "nap_mix":
        nap = nap_pairs(problem, args.out)
        return cat(nap, uniform_pairs(problem, len(nap), args.out))
    return sample_pairs(problem, args.n_pairs, SAMPLERS[args.sampler], seed=0, device="cuda")


def evaluate(field: TrainedField, teacher: TrainedField, q_s: torch.Tensor, q_g: torch.Tensor) -> dict[str, float]:
    bench = benchmark_ur5e(field)

    def rate(group: str, key: str) -> float:
        return sum(c[key] for c in bench[group]) / sum(c["n"] for c in bench[group])

    t, tt = field.travel_time(q_s, q_g), teacher.travel_time(q_s, q_g)
    g, tg = field.gradient(q_s, q_g)[0], teacher.gradient(q_s, q_g)[0]
    fast = field.accelerate("compile")
    starts, goals = (x.to(field.device) for x in ntrl_random_pairs())
    mppi(fast, starts[0], goals[0])  # compile
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for s, gl in zip(starts, goals, strict=True):
        mppi(fast, s, gl)
    torch.cuda.synchronize()
    return {
        "scenarios": rate("scenarios", "success"),
        "random": rate("random", "success"),
        "random_converged": rate("random", "converged"),
        "random_collision_free": rate("random", "collision_free"),
        "teacher_err": float((t - tt).abs().mean()),
        "teacher_rel_err": float(((t - tt).abs() / tt.abs().clamp_min(1e-3)).mean()),
        "grad_cos": float(torch.nn.functional.cosine_similarity(g, tg, dim=-1).mean()),
        "mppi_ms": 1000 * (time.monotonic() - t0) / len(starts),
    }


def cosine(epochs: int, lr: float) -> LRScheduler:
    return LRScheduler(lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100))


def main(args: Args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = ntfields.load(teacher_checkpoint(args.out), device="cuda")
    gen = torch.Generator().manual_seed(0)
    q_s, q_g = (sample_valid(teacher.problem, 1000, gen).cuda() for _ in range(2))
    results: dict[str, dict[str, float]] = {}
    if args.teacher_eval:
        results["teacher 256x2"] = evaluate(teacher, teacher, q_s, q_g)
        print("teacher 256x2", results["teacher 256x2"], flush=True)
    tc = TrainConfig(
        n_pairs=args.n_pairs, batch_size=2000, batches_per_epoch=5, out_dir=args.out, log_every=10**9, seed=args.seed
    )
    pairs = pairs_for(args, teacher.problem)
    print(f"{len(pairs)} pairs", flush=True)
    name = f"{args.sampler}{args.tag}"
    for size in args.sizes:
        hidden, n_blocks = map(int, size.split("x"))
        cfg = replace(teacher.method.cfg, hidden=hidden, n_blocks=n_blocks, weight_decay=args.weight_decay, lr=args.lr)
        t0 = time.monotonic()
        field = ntfields.distill(
            teacher,
            "td_ntfields",
            Epochs(args.epochs),
            cfg,
            DistillConfig(w_grad=args.w_grad, w_physics=args.w_physics, near_weight=args.near_weight),
            tc,
            data=pairs,
            callbacks=[cosine(args.epochs, args.lr)] if args.cosine else [],
        )
        key = f"distill {size} {name}"
        results[key] = {**evaluate(field, teacher, q_s, q_g), "train_s": time.monotonic() - t0}
        print(key, results[key], flush=True)
        (args.out / f"results_{name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
