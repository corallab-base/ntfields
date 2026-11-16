"""Distill a box2d-v1 TD-NTFields benchmark checkpoint into smaller students, against students of the same size
trained from scratch with the method's own loss, on the teacher's recipe (1000 epochs of 5 batches of 2000).

Reports the travel-time error against FMM and against the teacher, and MPPI success on random valid pairs.

    uv run python benchmarks/distill/box2d.py --scene clutter
"""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import tyro

import ntfields
from ntfields import DistillConfig, Epochs, TrainConfig, TrainedField
from ntfields.data import sample_valid
from ntfields.eval import plan_succeeded, travel_time_error
from ntfields.gt import fmm_travel_time


@dataclass
class Args:
    scene: str = "clutter"
    seed: int = 0
    sizes: tuple[str, ...] = ("128x2", "128x1", "64x1", "32x1")  # hidden x n_blocks
    epochs: int = 1000
    scratch: bool = True  # also train each size from scratch
    w_grad: float = 0.1
    w_physics: float = 0.0
    weight_decay: float | None = None  # None: the teacher's (AdamW)
    n_goals: int = 5
    n_starts: int = 500
    n_plans: int = 100
    out: Path = Path("runs/distill")


def evaluate(
    field: TrainedField, teacher: TrainedField, goals: torch.Tensor, starts: torch.Tensor, n_plans: int
) -> dict[str, float]:
    problem, dev = field.problem, field.device
    gt_err = [travel_time_error(field, fmm_travel_time(problem, g), starts.to(dev)) for g in goals]
    q_s, q_g = starts.to(dev), goals[torch.arange(len(starts)) % len(goals)].to(dev)
    teacher_err = float((field.travel_time(q_s, q_g) - teacher.travel_time(q_s, q_g)).abs().mean())
    g_s, _ = field.gradient(q_s, q_g)
    tg_s, _ = teacher.gradient(q_s, q_g)
    grad_cos = float(torch.nn.functional.cosine_similarity(g_s, tg_s, dim=-1).mean())
    gen = torch.Generator().manual_seed(123)
    pairs = sample_valid(problem, 2 * n_plans, gen).view(2, n_plans, -1)
    ok = sum(plan_succeeded(problem, field.plan(s.to(dev), g.to(dev))) for s, g in zip(*pairs, strict=True))
    return {
        "fmm_err": sum(gt_err) / len(gt_err),
        "teacher_err": teacher_err,
        "grad_cos": grad_cos,
        "mppi_success": ok / pairs.shape[1],
    }


def main(args: Args) -> None:
    ckpt = Path(f"benchmarks/checkpoints/box2d-v1/{args.scene}/td_ntfields_2d/seed{args.seed}.pt")
    teacher = ntfields.load(ckpt, device="cuda")
    gen = torch.Generator().manual_seed(0)
    goals = sample_valid(teacher.problem, args.n_goals, gen)
    starts = sample_valid(teacher.problem, args.n_starts, gen)
    tc = TrainConfig(n_pairs=1_000_000, batch_size=2000, batches_per_epoch=5, out_dir=args.out / args.scene)
    results: dict[str, dict[str, float]] = {"teacher 256x2": evaluate(teacher, teacher, goals, starts, args.n_plans)}
    print("teacher 256x2", results["teacher 256x2"], flush=True)
    for size in args.sizes:
        hidden, n_blocks = map(int, size.split("x"))
        cfg = replace(teacher.method.cfg, hidden=hidden, n_blocks=n_blocks)
        if args.weight_decay is not None:
            cfg = replace(cfg, weight_decay=args.weight_decay)
        runs = {"distill": None, "scratch": None} if args.scratch else {"distill": None}
        for kind in runs:
            t0 = time.monotonic()
            if kind == "distill":
                dcfg = DistillConfig(w_grad=args.w_grad, w_physics=args.w_physics)
                field = ntfields.distill(teacher, "td_ntfields", Epochs(args.epochs), cfg, dcfg, tc)
            else:
                field = ntfields.train("td_ntfields", teacher.problem, Epochs(args.epochs), cfg, tc)
            key = f"{kind} {size}"
            results[key] = {**evaluate(field, teacher, goals, starts, args.n_plans), "train_s": time.monotonic() - t0}
            print(key, results[key], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.scene}_seed{args.seed}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
