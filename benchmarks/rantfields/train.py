"""Train RaNTFields on one of the paper's problems with the paper's recipe, then report collision-free success of the
default planner on random valid pairs.

    uv run python benchmarks/rantfields/train.py --problem flow_constant     # ~10 min on a GPU
    uv run python benchmarks/rantfields/train.py --problem arm   # needs [arm,dynamics]; then sim/arm_table.py

The reference trained on 90% of its datasets (20k pairs in 2D, 1M on the arm); these recipes sample that many.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from problems import PROBLEMS_2D, arm_problem

import ntfields
from ntfields.data import sample_valid
from ntfields.eval import is_collision_free

RECIPES = {  # TrainConfig fields and epochs, per problem family
    "2d": ({"n_pairs": 18_000, "batch_size": 512, "batches_per_epoch": 32}, 400),
    "arm": ({"n_pairs": 900_000, "batch_size": 512, "batches_per_epoch": 1000}, 100),
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=[*PROBLEMS_2D, "arm", "arm_n_riem", "arm_n_eik"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="override the recipe")
    p.add_argument("--eval-pairs", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/checkpoints/rantfields"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.problem.startswith("arm"):
        problem = arm_problem({"arm": "rantf", "arm_n_riem": "n_riem", "arm_n_eik": "n_eik"}[args.problem])
        method, (fields, epochs) = "rantfields_arm", RECIPES["arm"]
    else:
        problem = PROBLEMS_2D[args.problem]()
        method, (fields, epochs) = "rantfields_2d", RECIPES["2d"]
    train_cfg = ntfields.TrainConfig(**fields, seed=args.seed, out_dir=args.out_dir / args.problem)
    field = ntfields.train(method, problem, ntfields.Epochs(args.epochs or epochs), train_cfg=train_cfg)
    print(f"checkpoint: {field.ckpt_path}")

    g = torch.Generator().manual_seed(1000 + args.seed)
    starts, goals = sample_valid(problem, args.eval_pairs, g), sample_valid(problem, args.eval_pairs, g)
    device = train_cfg.device
    ok = 0
    for s, e in zip(starts, goals, strict=True):
        result = field.plan(s.to(device), e.to(device))
        ok += bool(result.success) and bool(is_collision_free(problem, result.path.cpu()))
    print(
        f"{args.problem}: {ok}/{args.eval_pairs} random pairs planned collision-free ({field.method.default_planner})"
    )


if __name__ == "__main__":
    main()
