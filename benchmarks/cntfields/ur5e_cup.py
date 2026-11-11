"""CNTFields on the UR5e cup task: train, then report the paper's metrics on 100 on-manifold pairs for both planners.

uv run python benchmarks/cntfields/ur5e_cup.py --scene cup
uv run python benchmarks/cntfields/ur5e_cup.py --scene cup_table --ckpt runs/cntfields/<stamp>/ckpt.pt
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch

import ntfields
from ntfields.envs.constraint import ConstrainedEnv
from ntfields.envs.tsr import TSRConstraint, ur5e_cup_problem
from ntfields.eval.constrained import evaluate_constrained
from ntfields.problem import Problem

PLANNER = {"beta": 0.01, "goal_tol": 0.03, "max_steps": 101}  # ntfields-old's predict_trajectory


def query_pairs(problem: Problem, n: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """n pairs on the manifold (no sampling offsets) with both ends valid."""
    env = problem.env
    assert isinstance(env, ConstrainedEnv) and isinstance(env.constraint, TSRConstraint)
    constraint = replace(env.constraint, offsets=torch.zeros(6))
    generator = torch.Generator().manual_seed(seed)
    q = constraint.sample(20 * n, generator, device).to(device)
    q = q[env.is_valid(q)]
    return q[:n], q[n : 2 * n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["cup", "cup_table"], default="cup")
    parser.add_argument("--ckpt", type=Path, help="evaluate this checkpoint instead of training")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--n-pairs", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt:
        field = ntfields.load(args.ckpt, device)
    else:
        train_cfg = ntfields.TrainConfig(n_pairs=args.n_pairs, batch_size=2000, batches_per_epoch=5, device=device)
        problem = ur5e_cup_problem(table=args.scene == "cup_table")
        field = ntfields.train("cntfields", problem, ntfields.Epochs(args.epochs), train_cfg=train_cfg)
        print(f"checkpoint: {field.ckpt_path}")

    starts, goals = query_pairs(field.problem, args.pairs, seed=2024, device=device)
    results = {
        "full_gradient": evaluate_constrained(field, starts, goals, **PLANNER),
        "tangent": evaluate_constrained(field, starts, goals, tangent=True, **PLANNER),
    }
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"scene": args.scene, "results": results}, indent=2))


if __name__ == "__main__":
    main()
