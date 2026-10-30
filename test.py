import logging
import os
import sys
import time

import torch

import ntfields
from ntfields.data import sample_valid
from ntfields.envs import load_env
from ntfields.eval import is_collision_free, path_cost, travel_time_error
from ntfields.gt import fmm_travel_time
from ntfields.speed import ClippedEased

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")  # the RX 9060 XT only (set before the first GPU call)
logging.basicConfig(level=logging.INFO, format="%(message)s")  # shows ntfields' "epoch N loss=..." lines


def stage(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if not torch.cuda.is_available():
    sys.exit(f"torch {torch.__version__} can't see a GPU. Run: bash setup_rocm.sh, then .venv/bin/python test.py")
stage(f"torch {torch.__version__} (HIP {torch.version.hip}) on {torch.cuda.get_device_name(0)}")

problem = ntfields.Problem(load_env("box2d/one_sphere"), ClippedEased(dmin=0.01, dmax=0.1))
train_cfg = ntfields.TrainConfig(n_pairs=1_000_000, batch_size=2000, batches_per_epoch=5)  # the reference recipe

stage("sampling 1M pairs, then training td_ntfields_2d for 1000 epochs (a loss line every 100 epochs)")
field = ntfields.train("td_ntfields_2d", problem, ntfields.Epochs(1000), train_cfg=train_cfg)
stage(f"trained; checkpoint at {field.ckpt_path}")

device = train_cfg.device
start, goal = torch.tensor([-0.6, 0.0], device=device), torch.tensor([0.6, 0.0], device=device)
stage("planning with MPPI")
result = field.plan(start, goal)
path = result.path.cpu()
stage(f"plan: success={result.success} collision_free={is_collision_free(problem, path)} cost={path_cost(problem, path)}")

stage("ground truth with scikit-fmm")
gt = fmm_travel_time(problem, goal.cpu())
err = travel_time_error(field, gt, sample_valid(problem, 500).to(device))
stage(f"mean |T - T_FMM| = {err} (the README gets about 0.03)")

field = ntfields.load(field.ckpt_path, device=device)
stage("reloaded from the checkpoint: done")
