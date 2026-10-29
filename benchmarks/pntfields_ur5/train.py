"""Train `pntfields_arm` on the reference's UR5 cabinet recipe.

    uv run python benchmarks/pntfields_ur5/train.py WORK SEED

2·10⁶ pairs from the reference's sampler, the first 6 batches of 10000 per epoch (as 2 micro-batches of 5000 to fit
in memory; same gradients), 10000 epochs. Writes WORK/seed<SEED>/state_<epoch>.pt every 500 epochs and log.jsonl.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch

from ntfields import Epochs, TrainState
from ntfields.methods.pntfields import PNTFields, PNTFieldsConfig
from ntfields.train import Callback, fit

sys.path.insert(0, str(Path(__file__).parent))
from common import cached_pairs, problem
from scene import ReferenceArmPairs

DEVICE = "cuda"


class Record(Callback):
    def __init__(self, out: Path, every: int = 500) -> None:
        self.out = out
        self.every = every
        self.start = time.monotonic()
        self.retries = 0

    def on_epoch_end(self, state: TrainState) -> None:
        if state.retry:
            self.retries += 1
            return
        e = state.epoch + 1
        if e % 50 == 0 or e == 1:
            row = {"epoch": e, "loss": state.loss, **state.terms, "retries": self.retries,
                   "minutes": (time.monotonic() - self.start) / 60}  # fmt: skip
            with (self.out / "log.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if e % self.every == 0:
            torch.save(state.model.state_dict(), self.out / f"state_{e:05d}.pt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    work, seed = Path(sys.argv[1]), int(sys.argv[2])
    out = work / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    prob = problem(work, DEVICE)
    data = cached_pairs(prob, 2_000_000, 1000 + seed, ReferenceArmPairs(), work / f"train_pairs_{seed}.h5")

    method = PNTFields(PNTFieldsConfig.arm())
    torch.manual_seed(seed)  # as ntfields.train: a data permutation, the init, then every shuffle
    data = data[torch.randperm(len(data))].to(DEVICE)
    model = method.build_model(prob).to(DEVICE)
    callbacks = [*method.callbacks(), Record(out)]
    scene = prob.env.condition()
    fit(method, model, data, scene, Epochs(10_000), 10_000, callbacks, torch.default_generator, 6, 5_000)
    torch.save(model.state_dict(), out / "state_final.pt")


if __name__ == "__main__":
    main()
