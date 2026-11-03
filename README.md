# ntfields

[![ruff](https://github.com/corallab-base/ntfields/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/ruff.yml)
[![pyright](https://github.com/corallab-base/ntfields/actions/workflows/pyright.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/pyright.yml)
[![python 3.11](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.11.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.11.yml)
[![python 3.12](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.12.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.12.yml)
[![python 3.13](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.13.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.13.yml)
[![python 3.14](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.14.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.14.yml)

Physics-informed neural travel-time fields (the NTFields lineage) with a shared problem definition, trainer,
planners and benchmark.

Implemented methods:
- `euclidean`: a sanity baseline
- `td_ntfields`: TD-NTFields, matched against the reference code (see
  [its README](src/ntfields/methods/td_ntfields/README.md))
- `pntfields`: P-NTFields, matched against the reference code (see
  [its README](src/ntfields/methods/pntfields/README.md))

## Install

```bash
uv sync                    # dev environment (all extras)
pip install "ntfields[cli,gt]" # as a library: CLI (tyro) and FMM ground truth (scikit-fmm)
pip install "ntfields[tensorrt]"  # TensorRT backend and engine export (Python 3.12+)
```

## TD-NTFields on a 2D point mass

One sphere of radius 0.3 in [−1, 1]², with the `td_ntfields_2d` preset.
1000 epochs reach a mean travel-time error of about 0.03 against FMM.

```python
import torch

import ntfields
from ntfields.data import sample_valid
from ntfields.envs import load_env
from ntfields.eval import is_collision_free, path_cost, travel_time_error
from ntfields.gt import fmm_travel_time
from ntfields.speed import ClippedEased

problem = ntfields.Problem(load_env("box2d/one_sphere"), ClippedEased(dmin=0.01, dmax=0.1))
train_cfg = ntfields.TrainConfig(n_pairs=1_000_000, batch_size=2000, batches_per_epoch=5)  # the reference recipe
field = ntfields.train("td_ntfields_2d", problem, ntfields.Epochs(1000), train_cfg=train_cfg)  # ~2 min on a GPU

device = train_cfg.device
start, goal = torch.tensor([-0.6, 0.0], device=device), torch.tensor([0.6, 0.0], device=device)
result = field.plan(start, goal)  # MPPI, TD-NTFields' default planner
path = result.path.cpu()
print(result.success, is_collision_free(problem, path), path_cost(problem, path))

gt = fmm_travel_time(problem, goal.cpu())  # ground truth (needs the [gt] extra)
print(travel_time_error(field, gt, sample_valid(problem, 500).to(device)))

field = ntfields.load(field.ckpt_path, device=device)  # reload later from the checkpoint
```

For many queries against one goal, `field.towards(goal)` encodes the goal once and gives `.travel_time(q)` and
`.gradient(q)` for batches q; MPPI plans this way.

The same run from the CLI (needs the `[cli]` extra). `--method.*` sets any field of the method's config:

```bash
ntfields train td_ntfields_2d --env box2d/one_sphere --speed.kind clipped_eased --epochs 1000 \
    --train.n-pairs 1000000 --train.batch-size 2000 --train.batches-per-epoch 5
```

## TD-NTFields on the UR5e arm

The UR5e among ntrl-demo's scanned cabinet (`ur5e/ntrl_scene`, needs the `[arm]` extra). Joints are scaled as
q = θ / 2π. `td_ntfields_arm` is ntrl-demo's recipe, fixed:
- IK-sampled near-surface pairs, which `ntfields.train` samples itself
- gradient clipping instead of rollback

```python
import torch

import ntfields
from ntfields.envs import load_env
from ntfields.envs.arm import ntrl_scenarios
from ntfields.eval.ur5e import benchmark_ur5e
from ntfields.speed import ClippedEased

problem = ntfields.Problem(load_env("ur5e/ntrl_scene"), ClippedEased(dmin=0.005, dmax=0.05))
train_cfg = ntfields.TrainConfig(n_pairs=500_000, batch_size=2000, batches_per_epoch=5)
field = ntfields.train("td_ntfields_arm", problem, ntfields.Epochs(5000), train_cfg=train_cfg)  # ~12 min on a GPU

device = train_cfg.device
starts, goals = ntrl_scenarios()
result = field.plan(starts[0].to(device), goals[0].to(device))  # MPPI in joint space

# ntrl-demo's 5 scenarios and 50 random pairs, 3 MPPI seeds, collision-checked
print(benchmark_ur5e(field))
```

To train on a fixed dataset instead of sampling, pass `data=` (a `PairDataset`, e.g. `PairDataset.load("pairs.h5")`)
to `ntfields.train`. The training loop underneath, `ntfields.train.fit`, takes any `PairSource`: an object with
`.batches(batch_size, scene, generator)`, asked once per epoch, so it can draw new pairs (or scenes) every epoch.

### Faster evaluation

`field.accelerate()` evaluates the field through its model's inference form: weights folded into plain matmuls,
gradients in forward mode, compiled with torch.compile and replayed from CUDA graphs. It takes the field's place in
the planners and benchmarks (TD-NTFields and P-NTFields):

```python
fast = field.accelerate()  # "compile" (a few seconds per batch size on first use), or "eager"
result = fast.plan(starts[0].to(device), goals[0].to(device))
print(benchmark_ur5e(fast))
```

On an RTX 3090, MPPI on the UR5e benchmark drops from 41 to 9 ms per plan (scenarios) and from 110 to 20 ms (random
pairs), with the same success counts.

With the `[tensorrt]` extra (Python 3.12+), `field.accelerate("tensorrt", precision="fp16")` builds a TensorRT engine
per function on its first call; fp16 applies to the travel times only (TD-NTFields), the goal encoding and gradients
stay fp32. In Python it plans about as fast as `"compile"`; its main use is exporting engines for other runtimes:

```bash
ntfields export --ckpt runs/.../ckpt.pt --out engines/ur5e --precision fp32 fp16
```

This writes one `<function>_<precision>.engine` per inference function (`encode_goal`, `travel_time`,
`pair_travel_time`, `gradient`, `pair_gradient`; `--functions` picks some), reference inputs and outputs of the
original model to check another runtime against, and a `meta.json` with every engine's binding names and shapes.
Engines load only with the TensorRT version they were built with, which `meta.json` records.

To build other arm scenes, use `ArmEnv.ur5e(points=..., spheres=...)`, which accepts point-cloud and sphere
obstacles.

## Custom training loops

`ntfields.train` wraps `ntfields.train.fit(method, model, data, scene, budget, batch_size, callbacks)`, whose
callbacks hook in at train start, epoch start, after each batch's backward pass (before clipping and the step), after
each step, and at epoch and train end. Besides the rollback, loss-scale, logging and checkpoint callbacks,
`LRScheduler` steps a torch scheduler per batch or per epoch (a rollback restores it with the model and the
optimizer, through `TrainState.stateful`), `Validation` evaluates the method's loss on held-out pairs, and
`Checkpoint("ckpt_{epoch}.pt", ...)` keeps one file per save.

```python
import torch

from ntfields.train import Checkpoint, Epochs, LRScheduler, Validation, fit

sched = LRScheduler(lambda opt: torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=1000 * 5), "batch")
callbacks = [sched, Validation(method, val_pairs, scene, batch_size=2000), Checkpoint(path, method, problem, 100)]
fit(method, model, train_pairs, scene, Epochs(1000), 2000, callbacks, batches_per_epoch=5)
```

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest
NTFIELDS_NTRL_DEMO=../ntrl-demo uv run pytest tests/fidelity  # against the reference code
```
