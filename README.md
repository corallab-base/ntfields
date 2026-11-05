# ntfields

[![ruff](https://github.com/corallab-base/ntfields/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/ruff.yml)
[![pyright](https://github.com/corallab-base/ntfields/actions/workflows/pyright.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/pyright.yml)
[![python 3.11](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.11.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.11.yml)
[![python 3.12](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.12.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.12.yml)
[![python 3.13](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.13.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.13.yml)
[![python 3.14](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.14.yml/badge.svg?branch=main)](https://github.com/corallab-base/ntfields/actions/workflows/tests-3.14.yml)

Physics-informed neural travel-time fields (the NTFields lineage) with a shared problem definition, trainer,
planners and benchmark.

Implemented methods (compared on shared suites by `ntfields bench`; see [Benchmark](#benchmark)):
- `euclidean`: a sanity baseline
- `td_ntfields`: TD-NTFields, matched against the reference code (see
  [its README](src/ntfields/methods/td_ntfields/README.md))
- `pntfields`: P-NTFields, matched against the reference code (see
  [its README](src/ntfields/methods/pntfields/README.md))

## Install

```bash
uv sync                    # dev environment (all extras)
pip install "ntfields[cli,gt]" # as a library: CLI (tyro) and FMM ground truth (scikit-fmm)
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
to `ntfields.train`.

To build other arm scenes, use `ArmEnv.ur5e(points=..., spheres=...)`, which accepts point-cloud and sphere
obstacles.

## Benchmark

`ntfields bench` evaluates any registered method, including one from another package plugged in through the
`ntfields.methods` entry point, on frozen, versioned suites. Every entry is scored by the same evaluator, so results
from different people are directly comparable.

```bash
ntfields bench list                                    # released suites
ntfields bench run --suite box2d-v1 --entries my_method --seeds 0 1 2 --out results/my_method.jsonl
ntfields bench report --records results/*.jsonl        # Markdown tables, mean ± std over seeds
ntfields bench validate --records results/my_method.jsonl
```

**Suites** live in [benchmarks/suites/](benchmarks/suites/). A suite release (`<name>-v<k>`) holds its scenes, a
`dev` and a `test` query set, reference costs, and the evaluation protocol, all pinned by checksums in `suite.json`.
A released suite never changes; any change is a new version, and results only compare within one version. Tune on
`dev`; report `test`. A method may train on anything derived from the scene and the speed S\*, never on test queries.

| Suite | Problems | Queries (dev / test) | Reference |
|---|---|---|---|
| `box2d-v1` | 2D point mass in [−1, 1]²: `empty`, `one_sphere`, `clutter`, `walls`, `thick_walls`; S\* = `Clipped(0.01, 0.1)` | 50 / 200 each | FMM |
| `box3d-v1` | 3D point mass in [−1, 1]³: `empty`, `four_spheres`, `clutter`, `window` (a 0.3-thick plate with one hole); S\* = `Clipped(0.01, 0.1)` | 50 / 200 each | FMM |
| `smoke-v1` | `one_sphere` only, for CI (not ranked) | 4 / 4 | FMM |

**Protocol.** A configuration is valid if it is inside the bounds with clearance above the suite's margin (0.01 in
2D). A plan succeeds if its path starts at the start, ends at the goal, and stays valid when checked on segments of at
most the suite's resolution. The planner's own convergence flag is recorded but not used.

| Metric | Meaning |
|---|---|
| success % | share of test queries solved |
| SPL | mean over all test queries of c\*/max(c, c\*) for a success and 0 for a failure, where c is the path's travel time ∫ds/S\* and c\* the reference's. 1 means every query solved optimally |
| cost ratio | mean c/c\* over solved queries (1 = optimal); it ignores failures |
| T error % | mean \|T − T\*\|/T\* of the field's travel time against the FMM travel time |
| plan ms, train min | wall time per query, and training time, with the hardware in each record |

References (c\*, T\*) come from fast marching with obstacles made impassable, so they exist in 2D and 3D only;
higher-dimensional suites report success, cost, length and times. `report` notes any query where an entry beats c\*
by more than the suite's tolerance, which would mean the reference needs a new suite version.

S\* is not zero inside obstacles (`Clipped` gives dmin/dmax = 0.1 there), while the reference treats obstacles as
impassable. In `walls` (0.1 thick), crossing a wall under S\* is cheaper than the detour for 37% of test queries, so
a field that faithfully fits S\* plans through the wall and fails them. `walls` keeps this on purpose, as a test of
how methods handle thin obstacles. `thick_walls` (0.3 thick) has the same layout, and the detour is the optimum
under S\* for every test query.

**Entries** are (method or preset, planner). Each method trains with its own reference recipe
(`ntfields.bench.RECIPES`; register yours with `register_recipe`), and training cost is reported alongside the
results. Planners come from `ntfields.plan`; external ones register through the `ntfields.planners` entry point.
`--planners all` also runs every method with every planner. Trained checkpoints go to `benchmarks/checkpoints/`
(not committed); rerunning with the same config and recipe evaluates them instead of retraining.

### Challenge pairs
`challenge %` is success on pairs whose straight line is blocked, so a query that needs no planning cannot flatter it.
`box2d-v1` and `box3d-v1` hold 200 such pairs per problem, beside `suite.json` so that their released pairs and records
stay valid; `empty` has none. Every test pair of `ur5e-mbm-v1` needs a detour, so there it equals the success rate.
`ntfields bench challenge --records <files>` adds the metric to existing records from their checkpoints.

### Viewer
`ntfields viz --suite ur5e-mbm-v1` (needs `ntfields[viz]`) opens a browser UI on port 8080: scenes as true geometry,
oracle boxes or point clouds; the arm as collision spheres or meshes; tested pairs with their recorded results; online
planning from a checkpoint, judged as the benchmark judges it; custom queries; and a T heatmap for 2D and 3D scenes.

### Results: `box2d-v1`

SPL on the test split, mean over seeds 0–2, with each method's reference recipe. The last three columns are over
all five problems. Per-problem std, cost ratios and plan times are in the
[full report](benchmarks/results/box2d-v1/README.md), and the records in
[benchmarks/results/box2d-v1/](benchmarks/results/box2d-v1/).

| Entry | empty | one_sphere | clutter | walls | thick_walls | **all** | success % | T error % | train min |
|---|---|---|---|---|---|---|---|---|---|
| `td_ntfields_2d` + gradient | 0.999 | 0.972 | 0.815 | **0.981** | **0.977** | **0.949** | **96.9** | **4.0** | **2.8** |
| `td_ntfields_2d` + MPPI | 0.997 | 0.983 | **0.845** | 0.848 | 0.766 | 0.888 | 90.1 | **4.0** | **2.8** |
| `pntfields_2d` + gradient | **1.000** | **0.995** | 0.807 | 0.630 | 0.874 | 0.861 | 88.0 | 7.9 | 18.7 |
| `pntfields_2d` + MPPI | 0.998 | 0.992 | 0.821 | 0.627 | 0.755 | 0.839 | 85.6 | 7.9 | 18.7 |
| `euclidean` + gradient | **1.000** | 0.604 | 0.189 | 0.185 | 0.195 | 0.435 | 44.7 | 18.8 | 0.0 |

The planners are `gradient_bidirectional` and `mppi`. Times are on a shared RTX 4070.

- **`pntfields_2d` on `walls`:** it fails 36% of queries because it fits S\* faithfully and so plans through the thin
  walls (see the note above). On `thick_walls` it solves 90.5%.
- **`pntfields_2d` on `clutter`:** it diverged on seeds 0 and 2, where the loss turned NaN at epochs 822 and 715.
  Training stopped at the last accepted state.

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest
NTFIELDS_NTRL_DEMO=../ntrl-demo uv run pytest tests/fidelity  # against the reference code
```
