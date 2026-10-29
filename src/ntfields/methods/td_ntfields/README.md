# TD-NTFields

Ni & Qureshi, *Physics-informed Temporal Difference Metric Learning for Robot Motion Planning*.

The reference code is [ruiqini/ntrl-demo](https://github.com/ruiqini/ntrl-demo) at commit `7a4aaf2`:
- the network is `models/metric*/model_network_metric.py::NN.out`
- the loss is `model_function_metric.py::Function.Loss`
- the training recipe is `model_train_metric.py`
- the planner is `tests/arm_plan_stat.py::MPPI`

ntfields matches ntrl-demo to within float32 rounding (see [Fidelity](#fidelity)). Its own operation order is fixed,
so a seed reproduces a run bit for bit on the same PyTorch build.

| Name | What it is |
|---|---|
| `td_ntfields` | generic reference (`models/metric`) |
| `td_ntfields_arm` | UR5e, ntrl-demo's recipe fixed for clean labels: `models/metric_arm` at the width of the published UR5 run, `UR5eSurfacePairs` (IK) pairs, gradient clipping 5.0, no rollback |
| `td_ntfields_arm_ntrl` | UR5e, ntrl-demo's recipe (rollback, `NearSurface` pairs) |
| `td_ntfields_2d` | a recipe for point masses in [−1, 1]²: the arm model settings and `SurfaceMix` pairs |
| `td_ntfields_stable` | `td_ntfields` with the causality weight detached |

## Model and loss

- **Model.** `T = out_scale · Σ_groups smoothmax_16 √((e(q_s) − e(q_g))² + 1e-6)`. The embedding `e` is built from:
  1. Fourier features
  2. Lipschitz-normalized sine layers in PirateNet-style gated blocks
  3. a per-row normalization
- **Loss.** `mean[(eikonal + TD + normal) · exp(−λ_C·T)]`, summed over the start and goal sides:
  - eikonal: `λ_E (√(S*·‖∇T‖) − 1)²`
  - TD: `λ_TD (T − T(q − Δt·S*·∇T) − Δt/S*)²`, which is zero where `T < Δt/S*`
  - normal: `λ_N (1.01 − S*) ‖S*·∇T + n‖²`, where `n` is the unit clearance normal
- **Training.**
  - AdamW with `lr` 5e-4 and weight decay 0.5
  - `LossRatioRollback(1.2)`
  - `LossScaleByPrevLoss(4)`: β = 4 / Σ batch losses of the last accepted epoch, the reference's 1/(Σ/4)
  - `NearSurface` pair sampling

Reference recipe settings that live outside the method config: 5 batches of 2000 per epoch
(`TrainConfig(batch_size=2000, batches_per_epoch=5)`) and 5000 epochs. The problem's speed should be
`speed.ClippedEased(dmin=offset, dmax=margin)`.

## Deviations from the reference code

| Detail | Reference | ntfields | Effect |
|---|---|---|---|
| `td_ntfields_arm` width | repo code `h_size=512`, B 256×dim | 256 and 128×dim, as in the run that produced the published UR5 checkpoint | none: `hidden=512` reproduces the repo code |
| unused layer | `encoder[0] = Linear(252, h)` exists but is never called | dropped | none: it only consumed RNG at init |
| residual gate | `Linear(1, 1)`, only `sigmoid(0.1·weight)` used | one scalar parameter per block | none |
| epoch loss | 5-batch sum / 4 (logged, used for β) | mean over batches (logged); β = 4 / (5·mean) (`loss_scale_numerator`) reproduces the reference's β | logged values are 4/5 of the reference's |
| learning rate | set to 1e-3, then reset to 5e-4 at every epoch | constant 5e-4 | none |
| causality weight | `exp(−0.5·T)` not detached | same by default; `detach_causality=True` (the `_stable` preset) | not detaching has collapsed training (`T(q,q)` ≈ 36) |
| operation order | `x (2πB)ᵀ`; start and goal embedded in one stacked call; each gated layer evaluated once; one `autograd.grad` over `(B, 2·dim)` | `2π·(x Bᵀ)`; separate start/goal embeddings; `lin1` and the sines evaluated twice; one `autograd.grad` per side | rounding only (T within 4e-7 of the reference) |

## Fidelity

Run `NTFIELDS_NTRL_DEMO=../ntrl-demo uv run pytest tests/fidelity`. ntrl-demo's code is imported unmodified.

| Level | Test | Tolerance |
|---|---|---|
| 1 | forward pass against ntrl-demo, with its trained UR5 checkpoint and with random weights | float64 default |
| 2 | `loss_n` and every parameter gradient against ntrl-demo's `Function.Loss`, on ntrl's UR5 data | float64 default |
| 3 | 5 epochs (25 AdamW steps, rollback and β included) against the ntrl-demo training loop | losses float32 default; weights 1e-4 relative |
| planner | `mppi` path against `arm_plan_stat.py::MPPI` from the same seed and weights (CUDA only, since the reference is) | float32 default |

Bit-for-bit reproduction holds on CPU with the same thread count (BLAS reductions change with the thread split) and
on GPU with the same PyTorch/CUDA build, because different builds pick different kernels. Training amplifies any
rounding difference: nudging the initial weights by 1e-7 gives a 5e-5 relative
weight drift after 5 epochs, while training moves the weights by 6e-2.

## Point-mass results (`box2d/one_sphere`, 1000 epochs)

Scored by one script: mean |T − T_FMM| over 4 goals × 500 starts, and collision-free plans out of 4 queries per
planner.

| Run | T error | MPPI | bidirectional |
|---|---|---|---|
| `td_ntfields_2d`, seeds 0/1/2 | 0.029 / 0.036 / 0.032 | 12/12 | 11/12 |
| `td_ntfields` (generic reference settings) | 0.548 | 0/4 | 3/4 |

The generic settings (Fourier scale 1.0, output scale 0.2, `NearSurface` starts only within 0.1 of an obstacle)
are tuned for ntrl-demo's own scaled scenes. With them, pairs far from the obstacle are never trained, and long-range
T comes out too small.

## UR5e arm (`ur5e/ntrl_scene`, `td_ntfields_arm`)

`td_ntfields_arm` fixes several problems in ntrl-demo's arm setup:
- **Corrupted labels.** ntrl's shipped labels are partly corrupted by a chunk-offset bug. `ArmEnv` computes its own.
- **Unstable recipe.** ntrl's recipe is unstable on clean labels: use gradient clipping (5.0) instead of rollback.
- **Pair sampling.** IK pairs need the corrected orientation centre (−π/2, 0, +π/2).
- **No collision check.** ntrl's evaluation never checked collisions.

`eval.ur5e.benchmark_ur5e` scores 5000-epoch models with MPPI seeds 2024 / 1 / 2. The checks are ntrl-demo's 5
scenarios and 50 random collision-free IK pairs. A plan succeeds if MPPI converges and no waypoint penetrates the
scene.

| Model | Scenarios | Random pairs | Random converged |
|---|---|---|---|
| ntrl-demo's checkpoint (epoch 3900) | 2 / 2 / 2 | 39 / 38 / 35 | 49 / 49 / 49 |
| ntfields `td_ntfields_arm`, seed 0 | 4 / 4 / 4 | 36 / 35 / 34 | 48 / 48 / 48 |
| ntfields `td_ntfields_arm`, seed 1 | 4 / 3 / 3 | 38 / 39 / 37 | 47 / 47 / 46 |

Single runs vary a lot with the seed: across training runs, random-pair successes ranged from 34 to 43. The rows above
were trained before `ntfields.train` fixed its order of random draws, so the same seeds now give different runs.
