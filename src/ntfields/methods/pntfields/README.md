# P-NTFields

Ni & Qureshi, *Progressive Learning for Physics-informed Neural Motion Planning*, RSS 2023 (arXiv:2306.00616).

The reference code is [ruiqini/P-NTFields](https://github.com/ruiqini/P-NTFields) at commit `bd773a8`:
- the network is `models/model_res_sigmoid.py::NN` (`out`; `out_laplace` for the loss)
- the loss is `Model.Loss`
- the training recipe is `Model.train`
- the data is `dataprocessing/speed_sampling_gpu.py`
- the planners are `test/arm_plan.py` and `test/gib_plan.py`

| Name | What it is |
|---|---|
| `pntfields` | generic reference: the `test`/`c3d` data settings (margin 0.1, offset 0.01) |
| `pntfields_arm` | the reference arm recipe: margin 0.5/12, offset margin/10, goal step up to √3, 2·10⁶ pairs |
| `pntfields_2d` | `pntfields` with the α and lr schedules compressed 10×, for 1000-epoch point-mass runs |

## Model and loss

- **Model.** `T = ‖q_g − q_s‖ / τ`, with `τ = g(f(q_s) ⊗ f(q_g)) ∈ (0, 1)`:
  - `f`: Fourier features (B = 0.5·N(0, 1), 128 × dim), `Linear(256, 128)`, two residual blocks, `Linear(128, 128)`
  - `⊗`: `[smoothmax, smoothmin]` by logsumexp with scale 10
  - `g`: three residual blocks at width 256, `Linear(256, 128)`, `Linear(128, 1)`, then `sigmoid(0.1·y)`

  All activations are softplus(β = 10).
- **Loss.** `mean[x_s + 1/x_s + x_g + 1/x_g − 4]`, with `x = √(S_visc / S*_α)` at each end.
  - The model's speed is `S = 1/‖∇T‖ = τ² / √(‖d‖²‖∇τ‖² − 2τ ∇τ·d + τ²)`, where `d` points to the other end.
  - Viscosity enters as `S_visc = 1/(1/S + γ·Δτ)`, with γ = 0.001 and Δτ the Laplacian of τ at that end.
  - `PNTFieldsModel.tau_jet` gives ∇τ and Δτ in one forward-mode pass, like the reference's `out_laplace`. That is
    3.6× faster than double backward on the 6-DOF arm and uses 57% of the memory.
- **Progressive speed.** `S*_α = (1 − α) + α·S*`, with `AlphaSchedule`:
  - α = 0.5 until epoch 2000
  - then α rises by 1/8000 per epoch, slowing at epochs 4000 and 8000
  - α reaches 1.0 at ≈ epoch 8000 and 1.05 at ≈ epoch 9200
- **Training.**
  - AdamW with lr 1e-3 and weight decay 0.1; `LRRamp` takes the lr down to 5e-4 over epochs 8000–8500
  - `LossRatioRollback(1.2)`
  - `LossScaleByPrevLoss(numerator)`: β = numerator / Σ batch losses of the last accepted epoch. The reference uses
    `β = len(loader) / Σ₆`, so the numerator is the reference dataset's batch count: 200 for the arm, 50 for `test`.
  - `NearSurface` pairs: starts with `offset < d < margin`; goals a uniform distance up to `max_step` away in a
    random direction, inside the bounds, with their clearance unchecked

Reference recipe settings that live outside the method config:
- the first 6 batches of 10000 per epoch (`TrainConfig(batch_size=10000, batches_per_epoch=6)`)
- 10000 epochs
- `n_pairs` = 5·10⁵ (`test`) or 2·10⁶ (arm)
- the speed `speed.Clipped(dmin=offset, dmax=margin)`, in bounds [−0.5, 0.5]^dim

The default planner is `gradient_bidirectional` (`q ← q − β ∇T/‖∇T‖²` at both ends). The reference's Gibson settings
are its defaults (β 0.03, d_g 0.06). For the arm, the reference uses
`gradient_bidirectional(field, start, goal, beta=0.015, goal_tol=0.03, max_steps=301)`.

## Deviations from the reference code

| Detail | Reference | ntfields | Effect |
|---|---|---|---|
| unused layer | `encoder1[0] = Linear(256, 128)` exists but is never called | dropped | none: it only consumed RNG at init |
| epoch loss | Σ₆ batch losses / `len(loader)` (logged, used for β and the rollback ratio) | mean over batches (logged); `loss_scale_numerator` reproduces β | logged values are `len(loader)/6` × the reference's |
| first rollback check | previous loss starts at 1.0, in Σ₆/`len(loader)` units | starts at 1000, in mean units | none in practice: both accept any sane first epoch |
| rollback queue | `randint(0, 4)`, which fails while fewer than 5 states are queued | draws from the states queued so far | the reference crashes where ntfields rolls back |
| rollback retries | unlimited | `LossRatioRollback(max_retries=200)`, then the epoch is accepted with a warning | none unless training is stuck |
| lr on a retried epoch | restored from the rolled-back optimizer state (the previous epoch's lr) | the ramp's lr for this epoch | at most 5e-7 apart, only in epochs 8000–8500 |
| shuffling | the loader re-permutes its already-permuted tensors | a fresh permutation each epoch | statistically the same |
| operation order | `x @ (2πB)`; start and goal stacked through one encoder call | `2π·(x Bᵀ)`; separate encoder calls | rounding only |

## Fidelity

Run `uv run pytest tests/fidelity/test_pntfields.py`. Set `NTFIELDS_PNTFIELDS` if the checkout is not at
`../P-NTFields`. The reference code is imported unmodified, with `pickle5` stubbed by `pickle`. Every comparison is
in float64.

All levels pass against `bd773a8` at float64's default tolerance (2026-10-29). The tests that use the reference's
pretrained UR5 model (`Experiments/UR5/Model_Epoch_10000_*.pt`, from the README's Google Drive link) are skipped
when it is absent.

| Level | Test | Tolerance |
|---|---|---|
| 1 | τ, ∇τ and the Hessian diagonal against `NN.out_laplace`, plus T against `Model.TravelTimes`, with random weights and with the pretrained UR5 model | float64 default |
| 2 | `loss_n` and every parameter gradient against `Model.Loss`: random weights at α = 0.5, 0.81 and 1.05 (epochs 0, 5000 and 9500), and the UR5 model at α = 1.05 | float64 default |
| 3 | 5 epochs (30 AdamW steps, rollback and β included) against a copy of `Model.train` | float64 default |
| planner | the `gradient_bidirectional` path against `test/arm_plan.py`'s loop on `Model.Gradient`: random weights for 40 steps, and the UR5 model on `arm_plan.py`'s own query, where both converge | float64 default |

Without the reference, the unit tests check:
- `tau_jet` against autograd
- the loss's closed-form speed against `1/‖∇T‖` from autograd
- `AlphaSchedule` and `LRRamp` against copies of the reference loop

## Point-mass result (`box2d/one_sphere`, `pntfields_2d`, 1000 epochs)

The run used seed 0 with the reference recipe: 5·10⁵ pairs, 6 batches of 10000 per epoch, and speed
`Clipped(0.01, 0.1)`. It took 58 min on a shared RTX 4070. It was scored by mean |T − T_FMM| over 4 goals × 500
starts, and by collision-free `gradient_bidirectional` plans across the sphere for 4 queries.

| Run | T error | bidirectional plans |
|---|---|---|
| `pntfields_2d`, seed 0 | 0.094 | 4/4 |
| `euclidean` baseline | 0.033 | 0/4 |

The T error is not comparable with TD-NTFields' table, which used a different speed (`ClippedEased`) and a different
script. With this speed, the field is 1 everywhere except within 0.1 of the sphere, so Euclidean distance already
scores well on T. The 10× compressed schedule moves α 10× faster per epoch. At epoch 390 this hit
`LossRatioRollback`'s 200-retry cap, and the epoch was accepted at a loss ratio of 1.48.

## UR5 cabinet result (`pntfields_arm`, 10000 epochs)

`benchmarks/pntfields_ur5/` rebuilds the reference's UR5 cabinet scene from the P-NTFields checkout: link-vertex
clearance to `untitled_scaled.off`, its pair sampler with the wrist OBB prefilter, and 2·10⁶ pairs. `train.py` runs
the full reference recipe; `evaluate.py` scores models on held-out pairs and 201 planning queries (200 random plus
`arm_plan.py`'s own). Seed 0 took 3.7 h on a shared RTX 4070, with no rollbacks.

| Model | held-out loss (ref / any pairs) | speed rel. error start, goal (ref pairs) | plans succeeded | mean cost |
|---|---|---|---|---|
| published `Model_Epoch_10000` | 0.140 / 0.334 | 0.41, 0.29 | 168/201 | 0.958 |
| `pntfields_arm`, seed 0 | 0.125 / 0.321 | 0.39, 0.27 | 172/201 | 0.958 |

"ref" pairs come from the training distribution and "any" pairs are uniform in free space. A plan succeeds if
`gradient_bidirectional` (β 0.015, tol 0.03, 301 steps) converges with every densely checked configuration clear by
more than the offset. 165 queries are solved by both models, and both solve `arm_plan.py`'s query.
