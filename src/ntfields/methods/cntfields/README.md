# CNTFields

Ni & Qureshi, *Physics-informed Neural Motion Planning on Constraint Manifolds*, ICRA 2024
([arXiv:2403.05765](https://arxiv.org/abs/2403.05765)).

The reference code is `ntfields-old` (corallab-base, a 2025 lab reimplementation) at commit `6481350`:
- the network is `ntfields/network/cntfield_network.py::CNTFieldNetwork`
- the loss, travel time and gradient are `ntfields/model/cntfield_model.py::CNTFields`
- the training recipe is `ntfields/model/base_model.py::NTFields.train`, with `configs/cntfields_hold_cup.yaml`
- the data is `ntfields/dataprocessing/sample_tsr.py`

That code covers a Franka holding a cup, without obstacles. This port targets the paper's task, a UR5e moving a cup,
and adds the paper's obstacle term and tangential planner. The paper publishes no hyperparameters, so those come from
the reference.

| Name | What it is |
|---|---|
| `cntfields` | the reference recipe |
| `cntfields_3d` | `cntfields` with the α schedule compressed 5×, for 1000-epoch runs on low-dimensional manifolds |

## What is in the method, and what is in the problem

CNTFields' network and loss are numerically P-NTFields', with other sizes. They live in `ntfields.nn` and
`ntfields.losses` and are shared. What makes the method constrained belongs to the problem:

- **`ConstrainedEnv(base, constraint, epsilon, tolerance)`** gives `distance(q) = ε − d(q)`, with the paper's
  `d(q) = max(d_M, ε − d_c)`. That equals `min(ε − d_M, d_c)`, so larger is better, as for every environment. With no
  obstacles `d_c` is inf and only `d_M` remains, as in the reference. `is_valid` also needs `d_M < tolerance`, the
  paper's δ.
- **`speed.GaussianBand(epsilon, lam, eased)`**: `S* = exp(−d²/(λε²))`, then `S*²(2 − S*)²` when eased. ε appears in
  both, so `ntfields.envs.tsr.ur5e_cup_problem()` builds them together.
- **`Constraint`** is the manifold: `violation`, `sample` and `tangent`. `TSRConstraint` is a task space region and
  `SphereConstraint` a test manifold with exact geodesics.
- **`data.ManifoldPairs`**: independent near-manifold starts and goals.

## Model and loss

- **Model.** `T = ‖q_g − q_s‖ / τ`, `FactorizedTauModel` with `hidden=256`:
  - Fourier features with B = N(0, 1) truncated to ±2, 128 × dim
  - `Linear(256, 256)`, two residual blocks, `Linear(256, 256)`, smoothed `[max, min]` (scale 10)
  - three residual blocks at width 512, `Linear(512, 256)`, `Linear(256, 1)`, `sigmoid(0.1·y)`; softplus(β = 10)
- **Loss.** As P-NTFields: `mean[x_s + 1/x_s + x_g + 1/x_g − 4]`, `x = √(S_visc / S*_α)`, with γ = 0.002.
- **Progressive speed.** `S*_α = (1 − α) + α·S*`, `α = min(0.5 + 10⁻⁴·epoch, 0.95)` (`LinearAlpha`).
- **Training.** AdamW lr 1e-3, weight decay 0.01, constant lr, `LossRatioRollback(1.2)`, no loss scaling.

Reference recipe settings that live outside the method config:
- the first 5 batches of 2000 per epoch (`TrainConfig(batch_size=2000, batches_per_epoch=5)`)
- 5000 epochs, `n_pairs` = 10⁶
- `GaussianBand(epsilon=0.7, lam=0.1)`

The planner is `gradient_bidirectional`. The reference's settings are `beta=0.01, goal_tol=0.03, max_steps=101`; pass
`tangent=True` for the paper's step.

## The UR5e cup task

`ntfields.envs.tsr.ur5e_cup_problem(table=False)`, or env presets `ur5e/cup` and `ur5e/cup_table`:

- The cup stays upright (roll = pitch = 0), with the tool 0.15 m behind it (the reference's grasp, offset for the
  2F-85). Samples widen roll and pitch by ±0.25 rad and come from the analytic IK.
- The region is x ∈ [0.3, 0.7], y ∈ [−0.4, 0.4], z ∈ [0.15, 0.65] m, yaw ±60°, on one connected sheet of IK solutions
  (branches 0 and 2, θ6 < 0). Self-colliding samples are dropped.
- `table=True` adds the `hold_cup.npz` table and boxes, which the reference ships but never reads, lowered 0.4 m so
  that the table top is level with the arm's base. It is the only obstacle geometry the reference has.
- `ur5e_cup_env` sets `clearance=0.05` m. With Eq. 4 as written, the reference's ε = 0.7 also becomes the obstacle
  width, but the fixed shoulder keeps d_c below 0.15 m, so S* never exceeds 2e-5 and there is nothing to learn. The
  paper does not publish ε. A width of 0.05 m gives S* a median of 0.92 on the manifold.

Why this region and not the reference's box around the base:

- A pose has about 7 valid IK branches, about 6 rad apart. They do not connect within the band, so a travel time
  between them means nothing. Hence one sheet.
- One analytic branch is not one sheet. A branch fixes the sign of θ5, and a horizontal tool crosses the wrist
  singularity θ5 = 0 inside this region, so branch 0 alone jumps by π in θ4 and θ6 there and splits into two main
  pieces (θ6 ≈ ±π/2, about 79% and 21% of samples). Branches 0 and 2 are the two sides of that singularity. With θ6 < 0
  they join into one component, with one solution per pose, and cover 86% of poses, the same as branch 0 did.
- A box around the base cuts the manifold at the ±π seam of the shoulder pan (`q = ±0.5`), and free yaw lets the tool
  point back through the arm: about 30% of samples self-collide. On this sheet the pan and wrist 1 stay clear of the
  seam, and about 9% of samples on branch 0 self-collide before they are dropped.

## Deviations from the reference code

| Detail | Reference | ntfields | Effect |
|---|---|---|---|
| robot and task | Franka (7-DOF), cuRobo IK | UR5e + 2F-85 (6-DOF), analytic IK | fidelity covers the network, loss, recipe and planner; the data is new |
| TSR region | box around the base, free yaw | see above | connected manifold, few self-collisions |
| IK branches | one numeric solution per pose | one connected sheet: branches 0 and 2, θ6 < 0 | see above |
| self-collision | unchecked | samples dropped | no training pairs in self-collision |
| obstacles | none (`TSRObsSampler` is a stub) | the paper's Eq. 4 when the base env has obstacles | none without obstacles; the held cup is not collision-checked |
| obstacle width | none: no obstacle term | `clearance=0.05` scales d_c by ε / 0.05; `ConstrainedEnv(clearance=None)` is Eq. 4 as written | see above; no effect without obstacles |
| planner step | full ∇T | full ∇T by default; `tangent=True` for the paper's ∇⊥T | the paper does not say how to compute ∇⊥T: we project out the Jacobian of the signed roll and pitch |
| unmet fronts | bridged with a straight line (`interpolate=True`) | the plan fails | no false successes |
| loss scale | `β` is computed after `backward()`, so it has no effect | none | none |
| unused layer | `encoder1[0]` is never called | dropped | none: it only consumed RNG at init |
| first rollback check | previous loss 9999 | 1000 | none in practice |
| rollback retries | unlimited | `LossRatioRollback` accepts a finite loss after 200, and stops after 200 non-finite ones | the reference loops forever on a persistent NaN; see Results |

Deviations from the paper's text follow the reference, as everywhere in ntfields: the speed is eased, unless
`GaussianBand(eased=False)`.

## Fidelity

`tests/fidelity/test_cntfields.py`, against a checkout at `../ntfields-old` or `$NTFIELDS_CNTFIELDS`:

| Level | Test | Result |
|---|---|---|
| 1 | τ, ∇τ, Δτ against `out_laplace`; forward pass; T against `travel_time` (float64) | match |
| 2 | loss and every parameter gradient against `CNTFields.loss` at α = 0.5 and 0.95 | match |
| 3 | 12 epochs against a copy of `NTFields.train` | match; with forced rollbacks, within 1e-6 (float64 rounding amplified by training) |
| planner | path against `predict_trajectory(interpolate=False)` | match |

`tests/unit/test_cntfields.py` checks `TSRConstraint.violation` against a copy of the reference's `distance`, and
`GaussianBand` against its speed and easing lines.

## Results

UR5e, 5000 epochs, 1e6 pairs, reference recipe, on an RTX 4070. Each row is 100 fixed on-manifold pairs with
`benchmarks/cntfields/ur5e_cup.py`. Success means the fronts met along a path that stays within δ = 0.1 of the manifold
and clear of obstacles. Time, length and margin are over the successes, so they are not comparable across rows with
very different success rates.

| Scene | Planner | Fronts met | Success | Time | Length | Mean margin | Max margin |
|---|---|---|---|---|---|---|---|
| `cup` | full gradient | 100% | 98% | 0.027 s | 0.267 | 0.0136 | 0.0946 |
| `cup` | tangent | 100% | 99% | 0.131 s | 0.267 | 0.0016 | 0.0838 |
| `cup_table` | full gradient | 100% | 7% | 0.012 s | 0.144 | 0.0294 | 0.0917 |
| `cup_table` | tangent | 92% | 78% | 0.208 s | 0.443 | 0.0008 | 0.0805 |

The paper reports 0.14 s and 92% for its UR5e move-cup task. That is context, not a target: its scene, gripper and
hyperparameters differ.

- `cup_table`: all 100 pairs are in one free component of the sampled manifold, so the failures are the field's and the
  planner's, not blocked queries. Full gradient: all 93 failures breach δ, and 2 of them also touch an obstacle.
  Tangent: of 22 failures, 8 have fronts that never met, 16 breach δ and 2 touch an obstacle, and these overlap. About
  26% of on-manifold samples collide with this scene.
- Sphere (`cntfields_3d`, a run stopped by NaN at epoch 708 of 1000, before the rollback change below): full gradient
  100% at 0.920 of the great circle, margin 0.0487, so it cuts through the sphere; tangent 99% at 1.004, margin 0.0015.

`benchmarks/cntfields/view_cup.py --ckpt <ckpt>` browses these pairs in viser with both planners, the arm along each
path and d_M and d_c along it. It needs the `[viz]` extra.

**Training stability.** Rollback stalls that reach the 200-retry cap are common on `cup`: 47 in 2h07m. There were none
on `cup_table`. A NaN comes from a single sample whose Δτ is large and negative (−2190 in one replay) making
1/S + γ·Δτ non-positive, so √S_visc is undefined: one sample in 4000 is enough. The reference has the same formula, and
its unlimited retries would hide this. `LossRatioRollback` used to stop if a NaN landed on the last retry of any stall;
it now stops only after 200 non-finite retries in one stall.
