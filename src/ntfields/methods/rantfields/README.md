# RaNTFields

Randers neural travel-time fields: travel time for dynamics with a drift and an anisotropic control bound,

    q̇ = f(q) + u,   uᵀ G(q) u ≤ 1,   cost ∫ dt / S*(q),

whose HJB equation is the Randers eikonal equation √(∇Tᵀ G⁻¹ ∇T) = ⟨∇T, f⟩ + 1/S*. A gravity-loaded arm (G the
inertia, f the gravity drift) and a point mass on a slope or in a current are instances.

The reference code is the RaNTFields research repo (`nap`) at commit `d4424eb`:
- the network is `src/nap/models/nap_net_utils.py::FinslerMetricWithResidual`
- the loss is `src/nap/models/loss.py::randers_5_loss`
- the training recipe is `src/nap/scripts/train.py`
- the planner is `src/nap/scripts/test.py::extract_path_gradient_descent_continuous` (`use_riemannian_control`)
- the controller is `src/nap/sim/controller/geodesic_qp.py::GeodesicTauQPController` (gradient direction)

Only what the paper uses is here. ntfields matches the reference to float32 rounding (see [Fidelity](#fidelity)).

| Name | What it is |
|---|---|
| `rantfields`, `rantfields_2d` | the paper's 2D recipe (point mass in [−1, 1]² under a flow or on a terrain) |
| `rantfields_arm` | the paper's 5-joint UR5e recipe; train with `Epochs(100)` (its reported checkpoint, on a 400-epoch LR cycle) |

## The problem

Drift and metric belong to the problem, as S\* does: `Problem(env, speed, metric)`, with a `Metric` from
`ntfields.metric`. Every method samples the same G and f (`sample_pairs` stores them as `Batch.metric_*` and
`Batch.drift_*`), and the planner and controller read them online.

| Paper setting | env | speed | metric |
|---|---|---|---|
| 2D flows | `box2d/one_sphere`, `box2d/empty`, `box2d/small_sphere` | `ExpRamp.for_env(env)` | `FlowField("constant" / "circular" / "saddle")` |
| 2D terrains | `box2d/two_spheres`, `box2d/three_small_spheres` | `ExpRamp.for_env(env)` | `Terrain("smooth_ridge", height=0.4)`, `Terrain("single_hill")` |
| UR5e, 5 joints | `ur5e/5j_sphere` | `ExpRamp(5 × 0.2803711)` | `ArmDynamics(lambda_reg=0.1, …)` (the `[dynamics]` extra) |

The exact paper problems, with the constants the reference derived from its datasets, are in
[benchmarks/rantfields/problems.py](../../../../benchmarks/rantfields/problems.py). The table's baselines are the same
method on other problems: N-Riem. is `ArmDynamics(with_drift=False)`, N-Eik. has no metric.

```python
import torch

import ntfields
from ntfields.envs import load_env
from ntfields.metric import FlowField
from ntfields.speed import ExpRamp

env = load_env("box2d/one_sphere")
problem = ntfields.Problem(env, ExpRamp.for_env(env), FlowField("constant", vector=(0.8, 0.0)))
train_cfg = ntfields.TrainConfig(n_pairs=18_000, batch_size=512, batches_per_epoch=32)  # the paper's recipe
field = ntfields.train("rantfields_2d", problem, ntfields.Epochs(400), train_cfg=train_cfg)
result = field.plan(torch.tensor([-0.8, -0.3]), torch.tensor([0.8, 0.3]))  # riemannian_gradient
```

## Model and loss

- **Model.** `T(q_s → q_g) = α·d_IQE(e_g, e_s)·m + (1 − α)·r`, with α = sigmoid(a) one learned scalar.
  - One tied encoder `e = LayerNorm(Linear(tanh(Linear(q))))` (64, then 32 units) embeds the goal, the start and
    their difference.
  - `m` and `r` are softplus-positive residual networks on `[e_diff, e_g, e_s]`.
  - `d_IQE` is the interval quasimetric embedding (IQE-maxmean, 16 dims per component; vendored in `iqe.py`). It is
    zero at q_s = q_g by construction and asymmetric.
- **Loss.** `mean[(L_R + 5·T(q_g, q_g)²) / (1 + T)]`, with `L_R = (√(∇Tᵀ G⁻¹ ∇T) − ⟨∇T, κ·f⟩ − 1/S*)²` at the start
  (the departure point). The boundary term is the batch mean. κ = `flow_scale`.
- **Training.**
  - AdamW with weight decay 1e-3, and OneCycleLR stepped every batch (pct_start 0.1, max_lr 1e-3; arm 3e-3).
  - The arm uses the reference's rollback: when loss / previous loss ≥ 1.2, a random state from the starts of the
    last 5 accepted epochs is restored (with the LR schedule), at most twice per epoch.
  - `SurfaceMix` pairs in 2D: half uniform; half with a goal at clearance U(0, 0.1) and a start offset by up to
    0.5 per coordinate.
  - `JointSpaceMix` pairs on the arm: half uniform; half with a goal at clearance 0.01-0.06 m and a start up to π
    rad away in a random joint-space direction.

## Planning and control

- **`riemannian_gradient`** (the default planner) integrates the optimal control u\* = −G⁻¹∇T / √(∇Tᵀ G⁻¹ ∇T).
  - It moves along f + u\*, blended with the straight line to the goal by λ = βT/(1 + βT), β = 0.1.
  - It takes steps of 0.01, shrinking on direction reversals.
- **`ntfields.control.GeodesicTauController`** is the paper's online controller.
  - Every 10 steps it takes the direction of f + u\* at the current state and blends it with the goal direction
    (λ = 0.5 in the paper's table).
  - The result is q̇_des at the cruise speed, slowing down within 0.2 of the goal.
  - Every step it applies τ = ID(q, q̇, (q̇_des − q̇)/τ_track) through any `Dynamics`: `PinocchioDynamics`,
    `PointMassDynamics`, or the MuJoCo one in `benchmarks/rantfields/sim`.

## Deviations from the reference code

| Detail | Reference | ntfields | Effect |
|---|---|---|---|
| online drift for the arm's `flow_via_anisotropy` model | the controller recomputed f as G·b in joint radians, without the training-time scaling to q, global rescale or clip: 0.24-0.79× the trained drift, at up to ~80° from it | the trained drift, everywhere | the published table used the reference's drift; `benchmarks/rantfields/sim/arm_table.py --reference-drift` reproduces it |
| flow projector | built and initialized, never read by this head | dropped | none on outputs; it shifts the RNG at init, so a seed gives different initial weights |
| T(q, q) = 0 | through the loss only | same; `check_method(..., soft_boundary=True)` | none |
| flow and G along a planned path | nearest neighbour among the dataset's points (planner) | the problem's exact `Metric` | paths within 1e-4 of the reference's under a constant drift, 4e-4 on the terrain |
| `use_adaptive_loss_scale` (arm run) | set, but `randers_5` never receives the scale | not implemented | none |
| arm coordinates | q = θ/θ_max per joint (elbow θ/π, others θ/2π); G applied unscaled to ∇_q T | same (`ArmEnv.ur5e_spherized`) | other ntfields arm presets use θ/2π |
| validation split | 10% held out | none: train on all sampled pairs | the paper trained on 90% of its 20k (2D) and 1M (arm) pairs |

## Fidelity

Checked against the reference's own paper checkpoints and datasets:
- **Physics:**
  - flows, terrain G and drift, and S\*: 6e-7;
  - arm clearance: 2.4e-7;
  - arm G and drift (training-time, `via_anisotropy`): 5e-7, with m_ref and the drift calibration bit-identical;
  - the reference's online drift: 8e-8.
- **Forward and loss**, on paper checkpoints (constant flow `17-23-39`, ridge `09-29-47`, arm `02-52-19`):
  - T matches to 1e-6;
  - the loss matches to 1e-7 relative;
  - every parameter gradient matches to 3e-5 relative.
- **Planner:** on paper checkpoints, every success and collision outcome is identical; paths agree to 1e-4 under a
  constant drift and to 4e-4 on the terrain, where the reference's nearest-neighbour drift lookup differs.

The scripts that load reference checkpoints are in `benchmarks/rantfields/` (`nap_import.py`).
- **Closed loop:** `benchmarks/rantfields/sim/arm_table.py --reference-drift` reproduces the reference's 300 MuJoCo
  runs behind the paper's table:
  - same success on every pair: RaNTF / N-Eik. / N-Riem. 96 / 98 / 99 %;
  - the same 93 common pairs;
  - identical means over them: time 10.827 / 10.264 / 10.508 s, traced length 12.065 / 11.474 / 11.775 rad, peak
    torque 74.20 / 90.22 / 78.05 N·m;
  - per-pair time to goal within 0.02 s.

  Queries run at about 500 Hz on one CPU (the reference: about 240 Hz).

## Closed-loop results (5-joint UR5e, 100 detour pairs)

`arm_table.py` without `--reference-drift`: RaNTF driven by the drift it was trained with. Means are over the pairs
every model completes collision-free (91 with the trained drift, 93 with the reference's).

| Model | success % | time to goal (s) | traced length (rad) | peak torque (N·m) |
|---|---|---|---|---|
| RaNTF, trained drift | 92 | 11.35 | 12.47 | 75.0 |
| RaNTF, reference's online drift (published) | 96 | 10.83 | 12.07 | 74.2 |
| N-Eik. | 98 | 10.39 | 11.61 | 90.2 |
| N-Riem. | 99 | 10.62 | 11.90 | 77.9 |

Over the 91 pairs, N-Eik. and N-Riem. are the same runs as in the published table. Peak torque stays lowest for
RaNTF with either drift. Success and time to goal are worse with the trained drift.

