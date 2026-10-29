"""Score P-NTFields UR5 models on the same held-out pairs and planning queries.

    uv run python benchmarks/pntfields_ur5/evaluate.py WORK [NAME=STATE_DICT ...]

The published reference model is always included. Metrics:
- loss: the training objective at α = 1.05 on 20000 held-out pairs, from the reference's sampler (`ref_*`, with its
  wrist box filter) and from plain `NearSurface` (`any_*`, near-surface starts anywhere)
- speed error: mean |S - S*| / S* at each end of those pairs, with S = 1/‖∇T‖ the model's own speed
- plans: `gradient_bidirectional` with the reference's arm settings (β 0.015, d_g 0.03, 301 steps) on test/arm_plan.py's
  query and 200 random queries whose ends have clearance > 0.02. A plan succeeds if it converges and its
  densely interpolated path keeps clearance > OFFSET (the reference's own clearance: unsigned, at the links'
  collision-mesh vertices); cost is ∫ ds / S* along the path.
"""

import json
import sys
from pathlib import Path

import torch

from ntfields import Batch, SceneCondition, TrainedField, TrainState
from ntfields.methods.pntfields import PNTFields, PNTFieldsConfig
from ntfields.methods.pntfields.loss import _ratio_loss
from ntfields.methods.pntfields.model import PNTFieldsModel
from ntfields.plan import gradient_bidirectional

sys.path.insert(0, str(Path(__file__).parent))
from common import cached_pairs, ntfields_model, problem, reference_model, straight
from scene import OFFSET, ReferenceArmPairs, query

DEVICE = "cuda"


def held_out(model: PNTFieldsModel, data: Batch) -> dict[str, float]:
    cfg = PNTFieldsConfig.arm()
    state = TrainState(model=model, optimizer=torch.optim.SGD(model.parameters()), epoch=10_000)
    loss, err_s, err_g = [], [], []
    for i in range(0, len(data.q_s), 2000):
        part = Batch(*(t[i : i + 2000] for t in (data.q_s, data.q_g, data.speed_s, data.speed_g, data.normal_s,
                                                 data.normal_g)), scene=SceneCondition())  # fmt: skip
        per_pair = _per_pair_loss(model, part, state, cfg)
        loss.append(per_pair)
        q_s, q_g = part.q_s.clone().requires_grad_(True), part.q_g.clone().requires_grad_(True)
        g_s, g_g = torch.autograd.grad(model(q_s, q_g, SceneCondition()).sum(), (q_s, q_g))
        err_s.append((1 / g_s.norm(dim=-1) - part.speed_s).abs() / part.speed_s)
        err_g.append((1 / g_g.norm(dim=-1) - part.speed_g).abs() / part.speed_g)
    loss_all = torch.cat(loss)
    return {
        "loss": loss_all.nanmean().item(),
        "loss_nan_frac": loss_all.isnan().float().mean().item(),
        "speed_rel_err_start": torch.cat(err_s).mean().item(),
        "speed_rel_err_goal": torch.cat(err_g).mean().item(),
    }


def _per_pair_loss(model: PNTFieldsModel, part: Batch, state: TrainState, cfg: PNTFieldsConfig) -> torch.Tensor:
    """The training loss per pair (`pntfields_loss` is its batch mean), so pairs with a NaN can be counted."""
    tau, dtau, hess = model.tau_jet(part.q_s, part.q_g)
    d = part.q_g - part.q_s
    alpha = cfg.alpha(state.epoch)
    start = _ratio_loss(tau, dtau[:, :6], hess[:, :6].sum(-1), -d, part.speed_s, alpha, cfg.viscosity)
    goal = _ratio_loss(tau, dtau[:, 6:], hess[:, 6:].sum(-1), d, part.speed_g, alpha, cfg.viscosity)
    return (start + goal).detach()


def plan_queries(prob, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    g = torch.Generator().manual_seed(seed)
    starts, goals = [], []
    while len(starts) < n:
        q = torch.rand(4 * n, 6, generator=g) - 0.5
        d = prob.env.distance(q.to(DEVICE)).cpu()
        q = q[d > 0.02]
        starts += list(q[: len(q) // 2])
        goals += list(q[len(q) // 2 : 2 * (len(q) // 2)])
    s, g_ = query()
    return torch.stack([s, *starts[:n]]), torch.stack([g_, *goals[:n]])


def plans(field: TrainedField, starts: torch.Tensor, goals: torch.Tensor) -> dict[str, float]:
    prob = field.problem
    converged, success, costs, lengths, min_clear = 0, [], [], [], []
    for s, g in zip(starts.to(DEVICE), goals.to(DEVICE), strict=True):
        r = gradient_bidirectional(field, s, g, beta=0.015, goal_tol=0.03, max_steps=301)
        dense = torch.cat([straight(a, b, 5e-4)[:-1] for a, b in zip(r.path[:-1], r.path[1:], strict=True)])
        with torch.no_grad():
            d = torch.cat([prob.env.distance(c) for c in dense.split(8192)])
            seg = r.path.diff(dim=0)
            cost = (seg.norm(dim=-1) / prob.speed_at(r.path[:-1] + seg / 2)).sum().item()
        ok = r.success and bool(d.min() > OFFSET)
        converged += r.success
        success.append(ok)
        min_clear.append(d.min().item())
        if ok:
            costs.append(cost)
            lengths.append(seg.norm(dim=-1).sum().item())
    return {
        "query_success": success[0],
        "converged": converged,
        "success": sum(success),
        "n": len(success),
        "mean_cost": sum(costs) / max(len(costs), 1),
        "mean_length": sum(lengths) / max(len(lengths), 1),
        "success_mask": success,
    }


def main() -> None:
    work = Path(sys.argv[1])
    prob = problem(work, DEVICE)
    held = {}
    for tag, sampler in (("ref", ReferenceArmPairs()), ("any", PNTFieldsConfig.arm().sampler)):
        data = cached_pairs(prob, 20_000, 12345, sampler, work / f"held_out_{tag}.h5").to(DEVICE)
        held[tag] = Batch(
            data.q_s, data.q_g, data.speed_s, data.speed_g, data.normal_s, data.normal_g, SceneCondition()
        )
    starts, goals = plan_queries(prob, 200, seed=7)
    models = {"reference": reference_model()}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        models[name] = ntfields_model(Path(path))
    results = {}
    for name, model in models.items():
        model = model.to(DEVICE).eval()
        field = TrainedField(PNTFields(PNTFieldsConfig.arm()), prob, model)
        metrics = {f"{tag}_{k}": v for tag, b in held.items() for k, v in held_out(model, b).items()}
        results[name] = {**metrics, **plans(field, starts, goals)}
        shown = {k: v for k, v in results[name].items() if k != "success_mask"}
        print(name, json.dumps(shown), flush=True)
    masks = [r["success_mask"] for r in results.values()]
    print("solved by all:", sum(all(m) for m in zip(*masks, strict=True)))
    (work / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
