"""Reference side of the RaNTFields fidelity tests. Runs inside the reference repo's own environment (nap, with its
torchqmet/hydra stack), never imported by ntfields: `tests/fidelity/nap_repo.py` calls it as a subprocess.

    python nap_reference.py loss <run dir> <epoch> <out.pt>     T, the randers_5 loss and parameter gradients
    python nap_reference.py arm_fields <out.npz>               arm G and drift: training-time and online
    python nap_reference.py planner <run dir> <epoch> <pairs.pt> <out.pt>
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf


def _model_and_data(run: str, epoch: str):  # noqa: ANN202
    from nap.models.nap_net import NAPNet
    from nap.scripts.train_utils import setup_data

    cfg = OmegaConf.load(f"{run}/.hydra/config.yaml")
    m = cfg.model
    model = NAPNet(
        input_dim=cfg.dim,
        embed_dim=m.embed_dim,
        hidden_dim=m.hidden_dim,
        num_residual_blocks=m.num_residual_blocks,
        num_finsler_blocks=m.num_finsler_blocks,
        prediction_method=m.prediction_method,
        device="cpu",
        fourier_feature_config=m.fourier_feature_config,
    )
    sd = torch.load(f"{run}/checkpoints/model_epoch_{epoch}.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"])
    ds = setup_data(cfg)[0]
    return cfg, model, sd["model_state_dict"], getattr(ds, "dataset", ds)


def loss(run: str, epoch: str, out: str) -> None:
    from nap.models.loss import randers_5_loss

    cfg, model, sd, ds = _model_and_data(run, epoch)
    idx = torch.linspace(0, len(ds.start_flow) - 1, 256).long()
    start, goal = ds.start_configuration_normalized[idx], ds.goal_configuration_normalized[idx]
    speed, flow, start_flow = ds.goal_speed[idx], ds.goal_flow[idx], ds.start_flow[idx]
    G = ds.goal_anisotropy[idx] if ds.goal_anisotropy is not None else torch.eye(cfg.dim).expand(256, -1, -1)
    goal_pts = goal.unsqueeze(1).clone().requires_grad_(True)
    pred = model(start, goal_pts, flow.unsqueeze(1), start_flow)
    flow_scale = float(cfg.get("randers_flow_scale", 0.8))
    out_loss = randers_5_loss(
        model, pred, start, goal_pts, speed.unsqueeze(1), flow.unsqueeze(1), input_dim=cfg.dim,
        anisotropy_matrix=G.unsqueeze(1), start_flow=start_flow, randers_flow_scale=flow_scale,
    )  # fmt: skip
    out_loss["total"].backward()
    torch.save(
        {
            "state_dict": sd,
            "start": start,
            "goal": goal,
            "speed": speed,
            "flow": flow,
            "G": G,
            "T": pred.detach()[:, 0],
            "loss": {k: float(v) for k, v in out_loss.items()},
            "grads": {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None},
            "flow_scale": flow_scale,
        },
        out,
    )


def arm_fields(out: str) -> None:
    from nap.datasets.continuous_dataset_utils import compute_flow_speed_continuous

    _, _, _, ds = _model_and_data("outputs/2026-09-15/02-52-19", "100")
    idx = np.linspace(0, len(ds.goal_flow) - 1, 20).astype(int)
    q = ds.goal_configuration_normalized[idx].numpy()
    online, _ = compute_flow_speed_continuous(q, ds)
    np.savez(out, q=q, G=ds.goal_anisotropy[idx].numpy(), f=ds.goal_flow[idx].numpy(), f_online=online)


def planner(run: str, epoch: str, pairs: str, out: str) -> None:
    from nap.scripts.test import extract_path_gradient_descent_continuous

    _, model, _, ds = _model_and_data(run, epoch)
    p = torch.load(pairs)
    paths, ok = [], []
    for s, g in zip(p["starts"], p["goals"], strict=True):  # the reference descends from its goal_pos to its start_pos
        path, _, converged = extract_path_gradient_descent_continuous(
            model, ds, g[None], s[None], torch.zeros(1, len(s)), use_riemannian_control=True
        )
        paths.append(torch.tensor(np.stack(path)))
        ok.append(converged)
    torch.save({"paths": paths, "converged": ok}, out)


if __name__ == "__main__":
    {"loss": loss, "arm_fields": arm_fields, "planner": planner}[sys.argv[1]](*sys.argv[2:])
