"""Load a RaNTFields run of the reference repo (nap) as an ntfields `TrainedField`, for fidelity checks and to
re-evaluate the paper's checkpoints with ntfields' planners and controller. Reads the run's `.hydra/config.yaml` and
its dataset's `metadata.json`; needs pyyaml. Covers the paper's 2D point-mass runs and its three 5-joint UR5e runs."""

import json
from pathlib import Path

import torch
import yaml
from problems import arm_problem

from ntfields.envs import SphereBoxEnv
from ntfields.field import TrainedField
from ntfields.methods.rantfields import RaNTFields, RaNTFieldsConfig
from ntfields.metric import FlowField, Terrain
from ntfields.problem import Problem
from ntfields.speed import ExpRamp


def problem_from_metadata(meta: dict) -> Problem:
    spheres = [(o["center"], o["radius"]) for o in meta["obstacles"] if o["shape"] == "sphere"]
    boxes = [(o["center"], o["size"]) for o in meta["obstacles"] if o["shape"] == "box"]
    env = SphereBoxEnv.build(bounds=meta["bounds"], spheres=spheres, boxes=boxes)
    speed = ExpRamp.for_env(env, meta["min_speed"])
    flow, params = meta["flow_type"], meta["flow_params"]
    elevation = meta["elevation_type"]
    if elevation != "no_elevation":
        if flow != "no_flow":
            raise NotImplementedError("a flow on top of a terrain was not used in the paper")
        e = meta["elevation_params"]
        if elevation == "single_hill":
            metric = Terrain(
                "single_hill", e.get("max_elevation", 0.5), e.get("spread", 0.3), meta["max_feasible_norm_g_sq"]
            )
        else:
            metric = Terrain("smooth_ridge", e.get("max_elevation", 2.0), max_norm_g_sq=meta["max_feasible_norm_g_sq"])
    elif flow == "constant_flow":
        metric = FlowField("constant", vector=(params.get("flow_x", 0.0), params.get("flow_y", 0.0)))
    elif flow == "circular_flow":
        metric = FlowField("circular", magnitude=params.get("flow_magnitude", 1.0))
    elif flow == "saddle_flow":
        metric = FlowField("saddle", growth_rate=params.get("growth_rate", 1.0), magnitude=params["max_magnitude"])
    else:
        metric = None
    return Problem(env, speed, metric)


def load_nap_run(run: str | Path, epoch: int, device: str = "cpu") -> TrainedField:
    run = Path(run)
    cfg = yaml.safe_load((run / ".hydra/config.yaml").read_text())
    meta = json.loads((Path(cfg["dataset_file_path"]).parent / "metadata.json").read_text())
    if meta["generator_type"] == "robot_arm":
        model = "n_eik" if cfg.get("zero_anisotropy") else "n_riem" if cfg.get("zero_flow") else "rantf"
        if model == "rantf" and not cfg.get("flow_via_anisotropy"):
            raise NotImplementedError("only the paper's flow_via_anisotropy arm run is supported")
        problem = arm_problem(model)
    else:
        problem = problem_from_metadata(meta)
    m = cfg["model"]
    method = RaNTFields(
        RaNTFieldsConfig(
            embed=m["embed_dim"],
            hidden=m["hidden_dim"],
            n_residual_blocks=m["num_residual_blocks"],
            n_finsler_blocks=m["num_finsler_blocks"],
            flow_scale=cfg.get("randers_flow_scale", 0.8),
        )
    )
    model = method.build_model(problem)
    sd = torch.load(run / f"checkpoints/model_epoch_{epoch}.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(
        {k.removeprefix("predict_time."): v for k, v in sd["model_state_dict"].items() if "flow_proj" not in k}
    )
    return TrainedField(method, problem, model.to(device).eval())
