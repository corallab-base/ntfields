"""Browse a CNTFields UR5e cup checkpoint in viser: the benchmark's query pairs, both planners, and where each path
leaves the manifold or hits an obstacle.

uv run python benchmarks/cntfields/view_cup.py --ckpt runs/cntfields/<stamp>/ckpt.pt
"""

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser
from torch import Tensor
from ur5e_cup import PLANNER, query_pairs
from viser import transforms, uplot

import ntfields
from ntfields.envs import ArmEnv
from ntfields.envs.constraint import ConstrainedEnv
from ntfields.envs.tsr import TSRConstraint
from ntfields.eval.metrics import path_length, plan_succeeded
from ntfields.field import TrainedField
from ntfields.plan import gradient_bidirectional
from ntfields.viz.geometry import soup
from ntfields.viz.robot import ArmView, path_points

log = logging.getLogger(__name__)

PLANNERS = {"full gradient": False, "tangent": True}
FILTERS = ("failed", "succeeded", "all")
SCENE, ROBOT, START, GOAL = (112, 170, 120), (205, 80, 70), (60, 120, 220), (240, 160, 40)
VALID, INVALID = (40, 180, 90), (220, 50, 50)
FPS = 30


@dataclass
class Outcome:
    """One planner on one pair, on the CPU."""

    path: Tensor  # (N, dof)
    violation: Tensor  # (N,) d_M
    clearance: Tensor  # (N,) d_c, inf without obstacles
    met: bool
    success: bool
    reason: str


def failure_reason(met: bool, success: bool, off_manifold: bool, collides: bool) -> str:
    """Why `plan_succeeded` said what it did, in the terms of the README's results."""
    if success:
        return "success"
    reasons = [] if met else ["fronts never met"]
    reasons += ["left the manifold"] if off_manifold else []
    reasons += ["hit an obstacle"] if collides else []
    return ", ".join(reasons) or "invalid between waypoints"


def plan_pair(field: TrainedField, start: Tensor, goal: Tensor, tangent: bool) -> Outcome:
    env = field.problem.env
    assert isinstance(env, ConstrainedEnv)
    result = gradient_bidirectional(field, start, goal, tangent=tangent, **PLANNER)
    violation, clearance = env.constraint.violation(result.path), env.base.distance(result.path)
    success = plan_succeeded(field.problem, result)
    reason = failure_reason(
        result.success, success, bool((violation >= env.tolerance).any()), bool((clearance <= 0).any())
    )
    return Outcome(result.path.cpu(), violation.cpu(), clearance.cpu(), result.success, success, reason)


def cup_pose(constraint: TSRConstraint, q: Tensor) -> np.ndarray:
    """(4, 4) pose of the held cup in the arm's base frame, its z axis up when the cup is upright."""
    pose = constraint.arm.end_pose(q[None].to(constraint.T_w_e)) @ torch.inverse(constraint.T_w_e)
    return pose[0].detach().cpu().numpy().astype(np.float64)


class Viewer:
    def __init__(self, field: TrainedField, starts: Tensor, goals: Tensor, port: int) -> None:
        env = field.problem.env
        assert isinstance(env, ConstrainedEnv)
        assert isinstance(env.base, ArmEnv) and isinstance(env.constraint, TSRConstraint)
        self.field, self.env, self.arm, self.constraint = field, env, env.base, env.constraint
        self.starts, self.goals = starts.cpu(), goals.cpu()
        self.outcomes = {
            name: [plan_pair(field, s, g, tangent) for s, g in zip(starts, goals, strict=True)]
            for name, tangent in PLANNERS.items()
        }
        self.lock = threading.RLock()
        self.current: Outcome | None = None
        self.visible: list[int] = []
        self.server = viser.ViserServer(port=port)
        self._scene()
        self._gui()
        self.refresh()

    def _scene(self) -> None:
        scene = self.server.scene
        scene.set_up_direction("+z")
        self.server.initial_camera.position = (-1.3, -1.5, 1.1)  # as `ntfields viz` frames an arm
        self.server.initial_camera.look_at = (0.4, 0.0, 0.0)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        if self.arm.solids is None:  # the table stands in for the ground
            scene.add_grid("/ground", 2.0, 2.0, cell_size=0.1, section_size=0.5)
        else:
            vertices, faces = soup(self.arm.solids.triangles().numpy())
            scene.add_mesh_simple("/scene", vertices, faces, color=SCENE, opacity=0.55, side="double")
        self.robot = ArmView(scene, "/robot", self.arm, ROBOT, 1.0)
        self.start_view = ArmView(scene, "/start", self.arm, START, 0.3)
        self.goal_view = ArmView(scene, "/goal", self.arm, GOAL, 0.3)
        self.cup = scene.add_frame("/cup", axes_length=0.12, axes_radius=0.006)
        self.line: viser.LineSegmentsHandle | None = None

    def _gui(self) -> None:
        gui = self.server.gui
        counts = {n: sum(o.success for o in out) for n, out in self.outcomes.items()}
        gui.add_markdown("  \n".join(f"**{n}**: {c}/{len(self.starts)}" for n, c in counts.items()))
        self.g_planner = gui.add_dropdown("Planner", list(PLANNERS))
        self.g_filter = gui.add_dropdown("Show", FILTERS)
        self.g_pair = gui.add_slider("Pair", 0, 1, 1, 0)
        self.g_style = gui.add_dropdown("Draw as", ["meshes", "collision spheres"])
        self.g_ghosts = gui.add_checkbox("Start and goal poses", True)
        self.g_time = gui.add_slider("Along path", 0.0, 1.0, 0.001, 0.0)
        self.g_play = gui.add_checkbox("Play", False)
        self.g_result = gui.add_markdown("")
        self.g_now = gui.add_markdown("")
        series: tuple[uplot.Series, ...] = (
            {"label": "waypoint"},
            {"label": "d_M", "stroke": "#dc3232", "width": 2},
            {"label": "δ", "stroke": "#888888", "dash": (6.0, 4.0)},
            {"label": "d_c", "stroke": "#2878dc", "width": 2},
        )
        empty = tuple(np.zeros(1) for _ in series)
        self.g_plot = gui.add_uplot(empty, series, title="Along the path", aspect=1.6, scales={"x": {"time": False}})

        self.g_planner.on_update(lambda _: self.refresh())
        self.g_filter.on_update(lambda _: self.refresh())
        self.g_pair.on_update(lambda _: self.select())
        self.g_style.on_update(lambda _: self.style())
        self.g_ghosts.on_update(lambda _: self.style())
        self.g_time.on_update(lambda _: self.scrub())

    def refresh(self) -> None:
        """Apply the planner and the filter."""
        with self.lock:
            keep = {"failed": False, "succeeded": True}.get(self.g_filter.value)
            outcomes = self.outcomes[self.g_planner.value]
            self.visible = [i for i, o in enumerate(outcomes) if keep is None or o.success == keep]
            self.g_pair.max = max(1, len(self.visible) - 1)
            self.g_pair.value = min(self.g_pair.value, self.g_pair.max)
            self.select()

    def select(self) -> None:
        """Show the chosen pair."""
        with self.lock:
            if self.line is not None:
                self.line.remove()
                self.line = None
            if not self.visible:
                self.current = None
                self.g_result.content = "No pairs to show."
                return
            i = self.visible[min(self.g_pair.value, len(self.visible) - 1)]
            self.current = outcome = self.outcomes[self.g_planner.value][i]
            self.start_view.set(self.starts[i])
            self.goal_view.set(self.goals[i])
            self.line = self._draw_path(outcome)
            self.g_result.content = (
                f"Pair {i}: **{outcome.reason}**  \n"
                f"{len(outcome.path)} waypoints, length {path_length(outcome.path):.3f}, "
                f"largest d_M {outcome.violation.max():.3f}, smallest d_c {outcome.clearance.min():.3f}"
            )
            n = len(outcome.path)
            clearance = outcome.clearance.nan_to_num(posinf=0.0).numpy() if self.arm.solids is not None else np.zeros(n)
            self.g_plot.data = (
                np.arange(n, dtype=np.float64), outcome.violation.numpy(), np.full(n, self.env.tolerance), clearance,
            )  # fmt: skip
            self.style()
            self.scrub()

    def _draw_path(self, outcome: Outcome) -> viser.LineSegmentsHandle:
        """The tool's path, red where a waypoint is invalid."""
        points = path_points(self.arm, outcome.path)
        valid = ((outcome.violation < self.env.tolerance) & (outcome.clearance > 0)).numpy()
        colors = np.where(valid[:, None], np.array(VALID), np.array(INVALID)).astype(np.uint8)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        pairs = np.stack([colors[:-1], colors[1:]], axis=1)
        return self.server.scene.add_line_segments("/path", segments, pairs, thickness=0.006)

    def style(self) -> None:
        with self.lock:
            meshes = self.g_style.value == "meshes"
            self.robot.set_style(meshes)
            self.start_view.set_style(meshes, self.g_ghosts.value)
            self.goal_view.set_style(meshes, self.g_ghosts.value)

    def scrub(self) -> None:
        """Pose the arm and the cup at the time slider."""
        with self.lock:
            outcome = self.current
            if outcome is None:
                return
            k = round(self.g_time.value * (len(outcome.path) - 1))
            q = outcome.path[k]
            self.robot.set(q)
            pose = cup_pose(self.constraint, q)
            self.cup.wxyz, self.cup.position = transforms.SO3.from_matrix(pose[:3, :3]).wxyz, pose[:3, 3]
            d = self.constraint.displacement(q[None].to(self.constraint.T_w_e))[0]
            self.g_now.content = (
                f"Waypoint {k}: d_M {outcome.violation[k]:.3f}, d_c {outcome.clearance[k]:.3f}, "
                f"roll {torch.rad2deg(d[3]):.1f}°, pitch {torch.rad2deg(d[4]):.1f}°"
            )

    def spin(self) -> None:
        while True:
            time.sleep(1 / FPS)
            if self.g_play.value:
                self.g_time.value = (self.g_time.value + 0.25 / FPS) % 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    field = ntfields.load(args.ckpt, args.device)
    starts, goals = query_pairs(field.problem, args.pairs, seed=2024, device=args.device)
    log.info("planning %d pairs with both planners", len(starts))
    viewer = Viewer(field, starts, goals, args.port)
    log.info("open http://localhost:%d", args.port)
    viewer.spin()


if __name__ == "__main__":
    main()
