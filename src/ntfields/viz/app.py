"""A viser viewer for suites, recorded results and trained fields.

ntfields viz [--port 8080] [--suite ur5e-mbm-v1]
"""

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import viser
from torch import Tensor

from ntfields.bench.run import CHALLENGE_SEED_OFFSET
from ntfields.bench.schema import Record
from ntfields.bench.suite import DEFAULT_ROOT, Suite, SuiteProblem, load_suite
from ntfields.checkpoint import load
from ntfields.envs import ArmEnv
from ntfields.field import TrainedField
from ntfields.plan import PLANNERS
from ntfields.viz.catalog import RESULTS, SPLITS, Catalog, queries_of, recorded_success
from ntfields.viz.geometry import layers, mount_layer, soup
from ntfields.viz.inference import PlanView, Worker, field_slice, judged_as, plan_view
from ntfields.viz.robot import path_points, robot_view, tool_positions

log = logging.getLogger(__name__)

SCENE = (112, 170, 120)
MOUNT = (150, 150, 150)
ROBOT = (205, 80, 70)
START = (60, 120, 220)
GOAL = (240, 160, 40)
GOOD = (40, 180, 90)
BAD = (220, 50, 50)
UNKNOWN = (150, 150, 150)
NONE = "(none)"
FILTERS = ("all", "failed", "succeeded", "task", "uniform")
FPS = 30


@dataclass
class Config:
    port: int = 8080
    suite: str | None = None  # the suite to open with
    suites_dir: Path = DEFAULT_ROOT
    checkpoints_dir: Path = Path("benchmarks/checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Viewer:
    def __init__(self, cfg: Config, results: Sequence[Path] = RESULTS) -> None:
        self.cfg = cfg
        self.catalog = Catalog(cfg.suites_dir, results, cfg.checkpoints_dir)
        self.server = viser.ViserServer(port=cfg.port)
        self.scene = self.server.scene
        self.lock = threading.RLock()
        self.worker = Worker()
        self.suite: Suite | None = None
        self.problem: SuiteProblem | None = None
        self.records: list[Record] = []
        self.visible: list[int] = []  # the pair indices that pass the filter
        self.trained: TrainedField | None = None
        self.plan: PlanView | None = None
        self.sliders: dict[str, list[Any]] = {"start": [], "goal": []}
        self.views: dict[str, Any] = {}
        self.handles: dict[str, Any] = {}
        self._gui()

    def _gui(self) -> None:
        gui = self.server.gui
        suites = self.catalog.suites()
        with gui.add_folder("Scene"):
            start = self.cfg.suite if self.cfg.suite in suites else suites[0]
            self.g_suite = gui.add_dropdown("Suite", suites, initial_value=start)
            self.g_problem = gui.add_dropdown("Problem", [NONE])
            self.g_layer = gui.add_dropdown("Geometry", [NONE])
            self.g_mount = gui.add_checkbox("Pedestal and floor", True)
        with gui.add_folder("Robot"):
            self.g_meshes = gui.add_dropdown("Draw as", ["collision spheres", "meshes"])
            self.g_ghosts = gui.add_checkbox("Start and goal poses", True)
        with gui.add_folder("Tested pairs"):
            self.g_split = gui.add_dropdown("Split", list(SPLITS))
            self.g_entry = gui.add_dropdown("Entry", [NONE])
            self.g_seed = gui.add_dropdown("Seed", [NONE])
            self.g_filter = gui.add_dropdown("Show", list(FILTERS))
            self.g_pair = gui.add_slider("Pair", 0, 1, 1, 0)
            self.g_label = gui.add_markdown("")
            self.g_overview = gui.add_checkbox("All pairs", False)
        with gui.add_folder("Inference"):
            self.g_checkpoint = gui.add_dropdown("Checkpoint", [NONE])
            self.g_planner = gui.add_dropdown("Planner", list(PLANNERS))
            self.g_self = gui.add_checkbox("Self-collision counts", False)
            self.g_source = gui.add_dropdown("Query", ["tested pair", "custom"])
            self.g_run = gui.add_button("Run")
            self.g_result = gui.add_markdown("Load a checkpoint and press Run.")
            self.g_time = gui.add_slider("Along path", 0.0, 1.0, 0.001, 0.0)
            self.g_play = gui.add_checkbox("Play", False)
        self.g_custom = gui.add_folder("Custom query", expand_by_default=False)
        with gui.add_folder("Travel-time field"):
            self.g_field = gui.add_checkbox("Show T from start", False)
            self.g_height = gui.add_slider("Height z", -1.0, 1.0, 0.01, 0.0)
            self.g_resolution = gui.add_slider("Resolution", 41, 301, 20, 161)

        self.g_suite.on_update(lambda _: self.load_suite())
        self.g_problem.on_update(lambda _: self.load_problem())
        self.g_layer.on_update(lambda _: self.draw_layer())
        self.g_mount.on_update(lambda _: self.draw_mount())
        self.g_meshes.on_update(lambda _: self.draw_query())
        self.g_ghosts.on_update(lambda _: self.draw_query())
        for g in (self.g_split, self.g_entry, self.g_seed, self.g_filter):
            g.on_update(lambda _: self.load_pairs())
        self.g_pair.on_update(lambda _: self.draw_query())
        self.g_overview.on_update(lambda _: self.draw_overview())
        self.g_checkpoint.on_update(lambda _: self.load_checkpoint())
        self.g_source.on_update(lambda _: self.draw_query())
        self.g_run.on_click(lambda _: self.run())
        self.g_time.on_update(lambda _: self.draw_along_path())
        for g in (self.g_field, self.g_height, self.g_resolution):
            g.on_update(lambda _: self.draw_field())
        self.load_suite()

    # scene

    def load_suite(self) -> None:
        with self.lock:
            self.suite = load_suite(self.g_suite.value, self.cfg.suites_dir)
            self.records = self.catalog.records(self.suite.id)
            names = [p.name for p in self.suite.problems]
            self.g_problem.options = names
            self.g_problem.value = names[0]
            self.load_problem()

    def load_problem(self) -> None:
        with self.lock:
            assert self.suite is not None
            self.problem = self.suite.get(self.g_problem.value)
            self.env = self.problem.problem().env
            self.layers = layers(self.problem, self.cfg.suites_dir / self.suite.id)
            self.mount = mount_layer(self.problem)
            self.g_layer.options = list(self.layers)
            self.g_layer.value = next(iter(self.layers))
            for name, view in list(self.views.items()):
                view.remove()
                del self.views[name]
            self.views = {
                "robot": robot_view(self.scene, "/robot", self.env, ROBOT, 1.0),
                "start": robot_view(self.scene, "/start", self.env, START, 0.35),
                "goal": robot_view(self.scene, "/goal", self.env, GOAL, 0.35),
            }
            self._custom_sliders()
            self.trained, self.plan = None, None
            self._drop("path", "pair", "overview", "field")
            self.g_result.content = "Load a checkpoint and press Run."
            options = [str(p.relative_to(self.cfg.checkpoints_dir)) for p in
                       self.catalog.checkpoints(self.suite.id, self.problem.name)]  # fmt: skip
            self.g_checkpoint.options = [NONE, *options]
            self.g_checkpoint.value = NONE
            dim3 = self.env.dim == 3
            self.g_height.visible = dim3
            self.g_field.disabled = self.env.dim > 3
            self.g_field.value = False
            if dim3:
                lo, hi = (float(v) for v in self.env.bounds[2])
                self.g_height.min, self.g_height.max, self.g_height.value = lo, hi, (lo + hi) / 2
            self._frame_camera()
            self.draw_layer()
            self.draw_mount()
            self._refresh_entries()

    def _frame_camera(self) -> None:
        """Straight down on a 2D scene, and from the front and above otherwise, for clients yet to connect or there."""
        if isinstance(self.env, ArmEnv):
            position, look_at = (-1.3, -1.5, 1.1), (0.4, 0.0, 0.0)
        else:
            span = float((self.env.bounds[:, 1] - self.env.bounds[:, 0]).max())
            position = (0.0, 0.0, 1.6 * span) if self.env.dim == 2 else (-span, -span, span)
            look_at = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0) if self.env.dim == 2 else (0.0, 0.0, 1.0)
        self.server.initial_camera.position = position
        self.server.initial_camera.look_at = look_at
        self.server.initial_camera.up = up
        for client in self.server.get_clients().values():
            client.camera.position, client.camera.look_at, client.camera.up_direction = position, look_at, up

    def _custom_sliders(self) -> None:
        for sliders in self.sliders.values():
            for s in sliders:
                s.remove()
        self.sliders = {"start": [], "goal": []}
        assert self.problem is not None
        q = self.problem.queries["test"]
        with self.g_custom:
            for key, values in (("start", q.start[0]), ("goal", q.goal[0])):
                for i, (lo, hi) in enumerate(self.env.bounds.tolist()):
                    label = f"{key} joint {i + 1}" if isinstance(self.env, ArmEnv) else f"{key} {'xyz'[i]}"
                    step = (hi - lo) / 500
                    slider = self.server.gui.add_slider(label, lo, hi, step, float(values[i]))
                    slider.on_update(lambda _: self.draw_query())
                    self.sliders[key].append(slider)

    def draw_layer(self) -> None:
        with self.lock:
            layer = self.layers.get(self.g_layer.value)
            self._drop("layer")
            if layer is None:
                return
            if len(layer.points):
                self.handles["layer"] = self.scene.add_point_cloud(
                    "/scene/points", layer.points, np.array(SCENE, np.uint8), point_size=0.008, precision="float32"
                )
            elif len(layer.triangles):
                vertices, faces = soup(layer.triangles)
                self.handles["layer"] = self.scene.add_mesh_simple(
                    "/scene/mesh", vertices, faces, color=SCENE, opacity=0.55, side="double"
                )

    def draw_mount(self) -> None:
        with self.lock:
            self._drop("mount")
            if self.g_mount.value and len(self.mount.triangles):
                vertices, faces = soup(self.mount.triangles)
                self.handles["mount"] = self.scene.add_mesh_simple("/mount", vertices, faces, color=MOUNT, opacity=0.4)

    def _drop(self, *names: str) -> None:
        for name in names:
            handle = self.handles.pop(name, None)
            if handle is not None:
                handle.remove()

    # tested pairs

    def _problem_records(self) -> list[Record]:
        assert self.problem is not None
        split = self.g_split.value
        return [r for r in self.records if r.problem == self.problem.name and recorded_success(r, split) is not None]

    def _refresh_entries(self) -> None:
        entries = sorted({r.entry for r in self._problem_records()})
        self.g_entry.options = [NONE, *entries]
        self.g_entry.value = entries[0] if entries else NONE
        self.load_pairs()

    def _record(self) -> Record | None:
        seed = self.g_seed.value
        rows = [r for r in self._problem_records() if r.entry == self.g_entry.value]
        return next((r for r in rows if str(r.seed) == seed), None)

    def _outcomes(self) -> list[bool] | None:
        record = self._record()
        return None if record is None else recorded_success(record, self.g_split.value)

    def load_pairs(self) -> None:
        with self.lock:
            assert self.problem is not None
            rows = [r for r in self._problem_records() if r.entry == self.g_entry.value]
            seeds = sorted({str(r.seed) for r in rows})
            options = seeds or [NONE]
            if tuple(self.g_seed.options) != tuple(options):
                self.g_seed.options = options
            if self.g_seed.value not in options:
                self.g_seed.value = options[0]
            pairs = queries_of(self.problem, self.g_split.value)
            outcomes, shown = self._outcomes(), self.g_filter.value
            n = 0 if pairs is None else len(pairs)
            kinds = (pairs.kind if pairs is not None else None) or []

            def keep(i: int) -> bool:
                if shown in ("failed", "succeeded"):
                    return outcomes is not None and outcomes[i] == (shown == "succeeded")
                return shown == "all" or (bool(kinds) and kinds[i] == shown)

            self.visible = [i for i in range(n) if keep(i)]
            self.g_pair.max = max(len(self.visible) - 1, 1)
            self.g_pair.value = 0
            self.draw_query()
            self.draw_overview()

    def _pair_index(self) -> int | None:
        if not self.visible:
            return None
        return self.visible[min(int(self.g_pair.value), len(self.visible) - 1)]

    def query(self) -> tuple[Tensor, Tensor] | None:
        """The start and goal being shown."""
        assert self.problem is not None
        if self.g_source.value == "custom":
            return tuple(
                torch.tensor([s.value for s in self.sliders[k]], dtype=torch.float32) for k in ("start", "goal")
            )  # pyright: ignore[reportReturnType]
        pairs, i = queries_of(self.problem, self.g_split.value), self._pair_index()
        return None if pairs is None or i is None else (pairs.start[i], pairs.goal[i])

    def _label(self) -> str:
        if self.g_source.value == "custom":
            return "**Custom query**"
        i = self._pair_index()
        if i is None:
            return "No pairs to show."
        assert self.problem is not None
        pairs = queries_of(self.problem, self.g_split.value)
        assert pairs is not None
        kind = f" · {pairs.kind[i]}" if pairs.kind else ""
        record, outcomes = self._record(), self._outcomes()
        if outcomes is None or record is None:
            return f"Pair {i}{kind} · no recorded result"
        cost = record.cost[i] if self.g_split.value != "challenge" else None
        text = f"{'✅ success' if outcomes[i] else '❌ failure'} · pair {i}{kind}"
        return text + (f" · cost {cost:.3f}" if cost is not None else "")

    def draw_query(self) -> None:
        with self.lock:
            self.g_label.content = self._label()
            query = self.query()
            meshes = self.g_meshes.value == "meshes"
            self.views["robot"].set_style(meshes)
            for key, q in zip(("start", "goal"), query or (None, None), strict=True):
                self.views[key].set_style(meshes, self.g_ghosts.value and q is not None)
                if q is not None:
                    self.views[key].set(q)
            self.plan = None
            self._drop("path", "collision")
            if query is not None:
                self.views["robot"].set(query[0])
                self._draw_pair(query)
            if self.g_field.value:
                self.draw_field()

    def _draw_pair(self, query: tuple[Tensor, Tensor]) -> None:
        ends = tool_positions(self.env, torch.stack(query))
        self._drop("pair")
        self.handles["pair"] = self.scene.add_line_segments(
            "/pair",
            ends[None].astype(np.float32),
            colors=np.array([[START, GOAL]], np.uint8),
            thickness=1.5,
            thickness_units="screen",
        )

    def draw_overview(self) -> None:
        with self.lock:
            self._drop("overview")
            assert self.problem is not None
            pairs = queries_of(self.problem, self.g_split.value)
            if not self.g_overview.value or pairs is None or not self.visible:
                return
            index = torch.tensor(self.visible)
            starts = tool_positions(self.env, pairs.start[index])
            goals = tool_positions(self.env, pairs.goal[index])
            outcomes = self._outcomes()
            colors = [UNKNOWN if outcomes is None else (GOOD if outcomes[i] else BAD) for i in self.visible]
            lines = np.stack([starts, goals], 1).astype(np.float32)
            tint = np.repeat(np.array(colors, np.uint8)[:, None], 2, axis=1)
            self.handles["overview"] = self.scene.add_line_segments(
                "/overview", lines, colors=tint, thickness=1.0, thickness_units="screen"
            )

    # inference

    def load_checkpoint(self) -> None:
        name = self.g_checkpoint.value
        self.trained = None
        self.g_run.disabled = name == NONE
        if name == NONE:
            return
        self.g_result.content = f"Loading {name}…"

        def done(result: Any, error: Exception | None) -> None:
            if error is not None:
                self.g_result.content = f"Could not load it: `{error}`"
                return
            if self.g_checkpoint.value != name:
                return  # another checkpoint was chosen meanwhile
            self.trained = result
            self.g_result.content = f"Loaded `{name}`."
            if self.g_field.value:
                self.draw_field()

        path = self.cfg.checkpoints_dir / name
        self.worker.submit(lambda: load(path, self.cfg.device), done)

    def _rng_seed(self) -> int:
        """The seed `evaluate` used for this pair, so that a stochastic planner replays the recorded run."""
        seed = int(self.g_seed.value) if self.g_seed.value != NONE else 0
        i = self._pair_index() or 0
        return seed * 1_000_003 + (CHALLENGE_SEED_OFFSET if self.g_split.value == "challenge" else 0) + i

    def run(self) -> None:
        query, trained = self.query(), self.trained
        if self.g_checkpoint.value != NONE and trained is None:
            self.g_result.content = "Still loading the checkpoint; press Run again in a moment."
            return
        if trained is None or self.suite is None or query is None:
            self.g_result.content = "Choose a checkpoint and a query first."
            return
        protocol = judged_as(self.suite, self.g_self.value)
        planner, seed, outcomes, i = self.g_planner.value, self._rng_seed(), self._outcomes(), self._pair_index()
        custom = self.g_source.value == "custom"
        self.g_result.content = "Planning…"

        def done(result: Any, error: Exception | None) -> None:
            if error is not None:
                self.g_result.content = f"Planning failed: `{error}`"
                return
            shown = self.query()
            if shown is None or not all(torch.equal(a, b) for a, b in zip(shown, query, strict=True)):
                self.g_result.content = "The query changed while planning; press Run again."
                return
            recorded = None if custom or outcomes is None or i is None else outcomes[i]
            self._show_plan(result, recorded)

        self.worker.submit(lambda: plan_view(trained, protocol, *query, planner, seed, self.cfg.device), done)

    def _show_plan(self, plan: PlanView, recorded: bool | None) -> None:
        with self.lock:
            self.plan = plan
            points = path_points(self.env, plan.path).astype(np.float32)
            color = GOOD if plan.success else BAD
            self._drop("path", "collision")
            self.handles["path"] = self.scene.add_line_segments(
                "/path",
                np.stack([points[:-1], points[1:]], 1),
                colors=np.array(color, np.uint8),
                thickness=3.0,
                thickness_units="screen",
            )
            self.g_result.content = result_text(plan, recorded)
            if plan.first_invalid is not None:
                at = points[round(plan.first_invalid * (len(points) - 1))]
                self.handles["collision"] = self.scene.add_icosphere("/collision", 0.02, BAD, position=at)
            self.g_time.value = 0.0
            self.draw_along_path()

    def draw_along_path(self) -> None:
        with self.lock:
            if self.plan is None:
                return
            path = self.plan.path
            at = self.g_time.value * (len(path) - 1)
            low = min(int(at), len(path) - 2)
            self.views["robot"].set(path[low] + (path[low + 1] - path[low]) * (at - low))

    def draw_field(self) -> None:
        with self.lock:
            self._drop("field")
            query = self.query()
            if not self.g_field.value or self.trained is None or query is None or self.env.dim > 3:
                return
            trained, start, height = self.trained, query[0], self.g_height.value
            resolution = int(self.g_resolution.value)
            device = self.cfg.device

        def done(result: Any, error: Exception | None) -> None:
            if error is not None:
                self.g_result.content = f"Field failed: `{error}`"
                return
            with self.lock:
                self._drop("field")
                if not self.g_field.value:
                    return
                bounds = self.env.bounds
                size = 1.05 * float((bounds[:, 1] - bounds[:, 0]).max()) / (resolution - 1)
                self.handles["field"] = self.scene.add_point_cloud(
                    "/field",
                    result[0],
                    result[1],
                    point_size=size,
                    point_shape="square",
                    precision="float32",
                    point_shading="flat",
                )

        self.worker.submit(lambda: field_slice(trained, start, height, resolution, device), done)

    def spin(self) -> None:
        while True:
            time.sleep(1 / FPS)
            if self.g_play.value and self.plan is not None:
                self.g_time.value = (self.g_time.value + 1 / (4 * FPS)) % 1.0


def result_text(plan: PlanView, recorded: bool | None) -> str:
    """A plan's outcome, and whether it agrees with the record of the same query."""
    verdict = "✅ success" if plan.success else "❌ failure"
    lines = [
        f"**{verdict}** (planner {'converged' if plan.converged else 'gave up'})",
        f"T = {plan.travel_time:.3f}, cost = {'-' if plan.cost is None else f'{plan.cost:.3f}'}, "
        f"plan {plan.seconds * 1000:.0f} ms",
        f"min clearance {plan.min_clearance:.3f}",
    ]
    if plan.first_invalid is not None:
        lines.append(f"first invalid at {plan.first_invalid:.0%} of the path")
    if recorded is not None and recorded != plan.success:
        lines.append("⚠️ differs from the recorded result for this pair")
    return "\n\n".join(lines)


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = tyro.cli(Config, args=args, prog="ntfields viz")
    viewer = Viewer(cfg)
    log.info("open http://localhost:%d", cfg.port)
    viewer.spin()


__all__ = ["Config", "Viewer", "main"]
