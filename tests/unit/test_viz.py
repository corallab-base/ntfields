import socket
import threading
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("viser")

from ntfields.bench.build import build_challenge, build_suite
from ntfields.bench.protocol import Protocol
from ntfields.bench.run import RunConfig, run_suite
from ntfields.bench.suite import load_suite
from ntfields.checkpoint import load
from ntfields.envs import ArmEnv
from ntfields.envs.solids import SolidScene
from ntfields.problem import Problem
from ntfields.speed import Clipped
from ntfields.viz.app import NONE, Config, Viewer
from ntfields.viz.catalog import Catalog, queries_of, recorded_success
from ntfields.viz.geometry import layers, mount_layer, soup
from ntfields.viz.inference import Worker, colormap
from ntfields.viz.robot import tool_positions

PROTOCOL = Protocol(margin=0.01, resolution=0.005)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _suite(root: Path) -> None:
    scene = {
        "kind": "sphere_box",
        "bounds": [[-1, 1]] * 2,
        "spheres": [{"center": [0.0, 0.0], "radius": 0.3}],
        "boxes": [{"center": [0.6, 0.6], "size": [0.2, 0.2]}],
    }
    speed = {"kind": "clipped", "dmin": 0.01, "dmax": 0.1}
    build_suite(root, "v", 1, "viz", {"one": scene}, speed, PROTOCOL, {"dev": 1, "test": 2}, min_distance=0.5,
                fmm_resolution=51, seed=0, min_seeds=1)  # fmt: skip
    build_challenge(root / "v-v1", 2, seed=1)


def _run(root: Path) -> Path:
    out = root / "results" / "v-v1" / "euclidean.jsonl"
    run_suite(RunConfig("v-v1", ["euclidean"], seeds=[0], out=out, suites_dir=root, device="cpu",
                        checkpoint_dir=root / "ckpt", epochs=1, n_pairs=16))  # fmt: skip
    return out


def _wait(viewer: Viewer, check: object, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while not check() and time.time() < deadline:  # pyright: ignore[reportCallIssue]
        time.sleep(0.01)
    assert check(), viewer.g_result.content  # pyright: ignore[reportCallIssue]


def test_catalog_finds_records_and_checkpoints(tmp_path: Path):
    _suite(tmp_path)
    _run(tmp_path)
    catalog = Catalog(tmp_path, results=(tmp_path / "results",), checkpoints_dir=tmp_path / "ckpt")
    assert catalog.suites() == ["v-v1"]
    (record,) = catalog.records("v-v1")
    assert recorded_success(record, "test") == record.success
    assert recorded_success(record, "dev") is None
    assert recorded_success(record, "challenge") == record.challenge_success
    assert [p.name for p in catalog.checkpoints("v-v1", "one")] == ["seed0.pt"]
    problem = load_suite("v-v1", tmp_path).get("one")
    assert len(queries_of(problem, "challenge")) == 2  # pyright: ignore[reportArgumentType]


def test_flat_scene_layer_and_soup(tmp_path: Path):
    _suite(tmp_path)
    problem = load_suite("v-v1", tmp_path).get("one")
    layer = layers(problem, tmp_path / "v-v1")["true geometry"]
    assert len(layer.triangles) > 12 and abs(layer.triangles[..., 2]).max() < 0.02
    vertices, faces = soup(layer.triangles)
    assert len(vertices) == 3 * len(faces) and faces.max() == len(vertices) - 1
    assert not len(mount_layer(problem).triangles)


def test_colormap_handles_infinite_values():
    colors = colormap(torch.tensor([0.0, 1.0, float("inf")]).numpy())
    assert colors.shape == (3, 3) and colors.min() >= 0 and colors.max() <= 1
    assert (colors[1] == colors[2]).all()


def test_tool_positions_of_points_and_arms():
    pytest.importorskip("pytorch_kinematics")
    arm = ArmEnv.ur5e_2f85()
    assert tool_positions(arm, torch.zeros(2, 6)).shape == (2, 3)


def test_arm_scene_survives_a_checkpoint(tmp_path: Path):
    pytest.importorskip("pytorch_kinematics")
    box = {"id": "b", "type": "box", "center": [0.6, 0.0, 0.2], "size": [0.2, 0.2, 0.2], "quat": [0.0, 0.0, 0.0, 1.0]}
    from ntfields.checkpoint import save
    from ntfields.methods.euclidean.method import Euclidean, EuclideanConfig

    problem = Problem(ArmEnv.ur5e_2f85(SolidScene.from_json([box])), Clipped(0.01, 0.1))
    method = Euclidean(EuclideanConfig())
    save(tmp_path / "c.pt", method, problem, method.build_model(problem), 0)
    loaded = load(tmp_path / "c.pt")
    q = torch.rand(8, 6) - 0.5
    torch.testing.assert_close(loaded.problem.env.distance(q), problem.env.distance(q))


def test_worker_keeps_only_the_latest_waiting_job():
    worker = Worker()
    gate, finished, ran = threading.Event(), threading.Event(), []
    worker.submit(lambda: gate.wait(5), lambda *_: ran.append("first"))
    time.sleep(0.05)
    worker.submit(lambda: None, lambda *_: ran.append("stale"))

    def last(*_: object) -> None:
        ran.append("latest")
        finished.set()

    worker.submit(lambda: None, last)
    gate.set()
    assert finished.wait(5)
    assert ran == ["first", "latest"]


def test_worker_reports_errors_and_survives_them():
    worker, finished, errors = Worker(), threading.Event(), []

    def fail() -> None:
        raise ValueError("bad checkpoint")

    worker.submit(fail, lambda _, e: errors.append(e))

    def last(result: object, _: object) -> None:
        errors.append(result)
        finished.set()

    worker.submit(lambda: 1, last)
    assert finished.wait(5)
    assert errors[-1] == 1


def test_viewer_shows_pairs_runs_a_plan_and_draws_the_field(tmp_path: Path):
    _suite(tmp_path)
    _run(tmp_path)
    cfg = Config(port=_free_port(), suites_dir=tmp_path, checkpoints_dir=tmp_path / "ckpt", device="cpu")
    viewer = Viewer(cfg, results=(tmp_path / "results",))
    try:
        assert viewer.g_problem.value == "one" and viewer.g_entry.value != NONE
        assert "pair 0" in viewer.g_label.content and (
            "success" in viewer.g_label.content or "failure" in viewer.g_label.content
        )

        viewer.g_overview.value = True
        viewer.draw_overview()
        assert "overview" in viewer.handles

        viewer.g_split.value = "challenge"
        viewer.load_pairs()
        assert len(viewer.visible) == 2
        viewer.g_split.value = "test"
        viewer.load_pairs()

        viewer.g_filter.value = "succeeded"
        viewer.load_pairs()
        outcomes = viewer._outcomes()
        assert outcomes is not None and len(viewer.visible) == sum(outcomes)
        viewer.g_filter.value = "all"
        viewer.load_pairs()

        viewer.g_checkpoint.value = viewer.g_checkpoint.options[1]
        viewer.load_checkpoint()
        _wait(viewer, lambda: viewer.trained is not None)
        viewer.g_planner.value = "gradient_bidirectional"
        viewer.run()
        _wait(viewer, lambda: viewer.plan is not None)
        assert "path" in viewer.handles and (
            "success" in viewer.g_result.content or "failure" in viewer.g_result.content
        )
        assert "differs" not in viewer.g_result.content  # the same seed replays the recorded result
        viewer.g_time.value = 0.5
        viewer.draw_along_path()

        viewer.g_field.value = True
        viewer.draw_field()
        _wait(viewer, lambda: "field" in viewer.handles)

        viewer.g_source.value = "custom"
        viewer.draw_query()
        assert viewer.g_label.content == "**Custom query**"
        assert len(viewer.sliders["start"]) == 2
    finally:
        viewer.server.stop()
