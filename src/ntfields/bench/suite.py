import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ntfields.base import Environment, SpeedModel
from ntfields.bench.protocol import Protocol
from ntfields.envs import ArmEnv, SphereBoxEnv
from ntfields.envs.solids import SolidScene
from ntfields.problem import Problem
from ntfields.speed import Clipped, ClippedEased, Smoothstep

FORMAT_VERSION = 1
SPLITS = ("dev", "test")
DEFAULT_ROOT = Path("benchmarks/suites")


class SuiteIntegrityError(RuntimeError):
    """A suite file is missing or does not match the checksum in suite.json."""


@dataclass
class Queries:
    start: Tensor  # (N, dim)
    goal: Tensor
    c_star: Tensor | None = None  # (N,) cost of a near-optimal reference path, where a reference exists
    t_star: Tensor | None = None  # (N,) reference travel time
    straight_free: Tensor | None = None  # (N,) bool: the straight segment is already valid
    kind: list[str] | None = None  # per query, where a suite mixes kinds ("task", "uniform")

    def __len__(self) -> int:
        return self.start.shape[0]

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start": self.start.tolist(), "goal": self.goal.tolist()}
        for name in ("c_star", "t_star", "straight_free"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value.tolist()
        if self.kind is not None:
            out["kind"] = self.kind
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Queries":
        def tensor(name: str, dtype: torch.dtype) -> Tensor | None:
            return torch.tensor(data[name], dtype=dtype) if name in data else None

        start, goal = tensor("start", torch.float32), tensor("goal", torch.float32)
        assert start is not None and goal is not None
        return cls(start, goal, tensor("c_star", torch.float64), tensor("t_star", torch.float64),
                   tensor("straight_free", torch.bool), data.get("kind"))  # fmt: skip


@dataclass
class SuiteProblem:
    name: str
    scene: dict[str, Any]  # the scene as data (scenes/<name>.json)
    speed: dict[str, Any]  # {"kind": ..., **params}
    queries: dict[str, Queries]  # by split

    def problem(self) -> Problem:
        return Problem(build_env(self.scene), build_speed(self.speed))


@dataclass
class Suite:
    name: str
    version: int
    description: str
    leaderboard: bool  # False for suites only meant for testing (smoke)
    min_seeds: int
    protocol: Protocol
    problems: list[SuiteProblem]
    sha256: str  # of suite.json, which holds the checksums of every other file

    @property
    def id(self) -> str:
        return f"{self.name}-v{self.version}"

    def get(self, problem: str) -> SuiteProblem:
        for p in self.problems:
            if p.name == problem:
                return p
        raise KeyError(f"{self.id} has no problem {problem!r}; it has {[p.name for p in self.problems]}")


def build_env(scene: dict[str, Any]) -> Environment:
    if scene["kind"] == "sphere_box":
        return SphereBoxEnv.build(
            bounds=scene["bounds"],
            spheres=[(s["center"], s["radius"]) for s in scene.get("spheres", [])],
            boxes=[(b["center"], b["size"]) for b in scene.get("boxes", [])],
        )
    if scene["kind"] == "arm" and scene["robot"] == "ur5e_2f85":
        return ArmEnv.ur5e_2f85(SolidScene.from_json(scene["objects"]))
    raise ValueError(f"unknown scene kind {scene['kind']!r}")


def build_speed(spec: dict[str, Any]) -> SpeedModel:
    kinds: dict[str, type] = {"clipped": Clipped, "clipped_eased": ClippedEased, "smoothstep": Smoothstep}
    params = {k: v for k, v in spec.items() if k != "kind"}
    return kinds[spec["kind"]](**params)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available_suites(root: Path = DEFAULT_ROOT) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/suite.json"))


def load_suite(name: str, root: Path = DEFAULT_ROOT) -> Suite:
    """Load `root/name/suite.json` and its files, verifying every checksum."""
    base = root / name
    meta_path = base / "suite.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no suite {name!r} in {root}; available: {available_suites(root)}")
    meta = json.loads(meta_path.read_text())
    if meta["format_version"] != FORMAT_VERSION:
        raise SuiteIntegrityError(f"{name}: format {meta['format_version']}, this ntfields reads {FORMAT_VERSION}")
    for rel, digest in meta["files"].items():
        path = base / rel
        if not path.exists() or sha256(path) != digest:
            raise SuiteIntegrityError(f"{name}: {rel} is missing or does not match its checksum")
    problems = []
    for p in meta["problems"]:
        queries = json.loads((base / p["queries"]).read_text())
        scene = json.loads((base / p["scene"]).read_text())
        problems.append(SuiteProblem(p["name"], scene, p["speed"], {s: Queries.from_json(queries[s]) for s in SPLITS}))
    return Suite(meta["name"], meta["version"], meta["description"], meta["leaderboard"], meta["min_seeds"],
                 Protocol(**meta["protocol"]), problems, sha256(meta_path))  # fmt: skip
