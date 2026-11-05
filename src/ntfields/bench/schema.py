import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Record:
    """One (suite version, problem, split, entry, planner, seed) evaluation. Flat JSON, readable without ntfields."""

    # suite
    suite: str
    suite_version: int
    suite_sha256: str
    problem: str
    split: str
    # entry
    method: str  # the registered method class name
    preset: str  # the name the entry was created from (a method or preset name)
    method_package: str
    method_version: str
    config: dict[str, Any]
    planner: str
    # provenance
    ntfields_version: str
    seed: int
    device: str
    torch_version: str
    date: str
    planner_config: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    gpu: str | None = None
    submitter: str | None = None
    # scenes (one each until multi-scene suites)
    train_scenes: list[str] = field(default_factory=list)
    test_scenes: list[str] = field(default_factory=list)
    # training
    epochs: int = 0
    train_seconds: float = 0.0
    n_pairs: int = 0
    final_loss: float | None = None
    rollbacks: int = 0
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    # field quality (null without a reference)
    t_error_mean: float | None = None
    t_error_median: float | None = None
    # queries
    n: int = 0
    success_rate: float = 0.0
    cost_mean: float | None = None  # over successful queries
    length_mean: float | None = None
    spl: float | None = None  # null without a reference
    cost_ratio_mean: float | None = None
    plan_seconds_mean: float = 0.0
    plan_seconds_p95: float = 0.0
    self_collision: bool = False  # whether self-collision counted against success
    challenge_success_rate: float | None = None  # on pairs whose straight line is blocked; null where none can exist
    challenge_sha256: str | None = None  # of the suite's challenge.json; null if the test pairs themselves are the set
    success: list[bool] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)  # the planner's own flag; not used for success
    cost: list[float | None] = field(default_factory=list)
    length: list[float | None] = field(default_factory=list)
    plan_seconds: list[float] = field(default_factory=list)
    challenge_success: list[bool] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def suite_id(self) -> str:
        return f"{self.suite}-v{self.suite_version}"

    @property
    def entry(self) -> str:
        return f"{self.preset}+{self.planner}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"record schema_version {data.get('schema_version')}, this ntfields reads {SCHEMA_VERSION}"
            )
        names = {f.name for f in fields(cls)}
        unknown = data.keys() - names
        if unknown:
            raise ValueError(f"unknown record fields {sorted(unknown)}")
        return cls(**data)


def append_records(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def read_records(*paths: Path) -> list[Record]:
    return [Record.from_dict(json.loads(line)) for p in paths for line in p.read_text().splitlines() if line.strip()]
