from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ntfields.bench.schema import Record, read_records
from ntfields.bench.suite import DEFAULT_ROOT, Queries, SuiteProblem, available_suites

SPLITS = ("test", "dev", "challenge")
RESULTS = (Path("benchmarks/results"), Path("results"))


def queries_of(problem: SuiteProblem, split: str) -> Queries | None:
    return problem.challenge if split == "challenge" else problem.queries.get(split)


def recorded_success(record: Record, split: str) -> list[bool] | None:
    """A record's per-query results on a split: challenge results live in the test records."""
    if split == "challenge":
        return record.challenge_success or None
    return record.success if record.split == split else None


@dataclass
class Catalog:
    """What is on disk: suites, the records made on them, and their checkpoints."""

    suites_dir: Path = DEFAULT_ROOT
    results: Sequence[Path] = RESULTS
    checkpoints_dir: Path = Path("benchmarks/checkpoints")

    def suites(self) -> list[str]:
        return available_suites(self.suites_dir)

    def records(self, suite_id: str) -> list[Record]:
        paths = [p for root in self.results for p in (*root.glob(f"{suite_id}/*.jsonl"), root / f"{suite_id}.jsonl")]
        return read_records(*[p for p in paths if p.is_file()])

    def checkpoints(self, suite_id: str, problem: str) -> list[Path]:
        """<suite>/<problem>/<preset>/seed<k>.pt"""
        return sorted((self.checkpoints_dir / suite_id / problem).glob("*/seed*.pt"))
