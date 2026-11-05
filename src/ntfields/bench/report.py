import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from ntfields.bench.schema import Record
from ntfields.bench.suite import DEFAULT_ROOT, load_suite

Metric = tuple[str, Callable[[Record], float | None], float, str]  # header, value, scale, format

METRICS: list[Metric] = [
    ("success %", lambda r: r.success_rate, 100, ".1f"),
    ("SPL", lambda r: r.spl, 1, ".3f"),
    ("cost ratio", lambda r: r.cost_ratio_mean, 1, ".3f"),
    ("T error %", lambda r: r.t_error_mean, 100, ".1f"),
    ("plan ms", lambda r: r.plan_seconds_mean, 1000, ".0f"),
    ("train min", lambda r: r.train_seconds, 1 / 60, ".1f"),
]


def report(records: Sequence[Record]) -> str:
    """Markdown tables, one per (suite, split, self-collision setting): a row per entry and problem plus an 'all' row
    averaging the problems, each cell the mean ± std over seeds."""
    groups: dict[tuple[str, str, bool], list[Record]] = defaultdict(list)
    for r in records:
        groups[r.suite_id, r.split, r.self_collision].append(r)
    out = []
    for (suite_id, split, self_collision), rs in sorted(groups.items()):
        out.append(f"### {suite_id} ({split}{', with self-collision' if self_collision else ''})\n")
        out.append("| problem | entry | seeds | " + " | ".join(m[0] for m in METRICS) + " |")
        out.append("|---|---|---|" + "---|" * len(METRICS))
        problems = sorted({r.problem for r in rs})
        entries = sorted({r.entry for r in rs})
        for problem in [*problems, "all"]:
            for entry in entries:
                cell = [r for r in rs if r.entry == entry and (problem == "all" or r.problem == problem)]
                if not cell:
                    continue
                by_seed = _per_seed(cell)
                values = " | ".join(_cell([m[1](r) for r in by_seed], m[2], m[3]) for m in METRICS)
                out.append(f"| {problem} | {entry} | {len(by_seed)} | {values} |")
        out.append("")
    return "\n".join(out)


def _per_seed(records: list[Record]) -> list[Record]:
    """For an 'all' row, one pseudo-record per seed averaging the problems; otherwise the records themselves."""
    seeds: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        seeds[r.seed].append(r)
    if all(len(v) == 1 for v in seeds.values()):
        return [v[0] for v in seeds.values()]
    merged = []
    for rs in seeds.values():
        first = rs[0]
        merged.append(Record(**{
            **vars(first),
            "success_rate": statistics.fmean(r.success_rate for r in rs),
            "spl": _mean([r.spl for r in rs]),
            "cost_ratio_mean": _mean([r.cost_ratio_mean for r in rs]),
            "t_error_mean": _mean([r.t_error_mean for r in rs]),
            "plan_seconds_mean": statistics.fmean(r.plan_seconds_mean for r in rs),
            "train_seconds": statistics.fmean(r.train_seconds for r in rs),
        }))  # fmt: skip
    return merged


def _mean(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else statistics.fmean(v for v in values if v is not None)


def _cell(values: list[float | None], scale: float, fmt: str) -> str:
    if not values or any(v is None for v in values):
        return "-"
    xs = [v * scale for v in values if v is not None]
    if len(xs) == 1:
        return f"{xs[0]:{fmt}}"
    return f"{statistics.fmean(xs):{fmt}} ± {statistics.stdev(xs):{fmt}}"


def validate(records: Sequence[Record], suites_dir: Path = DEFAULT_ROOT) -> list[str]:
    """Problems that keep records off a leaderboard: unknown or changed suites, dev-split records, missing problems
    or seeds, and per-query lists that do not match the query set. Empty means valid."""
    errors: list[str] = []
    entries: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        entries[r.suite_id, r.entry].append(r)
    suites = {}
    for suite_id in {r.suite_id for r in records}:
        try:
            suites[suite_id] = load_suite(suite_id, suites_dir)
        except (FileNotFoundError, RuntimeError) as e:
            errors.append(f"{suite_id}: {e}")
    for r in records:
        suite = suites.get(r.suite_id)
        where = f"{r.suite_id}/{r.problem} {r.entry} seed {r.seed}"
        if suite is None:
            continue
        if r.suite_sha256 != suite.sha256:
            errors.append(f"{where}: suite checksum differs from the released {suite.id}")
        if r.split != "test":
            errors.append(f"{where}: split {r.split!r}; only test records are ranked")
        try:
            n = len(suite.get(r.problem).queries[r.split])
        except KeyError as e:
            errors.append(f"{where}: {e}")
            continue
        if r.n != n or any(len(x) != n for x in (r.success, r.cost, r.length, r.plan_seconds)):
            errors.append(f"{where}: per-query results do not cover the {n} queries")
    for (suite_id, entry), rs in entries.items():
        suite = suites.get(suite_id)
        if suite is None:
            continue
        for p in suite.problems:
            seeds = {r.seed for r in rs if r.problem == p.name}
            if len(seeds) < suite.min_seeds:
                errors.append(f"{suite_id}/{p.name} {entry}: {len(seeds)} seeds, the suite needs {suite.min_seeds}")
    return errors


def reference_check(records: Sequence[Record], suites_dir: Path = DEFAULT_ROOT) -> list[str]:
    """Queries where an entry's path beats the reference cost c* by more than the suite's `reference_tol`: the
    reference is too coarse there, and the fix is a new suite version."""
    notes = []
    for r in records:
        try:
            suite = load_suite(r.suite_id, suites_dir)
        except (FileNotFoundError, RuntimeError):
            continue
        c_star = suite.get(r.problem).queries[r.split].c_star
        if c_star is None:
            continue
        tol = suite.protocol.reference_tol
        beaten = [i for i, c in enumerate(r.cost) if c is not None and c < (1 - tol) * float(c_star[i])]
        if beaten:
            notes.append(
                f"{r.suite_id}/{r.problem} {r.entry} seed {r.seed}: beats c* by > {tol:.0%} on queries {beaten}"
            )
    return notes
