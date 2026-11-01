import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from ntfields.bench.report import reference_check, report, validate
from ntfields.bench.run import RunConfig, run_suite
from ntfields.bench.schema import read_records
from ntfields.bench.suite import DEFAULT_ROOT, available_suites, load_suite


@dataclass
class List:
    """List the released suites."""

    suites_dir: Path = DEFAULT_ROOT


@dataclass
class Run(RunConfig):
    """Train (or reuse) and evaluate entries on a suite, appending records to --out."""


@dataclass
class Report:
    """Markdown tables from record files, with a note wherever an entry beats a suite's reference."""

    records: list[Path] = field(default_factory=list)
    suites_dir: Path = DEFAULT_ROOT


@dataclass
class Validate:
    """Check record files against the released suites; exits 1 on any problem."""

    records: list[Path] = field(default_factory=list)
    suites_dir: Path = DEFAULT_ROOT


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = tyro.extras.subcommand_cli_from_dict(
        {"list": List, "run": Run, "report": Report, "validate": Validate}, args=args, prog="ntfields bench"
    )
    if isinstance(command, List):
        for name in available_suites(command.suites_dir):
            suite = load_suite(name, command.suites_dir)
            sizes = ", ".join(f"{p.name} ({len(p.queries['test'])})" for p in suite.problems)
            print(f"{suite.id}{'' if suite.leaderboard else ' [not ranked]'}: {suite.description} | {sizes}")
    elif isinstance(command, Run):
        run_suite(command)
    elif isinstance(command, Report):
        records = read_records(*command.records)
        print(report(records))
        for note in reference_check(records, command.suites_dir):
            print(f"note: {note}")
    else:
        errors = validate(read_records(*command.records), command.suites_dir)
        print("\n".join(errors) or "valid")
        if errors:
            sys.exit(1)
