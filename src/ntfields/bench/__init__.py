"""Frozen benchmark suites, the evaluation protocol, the results schema, and the runner behind `ntfields bench`."""

from ntfields.bench.protocol import Outcome, Protocol, evaluate_path, spl
from ntfields.bench.recipes import RECIPES, Recipe, get_recipe, register_recipe
from ntfields.bench.report import reference_check, report, validate
from ntfields.bench.run import RunConfig, run_suite
from ntfields.bench.schema import SCHEMA_VERSION, Record, append_records, read_records
from ntfields.bench.suite import Queries, Suite, SuiteIntegrityError, SuiteProblem, available_suites, load_suite

__all__ = [
    "RECIPES",
    "SCHEMA_VERSION",
    "Outcome",
    "Protocol",
    "Queries",
    "Recipe",
    "Record",
    "RunConfig",
    "Suite",
    "SuiteIntegrityError",
    "SuiteProblem",
    "append_records",
    "available_suites",
    "evaluate_path",
    "get_recipe",
    "load_suite",
    "read_records",
    "reference_check",
    "register_recipe",
    "report",
    "run_suite",
    "spl",
    "validate",
]
