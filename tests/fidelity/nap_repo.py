"""Access to the RaNTFields reference (the `nap` research repo): its paper runs, run through its own environment."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

NAP = Path(os.environ.get("NTFIELDS_NAP", Path(__file__).resolve().parents[3] / "nap")).resolve()
PYTHON = NAP / ".venv" / "bin" / "python"
RUNS = {  # the paper's checkpoints
    "flow_constant": ("outputs/2026-09-23/17-23-39", 400),
    "terrain_ridge": ("outputs/2026-09-25/09-29-47", 400),
    "arm": ("outputs/2026-09-15/02-52-19", 100),
}
TABLE_RESULTS = {  # the reference's closed-loop runs behind the paper's table
    "rantf": "test/sim_results/planner_comparison_20260915_035701/results.json",
    "n_eik": "test/sim_results/planner_comparison_20260915_003726/results.json",
    "n_riem": "test/sim_results/planner_comparison_20260915_004205/results.json",
}
TABLE_RUNS = {
    "rantf": RUNS["arm"],
    "n_eik": ("outputs/2026-09-14/21-37-49", 100),
    "n_riem": ("outputs/2026-09-14/22-18-41", 100),
}

requires_nap = pytest.mark.skipif(
    not (PYTHON.is_file() and (NAP / RUNS["arm"][0]).is_dir()),
    reason="set NTFIELDS_NAP to a checkout of the reference with its .venv and paper runs",
)


def reference(*args: str | Path | int) -> None:
    """Run `nap_reference.py` with `args` in the reference's environment (from its root, as its configs expect)."""
    script = Path(__file__).with_name("nap_reference.py")
    subprocess.run([str(PYTHON), str(script), *map(str, args)], cwd=NAP, check=True, capture_output=True)


def import_nap_import():
    """`benchmarks/rantfields/nap_import.py`, which rebuilds a reference run as an ntfields `TrainedField`."""
    bench = Path(__file__).resolve().parents[2] / "benchmarks" / "rantfields"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import nap_import  # type: ignore[import-not-found]

    return nap_import
