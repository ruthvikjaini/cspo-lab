"""Cross-foot the README against the measured results.

1. Recompute every aggregate in results/measured.json from the raw
   per-run rows in results/experiment.json and fail on any mismatch
   beyond rounding.
2. Check that every headline figure the README states in its
   "## Measured results" section appears in measured.json, formatted
   the way the README prints it.

Usage: python scripts/crossfoot.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_experiments import aggregate, summarise_run  # noqa: E402

TOL = 1e-3


def _close(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb  # inf - inf is nan, so compare directly
        return abs(fa - fb) <= TOL
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _compare(path, expected, actual, problems):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                _compare(f"{path}.{key}", value, actual[key], problems)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                _compare(f"{path}[{i}]", e, a, problems)
    elif not _close(expected, actual):
        problems.append(f"{path}: {expected!r} != {actual!r}")


def _collect_numbers(value, numbers):
    if isinstance(value, float):
        for fmt in ("{:.3f}", "{:.2f}", "{:.1f}", "{:.2e}", "{:.3e}",
                    "{:.0e}"):
            numbers.add(fmt.format(value))
    elif isinstance(value, int):
        numbers.add(str(value))
    elif isinstance(value, dict):
        for v in value.values():
            _collect_numbers(v, numbers)
    elif isinstance(value, list):
        for v in value:
            _collect_numbers(v, numbers)


def main() -> int:
    experiment = json.loads((RESULTS / "experiment.json").read_text())
    measured = json.loads((RESULTS / "measured.json").read_text())

    runs = experiment["runs"]
    summaries = [summarise_run(r) for r in runs]
    recomputed_runs = summaries
    recomputed_methods = aggregate(summaries)

    problems = []
    _compare("runs", recomputed_runs, measured["runs"], problems)
    _compare("methods", recomputed_methods, measured["methods"], problems)

    # README figures: every number printed in a table in the
    # "## Measured results" section must appear in measured.json.
    # Numbers elsewhere (paper values, hyperparameters, step counts)
    # are not checked: they come from the paper or the setup.
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Measured results", 1)[1]
    readme_section = section.split("\n## ", 1)[0]
    numbers = set()
    _collect_numbers(measured, numbers)
    claimed = re.findall(r"\|\s*(-?[\d.]+(?:e-?\d+)?)\s*\|", readme_section)
    small_ints = {str(i) for i in range(10)}
    missing = [c for c in claimed if c not in numbers
               and c not in small_ints]
    if missing:
        problems.append("README table figures not found in measured.json: "
                        f"{sorted(set(missing))}")
    if problems:
        print("CROSSFOOT FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("crossfoot OK: measured.json recomputes from experiment.json "
          "and README figures match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
