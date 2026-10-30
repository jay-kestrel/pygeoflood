"""Re-render a results directory's SUMMARY.md over ALL of its runs.

Topping up a baseline (spec D1 protocol: >=3 cold full-tile runs for the
median, >=5 warm reps per kernel on the crop) means several harness
invocations write into one results directory — and each invocation's
SUMMARY.md covers only its own runs. This tool re-renders the summary
over every run-*.json in the directory, recomputing run-to-run variance
across all warm runs, so the committed summary describes the whole
protocol.

Deterministic like the per-invocation writer: same runs in, same bytes
out.

Usage::

    python -m benchmarks.rollup benchmarks/results/baseline-3650c9b/full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import schema
from .results import render_summary, summarize_variance, write_summary


def load_runs(directory: Path) -> list[dict]:
    """All run-*.json in *directory*, ordered by file name (cold runs
    sort before warm runs; iterations are single-digit protocol ordinals)."""
    runs: list[dict] = []
    for path in sorted(Path(directory).glob("run-*.json")):
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    if not runs:
        raise FileNotFoundError(f"no run-*.json files in {directory}")
    return runs


def summarize_directory(directory: Path, title: str) -> list[dict]:
    """Recompute variance across ALL warm runs in *directory*, rewrite
    its SUMMARY.md over every run, and return the runs (variance updated
    in memory only — the per-run JSONs keep the values their invocation
    recorded)."""
    directory = Path(directory)
    runs = load_runs(directory)

    profiles = {run["profile"] for run in runs}
    variants = {run.get("numba_cache", False) for run in runs}
    if len(profiles) != 1 or len(variants) != 1:
        raise ValueError(
            f"{directory} mixes profiles {sorted(profiles)} or cache "
            f"variants {sorted(variants)} — one directory, one variant"
        )
    for run in runs:
        problems = schema.validate_result(run)
        if problems:
            raise ValueError(f"invalid run {run['run_id']}: {problems}")

    warm_durations = [
        run["wall_clock_s"] for run in runs if run["mode"] == "warm"
    ]
    variance = summarize_variance(warm_durations)
    for run in runs:
        run["variance"] = variance

    write_summary(runs, directory / "SUMMARY.md", title)
    return runs


def _default_title(first_run: dict) -> str:
    commit = first_run.get("manifest", {}).get("git_commit") or "unknown"
    title = (
        f"pygeoflood benchmark — {first_run['profile']} profile "
        f"(commit {commit})"
    )
    if first_run.get("numba_cache"):
        title += " — numba cache=True variant"
    return title


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.rollup",
        description=(
            "Re-render a results directory's SUMMARY.md over ALL its "
            "run-*.json files (for topped-up baselines)."
        ),
    )
    parser.add_argument("directory", type=Path, help="results directory")
    parser.add_argument("--title", default=None, help="summary title")
    args = parser.parse_args(argv)

    runs = load_runs(args.directory)
    title = args.title or _default_title(runs[0])
    summarize_directory(args.directory, title)
    print(f"rewrote {args.directory / 'SUMMARY.md'} over {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
