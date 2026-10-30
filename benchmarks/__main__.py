"""CLI for the pygeoflood benchmark harness.

    python -m benchmarks --profile crop --warm-runs 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .results import VARIANCE_THRESHOLD_PCT
from .runner import default_work_root, repo_root, run_suite

PROFILES = ("crop", "full")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description=(
            "Time the canonical pygeoflood FIM workload (run_fim_workflow "
            "on OC1mTest, custom_Q=500) cold and warm, end to end and per stage."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="crop",
        help="crop = fixed 2000x2000 window (fast iteration); "
        "full = the full 16.0M-pixel tile (headline numbers)",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=2,
        help="warm (in-process) repetitions after the cold run (default: 2)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: benchmarks/results/<profile>-<utc>)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="scratch directory for DEM copies and pipeline outputs "
        "(default: %s)" % default_work_root(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.outdir or root / "benchmarks" / "results" / f"{args.profile}-{stamp}"

    suite = run_suite(
        args.profile,
        out_dir,
        warm_runs=args.warm_runs,
        work_root=args.workdir,
    )

    print()
    print("=== benchmark complete ===")
    for result in suite["results"]:
        print(f"  {result['mode']:<5} run {result['iteration']}: {result['wall_clock_s']:.2f} s")
    variance = suite["variance"]
    if variance:
        verdict = "within" if variance["within_threshold"] else "EXCEEDS"
        print(
            f"  warm-run spread: {variance['spread_pct']:.2f}% — {verdict} "
            f"the {VARIANCE_THRESHOLD_PCT:.0f}% guidance"
        )
    else:
        print("  warm-run spread: n/a (fewer than two warm runs)")
    print(f"  results written to: {suite['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
