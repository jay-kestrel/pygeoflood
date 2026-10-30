"""Suite orchestration: one cold run (fresh process, captures the numba
JIT compile tax) plus repeated warm runs per profile, then JSON and
markdown output with the run-to-run variance stated explicitly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from . import schema
from .instrumentation import PipelineProbe
from .manifest import collect_manifest
from .profiles import PreparedProfile, prepare_profile
from .results import summarize_variance, write_run_json, write_summary

# The canonical workload: findings art_OZ9DYcjn section 6.1 —
# run_fim_workflow on OC1mTest with a 500 cfs custom discharge.
DEFAULT_CONFIG = {"calculate_flood_stage": {"custom_Q": 500}}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_work_root() -> Path:
    # Outside the checkout so benchmark runs never pollute the repo.
    return repo_root().parent / "pgf-bench-runs"


def build_factory(
    prepared: PreparedProfile, config: dict | None = None
) -> Callable[[], Any]:
    """Fresh pyGeoFlood instance per run, in the canonical configuration."""
    from pygeoflood import pyGeoFlood

    def factory() -> Any:
        pgf = pyGeoFlood(
            dem_path=str(prepared.dem_path),
            project_dir=str(prepared.run_dir),
        )
        pgf.flowline_path = str(prepared.flowline_path)
        pgf.catchment_path = str(prepared.catchment_path)
        pgf.config = config if config is not None else DEFAULT_CONFIG
        return pgf

    return factory


def flatten_io(io: dict[str, dict[str, float]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for kind in schema.IO_KINDS:
        flattened[f"{kind}_count"] = io[kind]["count"]
        flattened[f"{kind}_seconds"] = io[kind]["seconds"]
    return flattened


def run_once(
    prepared: PreparedProfile,
    factory: Callable[[], Any],
    manifest: dict,
    mode: str,
    iteration: int,
) -> dict:
    """One end-to-end workflow run under the probe; raises if it fails."""
    from pygeoflood import pyGeoFlood  # noqa: F401 — factory resolves it lazily

    pgf = factory()
    probe = PipelineProbe()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.perf_counter()
    try:
        with probe.install(pgf):
            pgf.run_fim_workflow()
    finally:
        # wrappers come off even when the workflow raises
        probe.remove()
    wall_clock = time.perf_counter() - start

    result = {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": f"{prepared.profile}-{mode}-{iteration}",
        "profile": prepared.profile,
        "mode": mode,
        "iteration": iteration,
        "started_utc": started_utc,
        "wall_clock_s": wall_clock,
        "stages": probe.as_stages(),
        "wbt_calls": probe.as_wbt_calls(),
        "kernels": probe.as_kernels(),
        "io": flatten_io(probe.io),
        "manifest": manifest,
        "variance": None,
    }
    problems = schema.validate_result(result)
    if problems:
        raise RuntimeError(f"harness produced an invalid result: {problems}")
    return result


def run_suite(
    profile: str,
    out_dir: Path,
    warm_runs: int = 2,
    work_root: Path | None = None,
    config: dict | None = None,
) -> dict:
    """Cold run + *warm_runs* warm runs of *profile*; writes per-run JSON,
    manifest.json and SUMMARY.md into *out_dir*."""
    root = repo_root()
    work = work_root if work_root is not None else default_work_root()
    prepared = prepare_profile(profile, root, work)
    manifest = collect_manifest(prepared.dem_path, root, prepared.crop)
    factory = build_factory(prepared, config)

    results = [
        run_once(prepared, factory, manifest, mode="cold", iteration=1)
    ]
    for iteration in range(1, warm_runs + 1):
        results.append(
            run_once(prepared, factory, manifest, mode="warm", iteration=iteration)
        )

    warm_durations = [
        result["wall_clock_s"] for result in results if result["mode"] == "warm"
    ]
    variance = summarize_variance(warm_durations)
    for result in results:
        result["variance"] = variance

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        write_run_json(result, out_dir / f"run-{result['run_id']}.json")
    write_run_json(manifest, out_dir / "manifest.json")
    title = (
        f"pygeoflood benchmark — {profile} profile "
        f"(commit {manifest.get('git_commit') or 'unknown'})"
    )
    write_summary(results, out_dir / "SUMMARY.md", title)
    return {
        "results": results,
        "out_dir": out_dir,
        "manifest": manifest,
        "variance": variance,
    }
