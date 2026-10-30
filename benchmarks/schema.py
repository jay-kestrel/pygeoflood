"""Result schema contract for the pygeoflood benchmark harness.

This module is the single source of truth for what a benchmark result
contains. It must stay importable with only the standard library so the
schema unit tests (``benchmarks/tests/test_schema.py``) can run in any
environment, including CI before the heavy geo stack is exercised.
"""

from __future__ import annotations

import copy
import time

SCHEMA_VERSION = 1

# Canonical stage order exactly as executed by pyGeoFlood.run_fim_workflow()
# (src/pygeoflood/pygeoflood.py at commit 3650c9b, lines 2714-2735).
# rasterize_custom_flowline is conditional upstream: it only runs when
# custom_flowline_path exists, which is not the case for the canonical
# workload (findings art_OZ9DYcjn section 6.1).
STAGES = [
    "apply_nonlinear_filter",
    "calculate_slope",
    "calculate_curvature",
    "fill_depressions",
    "calculate_mfd_flow_accumulation",
    "calculate_d8_flow_direction",
    "find_outlets",
    "delineate_basins",
    "define_skeleton",
    "find_endpoints",
    "calculate_binary_hand",
    "rasterize_custom_flowline",
    "extract_channel_network",
    "calculate_hand",
    "segment_channel_network",
    "delineate_segment_catchments",
    "calculate_src",
    "calculate_flood_stage",
    "inundate",
]

# Present in the upstream source but commented out of run_fim_workflow()
# at 3650c9b (the scikit-fmm geodesic path, unreliable per upstream
# issue #12). They are never executed by the canonical workload; they are
# listed here so the baseline shows they were bypassed by design, not
# accidentally omitted.
BYPASSED_STAGES = [
    "calculate_geodesic_distance",
    "identify_channel_heads",
]

# Python/numba kernels inside tools.py timed separately from their host
# stage, so a per-stage port comparison (spec deliverable D4) can separate
# kernel time from glue and I/O.
KERNELS = [
    "anisodiff",
    "get_fmm_points",
    "jit_binary_hand",
    "process_cells",
    "jit_inun",
]

# WhiteboxTools subprocess entry points used by the FIM workflow. The
# harness times every WBT call it observes; this list documents the six
# expected in the canonical workload (findings section 6.1).
WBT_TOOLS = [
    "fill_depressions",
    "quinn_flow_accumulation",
    "d8_pointer",
    "basins",
    "elevation_above_stream",
    "watershed",
]

MODES = ("cold", "warm")

STAGE_STATUSES = ("timed", "skipped", "failed")

IO_KINDS = ("raster_read", "raster_write", "csv_write", "vector_write")

REQUIRED_RESULT_KEYS = (
    "schema_version",
    "run_id",
    "profile",
    "mode",
    "iteration",
    "started_utc",
    "wall_clock_s",
    "stages",
    "wbt_calls",
    "kernels",
    "io",
    "manifest",
    "variance",
)

REQUIRED_MANIFEST_KEYS = (
    "generated_utc",
    "platform",
    "python_version",
    "cpu_count",
    "ram_total_bytes",
    "numba_num_threads",
    "numba_env_threads",
    "wbt_binary_version",
    "wbt_package_version",
    "wbt_max_procs",
    "packages",
    "git_commit",
    "dem_sha256",
    "dem",
    "crop",
)

REQUIRED_DEM_KEYS = (
    "path",
    "shape",
    "crs",
    "resolution",
    "cell_count",
)

REQUIRED_STAGE_KEYS = ("name", "duration_s", "status")

REQUIRED_WBT_CALL_KEYS = ("tool", "stage", "duration_s")

REQUIRED_KERNEL_KEYS = ("kernel", "stage", "duration_s")

REQUIRED_IO_KEYS = tuple(f"{kind}_{unit}" for kind in IO_KINDS for unit in ("count", "seconds"))

REQUIRED_CROP_KEYS = ("origin_row", "origin_col", "height", "width")

PROFILE_FULL = "full"
PROFILE_CROP = "crop"
CROP_SIZE = 2000
PROFILES = (PROFILE_CROP, PROFILE_FULL)


def validate_result(result: dict) -> list[str]:
    """Check a result dict against this module's contract.

    Returns a list of human-readable problems; an empty list means the
    result is schema-valid. Never raises.
    """
    problems: list[str] = []
    for key in REQUIRED_RESULT_KEYS:
        if key not in result:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems

    if result["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version {result['schema_version']!r} != {SCHEMA_VERSION}"
        )
    if result["mode"] not in MODES:
        problems.append(f"mode {result['mode']!r} not in {MODES}")
    if result["profile"] not in PROFILES:
        problems.append(f"profile {result['profile']!r} not in {PROFILES}")
    if not isinstance(result["wall_clock_s"], (int, float)):
        problems.append("wall_clock_s must be numeric")

    stages = result["stages"]
    if not isinstance(stages, list):
        problems.append("stages must be a list")
    else:
        names = []
        for stage in stages:
            for key in REQUIRED_STAGE_KEYS:
                if key not in stage:
                    problems.append(f"stage missing key {key!r}: {stage!r}")
            if stage.get("status") not in STAGE_STATUSES:
                problems.append(
                    f"stage status {stage.get('status')!r} not in {STAGE_STATUSES}"
                )
            names.append(stage.get("name"))
        if set(names) - set(STAGES):
            problems.append(f"unknown stage names: {sorted(set(names) - set(STAGES))}")
        if len(names) != len(set(names)):
            problems.append("duplicate stage entries")

    wbt_calls = result["wbt_calls"]
    if not isinstance(wbt_calls, list):
        problems.append("wbt_calls must be a list")
    else:
        for call in wbt_calls:
            for key in REQUIRED_WBT_CALL_KEYS:
                if key not in call:
                    problems.append(f"wbt_call missing key {key!r}: {call!r}")
            if "tool" in call and call["tool"] not in WBT_TOOLS:
                problems.append(f"unexpected WBT tool: {call['tool']!r}")

    kernels = result["kernels"]
    if not isinstance(kernels, list):
        problems.append("kernels must be a list")
    else:
        for kernel in kernels:
            for key in REQUIRED_KERNEL_KEYS:
                if key not in kernel:
                    problems.append(f"kernel missing key {key!r}: {kernel!r}")
            if "kernel" in kernel and kernel["kernel"] not in KERNELS:
                problems.append(f"unknown kernel: {kernel['kernel']!r}")

    io = result["io"]
    if not isinstance(io, dict):
        problems.append("io must be a dict")
    else:
        for key in REQUIRED_IO_KEYS:
            if key not in io:
                problems.append(f"io missing key: {key}")

    manifest = result["manifest"]
    if not isinstance(manifest, dict):
        problems.append("manifest must be a dict")
    else:
        for key in REQUIRED_MANIFEST_KEYS:
            if key not in manifest:
                problems.append(f"manifest missing key: {key}")
        dem = manifest.get("dem")
        if isinstance(dem, dict):
            for key in REQUIRED_DEM_KEYS:
                if key not in dem:
                    problems.append(f"manifest.dem missing key: {key}")
        elif manifest.get("dem") is not None:
            problems.append("manifest.dem must be a dict or null")
        crop = manifest.get("crop")
        if crop is not None:
            if not isinstance(crop, dict):
                problems.append("manifest.crop must be a dict or null")
            else:
                for key in REQUIRED_CROP_KEYS:
                    if key not in crop:
                        problems.append(f"manifest.crop missing key: {key}")

    return problems


def stub_result(profile: str = PROFILE_CROP, mode: str = "warm") -> dict:
    """Build a minimal schema-valid result for tests and examples."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stages = [
        {
            "name": name,
            "duration_s": 0.0 if name != "rasterize_custom_flowline" else None,
            "status": "skipped" if name == "rasterize_custom_flowline" else "timed",
        }
        for name in STAGES
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "stub-run",
        "profile": profile,
        "mode": mode,
        "iteration": 1,
        "started_utc": now,
        "wall_clock_s": 1.0,
        "stages": stages,
        "wbt_calls": [
            {"tool": "fill_depressions", "stage": "fill_depressions", "duration_s": 0.5}
        ],
        "kernels": [
            {"kernel": "process_cells", "stage": "calculate_src", "duration_s": 0.25}
        ],
        "io": {f"{kind}_{unit}": 0 for kind in IO_KINDS for unit in ("count", "seconds")},
        "manifest": {
            key: None for key in REQUIRED_MANIFEST_KEYS
        }
        | {
            "dem": {key: None for key in REQUIRED_DEM_KEYS},
            "crop": {key: None for key in REQUIRED_CROP_KEYS},
        },
        "variance": None,
    }


def result_without_key(result: dict, key: str) -> dict:
    """Copy of *result* with one top-level key removed (test helper)."""
    broken = copy.deepcopy(result)
    broken.pop(key, None)
    return broken
