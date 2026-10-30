"""The committed official baseline must satisfy the harness schema and the
D1 protocol minimums.

Protocol (spec deliverable D1): at least 3 cold full-tile runs (median
reported), at least 5 warm reps per kernel on the crop, and the two
comparison variants — numba-warm (standard) and numba cache=True —
recorded explicitly. These tests run over the committed JSON in CI, so
the baseline cannot drift into an invalid or protocol-short state
silently.
"""

import json
from pathlib import Path

from benchmarks import schema

BASELINE_DIR = Path(__file__).resolve().parents[1] / "results" / "baseline-3650c9b"

PROFILE_DIRS = {
    "full": BASELINE_DIR / "full",
    "crop": BASELINE_DIR / "crop",
    "full-cache": BASELINE_DIR / "full-cache",
    "crop-cache": BASELINE_DIR / "crop-cache",
}

# Commits the baseline manifests may record: 3650c9b itself (the measured
# source) and 233ee90, the pre-rebase harness-branch head the original
# runs ran from — verified byte-identical to 3650c9b under src/
# (git diff 233ee90 3650c9b -- src/ is empty).
ALLOWED_SOURCE_COMMITS = {
    "3650c9bb66802fd67839819aec21cbe241797cd0",
    "233ee90a582c71bd9a7d71dcf0548471eea7e940",
}


def _runs(directory):
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("run-*.json"))
    ]


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def test_every_committed_baseline_run_is_schema_valid():
    for name, directory in PROFILE_DIRS.items():
        assert directory.is_dir(), f"missing baseline directory: {name}"
        for run in _runs(directory):
            problems = schema.validate_result(run)
            assert problems == [], f"{name}/{run['run_id']}: {problems}"


def test_every_committed_baseline_manifest_is_complete():
    for name, directory in PROFILE_DIRS.items():
        manifest = _manifest(directory)
        missing = [
            key
            for key in schema.REQUIRED_MANIFEST_KEYS
            if key not in manifest
        ]
        assert not missing, f"{name} manifest missing {missing}"
        assert manifest["wbt_max_procs"] == 1, (
            f"{name}: WhiteboxTools not pinned to one thread"
        )


def test_full_profile_has_three_cold_runs():
    colds = [
        run for run in _runs(PROFILE_DIRS["full"]) if run["mode"] == "cold"
    ]
    assert len(colds) >= 3


def test_crop_profile_has_five_warm_reps_per_kernel():
    warm = [
        run
        for run in _runs(PROFILE_DIRS["crop"])
        if run["mode"] == "warm"
    ]
    assert len(warm) >= 5
    counts: dict[str, int] = {}
    for run in warm:
        kernels = {entry["kernel"] for entry in run["kernels"]}
        for kernel in kernels:
            counts[kernel] = counts.get(kernel, 0) + 1
    assert counts, "no kernels observed in the crop warm runs"
    for kernel in sorted(counts):
        assert counts[kernel] >= 5, (
            f"kernel {kernel} has only {counts[kernel]} warm reps"
        )


def test_variants_are_recorded_explicitly():
    for name in ("full", "crop"):
        for run in _runs(PROFILE_DIRS[name]):
            assert not run.get("numba_cache", False), (
                f"{name}/{run['run_id']} must be the numba-warm variant"
            )
    for name in ("full-cache", "crop-cache"):
        for run in _runs(PROFILE_DIRS[name]):
            assert run.get("numba_cache") is True, (
                f"{name}/{run['run_id']} must be the cache=True variant"
            )
        assert _manifest(PROFILE_DIRS[name])["numba_cache"] is True


def test_cache_variant_manifests_record_the_pinned_source():
    for name in ("full-cache", "crop-cache"):
        manifest = _manifest(PROFILE_DIRS[name])
        assert manifest["git_commit"] in ALLOWED_SOURCE_COMMITS, (
            f"{name} ran from {manifest['git_commit']}, not the pinned "
            "3650c9b source"
        )


def test_variants_share_the_same_workload():
    """Every variant measures the same DEM bytes (and the crop profile the
    same deterministic window) — same sha, same workload."""
    full_dem = _manifest(PROFILE_DIRS["full"])["dem_sha256"]
    crop_dem = _manifest(PROFILE_DIRS["crop"])["dem_sha256"]
    crop_window = _manifest(PROFILE_DIRS["crop"])["crop"]
    assert _manifest(PROFILE_DIRS["full-cache"])["dem_sha256"] == full_dem
    assert _manifest(PROFILE_DIRS["crop-cache"])["dem_sha256"] == crop_dem
    cache_window = _manifest(PROFILE_DIRS["crop-cache"])["crop"]
    assert (
        cache_window["origin_row"],
        cache_window["origin_col"],
        cache_window["height"],
        cache_window["width"],
    ) == (
        crop_window["origin_row"],
        crop_window["origin_col"],
        crop_window["height"],
        crop_window["width"],
    )
