"""Tests for the roll-up summary tool used to top up baselines."""

import copy
import json

import pytest

from benchmarks import results as bench_results
from benchmarks import schema
from benchmarks.rollup import load_runs, summarize_directory


def _write_run(
    directory,
    run_id,
    mode,
    iteration,
    seconds,
    profile="crop",
    numba_cache=None,
):
    run = schema.stub_result(profile=profile, mode=mode)
    run["run_id"] = run_id
    run["iteration"] = iteration
    run["wall_clock_s"] = seconds
    if numba_cache is not None:
        run["numba_cache"] = numba_cache
    bench_results.write_run_json(run, directory / f"run-{run_id}.json")


def test_load_runs_requires_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path)


def test_summarize_directory_covers_all_runs(tmp_path):
    _write_run(tmp_path, "crop-cold-1", "cold", 1, 25.0)
    _write_run(tmp_path, "crop-warm-1", "warm", 1, 15.0)
    _write_run(tmp_path, "crop-warm-2", "warm", 2, 16.0)
    runs = summarize_directory(tmp_path, "test title")
    assert [run["run_id"] for run in runs] == [
        "crop-cold-1",
        "crop-warm-1",
        "crop-warm-2",
    ]
    summary = (tmp_path / "SUMMARY.md").read_text()
    assert "test title" in summary
    # End-to-end table lists every run, not just the last invocation's.
    assert "| cold | 1 |" in summary
    assert "| warm | 2 |" in summary
    # Variance recomputed across ALL warm runs of the directory.
    variance = bench_results.summarize_variance([15.0, 16.0])
    assert f"{variance['spread_pct']:.2f}%" in summary


def test_summarize_directory_is_deterministic(tmp_path):
    _write_run(tmp_path, "crop-cold-1", "cold", 1, 25.0)
    _write_run(tmp_path, "crop-warm-1", "warm", 1, 15.0)
    _write_run(tmp_path, "crop-warm-2", "warm", 2, 16.0)
    summarize_directory(tmp_path, "t")
    first = (tmp_path / "SUMMARY.md").read_text()
    summarize_directory(tmp_path, "t")
    assert first == (tmp_path / "SUMMARY.md").read_text()


def test_summarize_directory_rejects_mixed_variants(tmp_path):
    _write_run(tmp_path, "crop-cold-1", "cold", 1, 25.0, numba_cache=False)
    _write_run(tmp_path, "crop-warm-1", "warm", 1, 15.0, numba_cache=True)
    with pytest.raises(ValueError, match="one directory, one variant"):
        summarize_directory(tmp_path, "t")


def test_summarize_directory_rejects_mixed_profiles(tmp_path):
    _write_run(tmp_path, "crop-cold-1", "cold", 1, 25.0, profile="crop")
    _write_run(
        tmp_path, "full-cold-1", "cold", 1, 80.0, profile="full"
    )
    with pytest.raises(ValueError, match="profiles"):
        summarize_directory(tmp_path, "t")


def test_summarize_directory_rejects_schema_invalid_runs(tmp_path):
    _write_run(tmp_path, "crop-warm-1", "warm", 1, 15.0)
    path = tmp_path / "run-crop-warm-1.json"
    broken = json.loads(path.read_text())
    broken["profile"] = "nope"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid run"):
        summarize_directory(tmp_path, "t")


def test_rollup_preserves_recorded_variance_in_run_jsons(tmp_path):
    # Only the summary is rewritten; the per-run JSONs keep the variance
    # their own invocation recorded (committed run files stay untouched).
    _write_run(tmp_path, "crop-warm-1", "warm", 1, 15.0)
    before = copy.deepcopy((tmp_path / "run-crop-warm-1.json").read_text())
    summarize_directory(tmp_path, "t")
    assert (tmp_path / "run-crop-warm-1.json").read_text() == before
