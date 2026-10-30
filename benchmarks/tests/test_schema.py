"""Schema tests for the benchmark harness.

These run in CI as the harness's own unit test (spec deliverable D1
verification gate). They import only the schema and serialization
modules — no pygeoflood, no geo stack — so they pass anywhere, fast.
"""

from benchmarks import results as bench_results
from benchmarks import schema


def test_stub_result_is_schema_valid():
    assert schema.validate_result(schema.stub_result()) == []


def test_every_required_result_key_is_enforced():
    stub = schema.stub_result()
    for key in schema.REQUIRED_RESULT_KEYS:
        broken = schema.result_without_key(stub, key)
        problems = schema.validate_result(broken)
        assert problems, f"removing {key!r} was not flagged"
        assert any(key in problem for problem in problems)


def test_stage_list_matches_canonical_fim_workflow():
    # 18 active stage calls + the conditional rasterize_custom_flowline,
    # in run_fim_workflow() execution order at 3650c9b.
    assert len(schema.STAGES) == 19
    assert schema.STAGES == list(dict.fromkeys(schema.STAGES)), "stages must be unique"
    critical = {
        "apply_nonlinear_filter",  # anisodiff host
        "fill_depressions",  # WBT
        "calculate_d8_flow_direction",  # WBT
        "define_skeleton",  # get_fmm_points host (FMM point selection)
        "calculate_binary_hand",  # jit_binary_hand host
        "extract_channel_network",
        "calculate_hand",  # WBT
        "delineate_segment_catchments",  # WBT
        "calculate_src",  # process_cells host
        "calculate_flood_stage",
        "inundate",  # jit_inun host
    }
    assert critical <= set(schema.STAGES)


def test_bypassed_geodesic_path_is_documented_not_missing():
    # calculate_geodesic_distance and identify_channel_heads are commented
    # out of run_fim_workflow() upstream (issue #12); the schema must say
    # so explicitly instead of silently omitting them.
    assert schema.BYPASSED_STAGES == [
        "calculate_geodesic_distance",
        "identify_channel_heads",
    ]


def test_kernel_and_wbt_sets_match_the_six_kernels_and_six_wbt_calls():
    assert set(schema.KERNELS) == {
        "anisodiff",
        "anisodiff_cobol",  # backend-tagged row (PYGEOFLOOD_COBOL=1)
        "get_fmm_points",
        "jit_binary_hand",
        "process_cells",
        "jit_inun",
    }
    assert set(schema.WBT_TOOLS) == {
        "fill_depressions",
        "quinn_flow_accumulation",
        "d8_pointer",
        "basins",
        "elevation_above_stream",
        "watershed",
    }


def test_invalid_stage_status_is_flagged():
    stub = schema.stub_result()
    stub["stages"][0]["status"] = "bogus"
    problems = schema.validate_result(stub)
    assert any("status" in problem for problem in problems)


def test_unknown_stage_is_flagged():
    stub = schema.stub_result()
    stub["stages"].append({"name": "not_a_stage", "duration_s": 1.0, "status": "timed"})
    problems = schema.validate_result(stub)
    assert any("not_a_stage" in problem for problem in problems)


def test_manifest_required_keys_cover_the_spec():
    assert set(schema.REQUIRED_MANIFEST_KEYS) >= {
        "cpu_count",
        "ram_total_bytes",
        "wbt_binary_version",
        "numba_num_threads",
        "packages",
        "dem_sha256",
    }


def test_variance_summary_states_or_skips_explicitly():
    assert bench_results.summarize_variance([10.0]) is None
    variance = bench_results.summarize_variance([10.0, 10.5])
    assert variance["spread_pct"] == 100 * 0.5 / 10.25
    assert variance["within_threshold"] is True
    wide = bench_results.summarize_variance([10.0, 12.0])
    assert wide["within_threshold"] is False
    assert wide["spread_pct"] > wide["threshold_pct"]


def test_render_summary_is_deterministic():
    import copy

    results = [schema.stub_result(mode=mode) for mode in ("cold", "warm", "warm")]
    results[1]["wall_clock_s"] = 2.0
    results[2]["wall_clock_s"] = 2.1
    variance = bench_results.summarize_variance([2.0, 2.1])
    for result in results:
        result["variance"] = variance
    first = bench_results.render_summary(results, "t")
    second = bench_results.render_summary(copy.deepcopy(results), "t")
    assert first == second
    assert "Run-to-run variance" in first
