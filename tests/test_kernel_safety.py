"""Determinism and bounds-safety regression tests for the numba kernels.

Covers the tools.py kernel fixes:

- get_fmm_points: shared-list append inside prange (thread-unsafe) ->
  per-label selection flags written from prange, folded serially in fixed
  order. Outputs must be bitwise identical across runs and thread counts.
- fast_marching: the except-ValueError branch never assigned travel_time ->
  UnboundLocalError (or, worse, reuse of the previous basin's array). It now
  assigns the same NaN array the IOError branch uses. Additionally,
  geodesic_distance inherited the integer basin dtype, so even the happy
  path crashed with OverflowError before any marching happened.
- jit_channel_heads: wrote past max_channel_heads before warning -> the
  write is guarded, an overflow flag is set, and the warning prints once.
- process_cells: searchsorted result used as an index without validating
  membership -> values absent from hid_dict_array (background/nodata pixels)
  aliased a wrong catchment or wrote past the buffer. Unknown values are now
  excluded; output columns stay in the caller's hydroids order even when
  hid_dict_array is unsorted.
- get_channel_heads: bin_edges[2] raised IndexError when the skeleton had
  fewer than four connected components (the size histogram is degenerate) ->
  fallback threshold for small component counts, and an early return for an
  empty skeleton.
- get_WhiteboxTools: an explicit whitebox_exe_dir argument was
  unconditionally overwritten by the WBT_PATH environment variable -> the
  explicit argument wins; the env var is only a fallback.
- jit_binary_hand: the forward chamfer sweep ran under prange while reading
  same-row and already-written neighbours (a data race); both sweeps are
  inherently sequential and now run serially.

Determinism contracts follow the pattern of test_process_cells_determinism.py:
results must be bitwise identical across repeated runs and thread counts.
"""

import os
import sys

import numba
import numpy as np
import pytest

# While running tests, run on this source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import skfmm  # noqa: E402
from pygeoflood import tools  # noqa: E402
from pygeoflood.tools import (  # noqa: E402
    get_fmm_points,
    fast_marching,
    jit_binary_hand,
    process_cells,
)

MAX_THREADS = numba.config.NUMBA_NUM_THREADS


# ---------------------------------------------------------------------------
# C1: get_fmm_points
# ---------------------------------------------------------------------------

def make_fmm_inputs():
    """Basins 1-4 covering 50%, 30%, 10%, 10% of a 10x10 grid, one outlet
    each. With area_threshold=20 and basin_elements=5, labels 1 and 2 must
    be selected (basins 3 and 4 fail the area threshold)."""
    basins = np.zeros((10, 10), dtype=np.int32)
    basins[:, :5] = 1
    basins[:, 5:8] = 2
    basins[:, 8:9] = 3
    basins[:, 9:] = 4
    # outlets rows are (row, col); values are carried through unchanged
    outlets = np.array([[0, 1, 2, 3], [4, 5, 8, 9]], dtype=np.int64)
    return basins, outlets


def fmm_reference(basins, outlets, basin_elements, area_threshold):
    """Pure-numpy, single-threaded reference for get_fmm_points."""
    fmmY, fmmX = [], []
    ravel = basins.ravel()
    n_pixels = basins.size
    for label in range(outlets.shape[1]):
        numelements = int(np.sum(ravel == label + 1))
        percent = numelements * 100.00001 / n_pixels
        if percent > area_threshold and numelements > basin_elements:
            fmmY.append(outlets[0, label])
            fmmX.append(outlets[1, label])
    return np.array([fmmY, fmmX])


def test_fmm_points_selects_expected_labels():
    basins, outlets = make_fmm_inputs()
    result = get_fmm_points(basins, outlets, basin_elements=5, area_threshold=20)
    expected = fmm_reference(basins, outlets, 5, 20)
    assert result.shape == (2, 2)  # labels 1 and 2 selected, ascending order
    np.testing.assert_array_equal(result, expected)


def test_fmm_points_empty_selection():
    """A threshold no basin passes must yield an empty (2, 0) selection."""
    basins, outlets = make_fmm_inputs()
    result = get_fmm_points(basins, outlets, basin_elements=5, area_threshold=99)
    assert result.shape == (2, 0)


def test_fmm_points_bitwise_identical_across_runs():
    basins, outlets = make_fmm_inputs()
    reference = get_fmm_points(basins, outlets, 5, 20)
    for _ in range(9):
        result = get_fmm_points(basins, outlets, 5, 20)
        assert np.array_equal(reference, result), (
            "get_fmm_points output varied between parallel runs — "
            "shared-list append race regression"
        )


@pytest.mark.skipif(MAX_THREADS < 2, reason="requires at least 2 threads")
def test_fmm_points_serial_matches_parallel_bitwise():
    basins, outlets = make_fmm_inputs()
    parallel = get_fmm_points(basins, outlets, 5, 20)

    numba.set_num_threads(1)
    try:
        serial = get_fmm_points(basins, outlets, 5, 20)
    finally:
        numba.set_num_threads(MAX_THREADS)

    assert np.array_equal(serial, parallel), (
        "serial and parallel executions of get_fmm_points diverge — "
        "output order is not deterministic"
    )


# ---------------------------------------------------------------------------
# C2: fast_marching
# ---------------------------------------------------------------------------

def fmm_arrays():
    basins = np.ones((8, 8), dtype=np.int32)
    mfd_fac = np.ones((8, 8))
    cost = np.ones((8, 8))
    return basins, mfd_fac, cost


def test_fast_marching_valueerror_fails_fast(monkeypatch):
    """skfmm raising ValueError must surface as a RuntimeError with basin
    context (merged fail-fast contract), never reuse a previous basin's
    travel-time array or silently drop the basin."""

    def raise_value_error(*args, **kwargs):
        raise ValueError("simulated degenerate phi")

    monkeypatch.setattr(skfmm, "travel_time", raise_value_error)
    basins, mfd_fac, cost = fmm_arrays()
    start = np.array([[4], [4]])

    with pytest.raises(
        RuntimeError, match="skfmm travel_time failed for basin"
    ):
        fast_marching(start, basins, mfd_fac, cost)


def test_fast_marching_no_start_points():
    """An empty start-point set must return an all-NaN array instead of
    crashing on the (now removed) post-loop assignment."""
    basins, mfd_fac, cost = fmm_arrays()
    start = np.empty((2, 0), dtype=np.int64)
    out = fast_marching(start, basins, mfd_fac, cost)
    assert out.shape == (8, 8)
    assert np.all(np.isnan(out))


def test_fast_marching_normal_path_small_basin():
    """The success path still produces finite travel times radiating from
    the seed cell."""
    basins, mfd_fac, cost = fmm_arrays()
    start = np.array([[4], [4]])
    out = fast_marching(start, basins, mfd_fac, cost)
    assert np.all(np.isfinite(out))
    # skfmm zeros the travel time at the zero level set between cells, not
    # at the seed cell itself, so anchor on the minimum instead of exact 0.
    assert out[4, 4] == out.min()
    assert out[4, 4] < out[4, 5] < out[7, 7]


# ---------------------------------------------------------------------------
# C3: jit_channel_heads overflow guard
# ---------------------------------------------------------------------------

def heads_inputs():
    """A single row where every cell is its own skeleton label: every cell
    is a channel head, so any max_channel_heads below 6 overflows."""
    labeled = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    skeleton_gridded_array = np.ones((1, 6), dtype=np.int64)
    geodesic = np.arange(6, dtype=np.float64).reshape(1, 6)
    return labeled, skeleton_gridded_array, geodesic


def test_channel_heads_bounded_when_overflowing():
    """More candidate heads than max_channel_heads must yield exactly
    max_channel_heads rows (in scan order) — never an out-of-bounds write."""
    labeled, gridded, geodesic = heads_inputs()
    result = jit_channel_heads_safe(labeled, gridded, geodesic, 2)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[0, 0], [0, 1]])


def test_channel_heads_returns_all_within_capacity():
    labeled, gridded, geodesic = heads_inputs()
    result = jit_channel_heads_safe(labeled, gridded, geodesic, 10)
    assert result.shape == (6, 2)
    np.testing.assert_array_equal(
        result, [[0, j] for j in range(6)]
    )


def test_channel_heads_overflow_deterministic():
    labeled, gridded, geodesic = heads_inputs()
    first = jit_channel_heads_safe(labeled, gridded, geodesic, 2)
    second = jit_channel_heads_safe(labeled, gridded, geodesic, 2)
    assert np.array_equal(first, second)


def jit_channel_heads_safe(labeled, gridded, geodesic, max_channel_heads):
    return tools.jit_channel_heads(
        labeled,
        gridded,
        geodesic,
        labeled.shape[0],
        labeled.shape[1],
        0,  # count_threshold
        1,  # channel_head_median_dist
        max_channel_heads,
    )


# ---------------------------------------------------------------------------
# C5: get_channel_heads degenerate histograms
# ---------------------------------------------------------------------------

def make_skeleton(k, shape=(12, 12)):
    """k disjoint 2x2 skeleton blocks; increasing geodesic distance so each
    block's maximum-distance cell is its only channel head."""
    skel = np.zeros(shape, dtype=np.int32)
    positions = [
        (1, 1), (1, 7), (7, 1), (7, 7), (4, 4),
        (10, 2), (2, 10), (10, 10), (6, 10), (10, 6),
    ]
    for r, c in positions[:k]:
        skel[r : r + 2, c : c + 2] = 1
    geo = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    return skel, geo


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_get_channel_heads_small_component_counts(k):
    """Fewer than four components made the size histogram degenerate and
    crashed (bin_edges[2] IndexError; k == 0 died even earlier). Merged
    contract: an empty skeleton returns (None, None); 1-3 components fall
    back to the median component size as the threshold, and each block
    still yields exactly one head with no exception."""
    skel, geo = make_skeleton(k)
    ch_rows, ch_cols = tools.get_channel_heads(
        skel, geo, channel_head_median_dist=1, max_channel_heads=100
    )
    if k == 0:
        assert ch_rows is None and ch_cols is None
        return
    assert ch_rows.shape == ch_cols.shape
    assert ch_rows.ndim == 1
    assert len(ch_rows) == k


def test_get_channel_heads_many_components_still_works():
    """Regression guard: four or more components take the histogram path,
    which must keep working."""
    skel, geo = make_skeleton(6)
    ch_rows, ch_cols = tools.get_channel_heads(
        skel, geo, channel_head_median_dist=1, max_channel_heads=100
    )
    assert ch_rows.shape == ch_cols.shape
    assert ch_rows.ndim == 1


# ---------------------------------------------------------------------------
# C4: process_cells input validation (merged Rust contract)
# ---------------------------------------------------------------------------

def make_foreign_inputs(hid):
    nrow, ncol = 40, 60
    seg = np.zeros((nrow, ncol), dtype=np.int32)  # 0 = background, foreign
    seg[: nrow // 3] = 100
    seg[nrow // 3 : 2 * nrow // 3] = 200
    seg[2 * nrow // 3 :] = 999  # a hydroid absent from hid
    ii, jj = np.mgrid[0:nrow, 0:ncol]
    hand = (0.1 + 0.4 * ((ii + jj) % 2)).astype(np.float32)
    hand[::7, ::11] = 0.0  # exercise the hand ~= 0 branch
    slope = (0.01 + 0.001 * ((ii * 3 + jj) % 7)).astype(np.float32)
    heights = np.array([1.0, 2.0])
    return hand, seg, slope, hid, heights, 100.0


def process_reference(hand, seg, slope, hid, heights, cell_area):
    """float64 NumPy reference counting only cells whose hydroid is known."""
    counted = (hand[None] < heights[:, None, None]) | np.isclose(hand, 0.0)[None]
    counted &= np.isin(seg, hid[:, 0])[None]
    nheight, ncatch = len(heights), len(hid)
    counts = np.zeros((nheight, ncatch), dtype=np.int32)
    surf = np.zeros((nheight, ncatch), dtype=np.float32)
    bed = np.zeros((nheight, ncatch), dtype=np.float32)
    vol = np.zeros((nheight, ncatch), dtype=np.float32)
    inc_bed = cell_area * np.sqrt(1.0 + slope.astype(np.float64) ** 2)
    for k in range(ncatch):
        m = counted & (seg[None] == hid[k, 0])
        counts[:, k] = m.sum(axis=(1, 2))
        surf[:, k] = m.sum(axis=(1, 2)) * cell_area
        bed[:, k] = (inc_bed[None] * m).sum(axis=(1, 2))
        vol[:, k] = (
            (heights[:, None, None] - hand[None]) * cell_area * m
        ).sum(axis=(1, 2))
    return counts, surf, bed, vol


def make_valid_inputs():
    """Sorted, fully-known hydroids — the merged Rust kernel's input contract."""
    hid = np.array([[100, 0], [200, 1]], dtype=np.int32)
    hand, seg, slope, hid_in, heights, cell_area = make_foreign_inputs(hid)
    seg[seg == 999] = 200  # fold foreign rows into known catchment 200
    return hand, seg, slope, hid_in, heights, cell_area


def test_foreign_hydroid_rejected():
    """Active cells with a hydroid absent from the lookup table (foreign
    ids) abort the kernel: the merged Rust contract fails fast instead of
    aliasing or silently dropping their pixels."""
    hid = np.array([[100, 0], [200, 1]], dtype=np.int32)
    hand, seg, slope, hid_in, heights, cell_area = make_foreign_inputs(hid)
    with pytest.raises(ValueError, match="outside the lookup table"):
        process_cells(hand, seg, slope, hid_in, heights, cell_area)


def test_unsorted_hydroids_rejected():
    """The merged Rust kernel requires hydroids sorted ascending; unsorted
    input is rejected instead of silently reordering output columns."""
    hid = np.array([[200, 0], [100, 1]], dtype=np.int32)  # unsorted keys
    hand, seg, slope, hid_in, heights, cell_area = make_foreign_inputs(hid)
    with pytest.raises(ValueError, match="finite and sorted"):
        process_cells(hand, seg, slope, hid_in, heights, cell_area)


def test_all_unknown_values_rejected():
    hid = np.array([[100, 0], [200, 1]], dtype=np.int32)
    hand, seg, slope, hid_in, heights, cell_area = make_foreign_inputs(hid)
    seg[:] = 999  # nothing matches
    with pytest.raises(ValueError, match="outside the lookup table"):
        process_cells(hand, seg, slope, hid_in, heights, cell_area)


def test_process_cells_bitwise_identical_across_runs():
    hand, seg, slope, hid_in, heights, cell_area = make_valid_inputs()
    expected_counts, expected_surf, expected_bed, expected_vol = process_reference(
        hand, seg, slope, hid_in, heights, cell_area
    )
    first = process_cells(hand, seg, slope, hid_in, heights, cell_area)
    first_counts, first_surf, first_bed, first_vol = first
    np.testing.assert_array_equal(first_counts, expected_counts)
    np.testing.assert_allclose(first_surf, expected_surf, rtol=1e-2)
    np.testing.assert_allclose(first_bed, expected_bed, rtol=1e-2)
    np.testing.assert_allclose(first_vol, expected_vol, rtol=1e-2)
    for _ in range(4):
        result = process_cells(hand, seg, slope, hid_in, heights, cell_area)
        for ref_arr, res_arr in zip(first, result):
            assert np.array_equal(ref_arr, res_arr)


# ---------------------------------------------------------------------------
# C6: get_WhiteboxTools explicit argument vs WBT_PATH
# ---------------------------------------------------------------------------

def test_get_wbt_explicit_dir_wins_over_env(monkeypatch):
    monkeypatch.setenv("WBT_PATH", "/env/wbt/path")
    wbt = tools.get_WhiteboxTools(verbose=False, whitebox_exe_dir="/explicit/wbt")
    assert os.path.normpath(wbt.exe_path) == os.path.normpath("/explicit/wbt")


def test_get_wbt_env_var_fallback(monkeypatch):
    monkeypatch.setenv("WBT_PATH", "/env/wbt/path")
    wbt = tools.get_WhiteboxTools(verbose=False)
    assert os.path.normpath(wbt.exe_path) == os.path.normpath("/env/wbt/path")


def test_get_wbt_default_when_no_arg_no_env(monkeypatch):
    monkeypatch.delenv("WBT_PATH", raising=False)
    wbt = tools.get_WhiteboxTools(verbose=False)
    assert os.path.normpath(wbt.exe_path) == _default_wbt_path()


def _default_wbt_path():
    from whitebox import WhiteboxTools

    return WhiteboxTools().exe_path


# ---------------------------------------------------------------------------
# M8: jit_binary_hand chamfer sweeps
# ---------------------------------------------------------------------------

def make_hand_inputs(nrow=24, ncol=32):
    """DEM rising away from a vertical stream column, so cells far from the
    stream are unambiguously above their allocated stream elevation."""
    rng = np.random.default_rng(7)
    stream_col = ncol // 2
    ii, jj = np.mgrid[0:nrow, 0:ncol]
    dem = (100.0 + 2.0 * np.abs(jj - stream_col) + rng.random((nrow, ncol))).astype(
        np.float32
    )
    flowline = np.zeros((nrow, ncol), dtype=np.int32)
    flowline[:, stream_col] = 1
    nodata = -32768.0
    return dem, flowline, nodata


def test_binary_hand_bitwise_identical_across_runs():
    dem, flowline, nodata = make_hand_inputs()
    reference = jit_binary_hand(dem, flowline, nodata)
    for _ in range(9):
        result = jit_binary_hand(dem, flowline, nodata)
        assert np.array_equal(reference, result), (
            "jit_binary_hand output varied between runs — chamfer sweep "
            "neighbor-read race regression"
        )


@pytest.mark.skipif(MAX_THREADS < 2, reason="requires at least 2 threads")
def test_binary_hand_serial_matches_multithread():
    dem, flowline, nodata = make_hand_inputs()
    parallel = jit_binary_hand(dem, flowline, nodata)

    numba.set_num_threads(1)
    try:
        serial = jit_binary_hand(dem, flowline, nodata)
    finally:
        numba.set_num_threads(MAX_THREADS)

    assert np.array_equal(serial, parallel), (
        "serial and multi-threaded executions of jit_binary_hand diverge — "
        "the chamfer sweeps must be sequential"
    )


def test_binary_hand_sanity():
    dem, flowline, nodata = make_hand_inputs()
    result = jit_binary_hand(dem, flowline, nodata)
    # stream cells allocate themselves: binary HAND is 1 there
    assert np.all(result[flowline == 1] == 1)
    # cells well above their allocated stream elevation are 0
    assert np.any(result == 0)
