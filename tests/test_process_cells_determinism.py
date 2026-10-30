"""Determinism regression tests for the parallel process_cells kernel.

Regression guard for the data race where prange workers accumulated directly
into shared per-catchment arrays (tools.py), losing updates whenever one
catchment spanned multiple rows. The fix gives each worker a private per-chunk
buffer and folds the buffers serially in fixed order, so results must be
bitwise identical across runs and thread counts.
"""

import os
import sys

import numpy as np
import numba
import pytest

# While running tests, run on this source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pygeoflood.tools import process_cells  # noqa: E402

MAX_THREADS = numba.config.NUMBA_NUM_THREADS


def make_inputs(nrow=240, ncol=320):
    """Deterministic synthetic input: two catchments, each spanning many rows
    (concurrent workers contend on the same accumulator cells)."""
    seg = np.empty((nrow, ncol), dtype=np.int32)
    seg[: nrow // 2] = 100
    seg[nrow // 2 :] = 200
    ii, jj = np.mgrid[0:nrow, 0:ncol]
    hand = (0.1 + 0.4 * ((ii + jj) % 2)).astype(np.float32)
    hand[::97, ::89] = 0.0  # exercise the hand ~= 0 branch
    slope = (0.01 + 0.001 * ((ii * 3 + jj) % 7)).astype(np.float32)
    hid = np.array([[100, 0], [200, 1]], dtype=np.int32)
    heights = np.array([1.0, 2.0, 4.0])
    cell_area = 100.0
    return hand, seg, slope, hid, heights, cell_area


def numpy_reference(hand, seg, slope, hid, heights, cell_area):
    """Independent float64 NumPy reference for the per-stage totals."""
    counted = (hand[None] < heights[:, None, None]) | np.isclose(hand, 0.0)[None]
    total_per_height = counted.sum(axis=(1, 2)).astype(np.int64)
    ncatch = hid.shape[0]
    per_catch = np.zeros((len(heights), ncatch), dtype=np.int64)
    for k in range(ncatch):
        per_catch[:, k] = (counted & (seg[None] == hid[k, 0])).sum(axis=(1, 2))
    inc_bed = cell_area * np.sqrt(1.0 + slope.astype(np.float64) ** 2)
    bed = np.zeros((len(heights), ncatch))
    vol = np.zeros((len(heights), ncatch))
    surf = np.zeros((len(heights), ncatch))
    for k in range(ncatch):
        m = counted & (seg[None] == hid[k, 0])
        surf[:, k] = m.sum(axis=(1, 2)) * cell_area
        bed[:, k] = (inc_bed[None] * m).sum(axis=(1, 2))
        vol[:, k] = ((heights[:, None, None] - hand[None]) * cell_area * m).sum(axis=(1, 2))
    # conservation: per-catchment counts must sum to the per-height total
    assert per_catch.sum(axis=1).tolist() == total_per_height.tolist()
    return per_catch, surf, bed, vol


@pytest.fixture(scope="module")
def warm_kernel():
    """Compile the kernel once so per-test timings reflect execution only."""
    hand, seg, slope, hid, heights, cell_area = make_inputs(nrow=8, ncol=8)
    process_cells(hand, seg, slope, hid, heights, cell_area)


def run_kernel():
    hand, seg, slope, hid, heights, cell_area = make_inputs()
    return process_cells(hand, seg, slope, hid, heights, cell_area)


@pytest.mark.usefixtures("warm_kernel")
def test_parallel_runs_bitwise_identical():
    """Repeated multi-threaded runs must be bitwise identical."""
    reference = run_kernel()
    for _ in range(9):
        result = run_kernel()
        for ref_arr, res_arr in zip(reference, result):
            assert np.array_equal(ref_arr, res_arr), (
                "process_cells output varied between parallel runs — "
                "shared-state race regression"
            )


@pytest.mark.skipif(MAX_THREADS < 2, reason="requires at least 2 threads")
@pytest.mark.usefixtures("warm_kernel")
def test_serial_matches_parallel_bitwise():
    """The serial path must be bitwise identical to the parallel result."""
    parallel = run_kernel()

    numba.set_num_threads(1)
    try:
        serial = run_kernel()
    finally:
        numba.set_num_threads(MAX_THREADS)

    for ser_arr, par_arr in zip(serial, parallel):
        assert np.array_equal(ser_arr, par_arr), (
            "serial and parallel executions of process_cells diverge — "
            "reduction is not deterministic"
        )


@pytest.mark.usefixtures("warm_kernel")
def test_matches_numpy_reference():
    """CellCount exact; float totals match an independent float64 reference."""
    hand, seg, slope, hid, heights, cell_area = make_inputs()
    counts, surf, bed, vol = run_kernel()
    per_catch, ref_surf, ref_bed, ref_vol = numpy_reference(
        hand, seg, slope, hid, heights, cell_area
    )

    assert counts.shape == (len(heights), hid.shape[0])
    np.testing.assert_array_equal(counts, per_catch)
    np.testing.assert_allclose(surf, ref_surf, rtol=1e-2)
    np.testing.assert_allclose(bed, ref_bed, rtol=1e-2)
    np.testing.assert_allclose(vol, ref_vol, rtol=1e-2)


@pytest.mark.parametrize("nrow, ncol", [(1, 3), (5, 7)])
@pytest.mark.usefixtures("warm_kernel")
def test_tiny_grids_deterministic(nrow, ncol):
    """Grids smaller than the chunk budget must not break the partition."""
    hand, seg, slope, hid, heights, cell_area = make_inputs(nrow=nrow, ncol=ncol)
    first = process_cells(hand, seg, slope, hid, heights, cell_area)
    second = process_cells(hand, seg, slope, hid, heights, cell_area)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
