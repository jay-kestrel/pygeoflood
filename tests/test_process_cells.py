"""Native SRC accumulation against the frozen deterministic Numba oracle."""

import os
import subprocess
import sys

import numpy as np
import pytest
from reference_process_cells import process_cells as reference

from pygeoflood import _rust
from pygeoflood.tools import process_cells


def inputs(rows=67, cols=13, dtype=np.float32, height_dtype=np.float64):
    rng = np.random.default_rng(1729)
    hand = rng.uniform(-0.1, 3, (rows, cols)).astype(dtype)
    if hand.size >= 10:
        hand.flat[:10] = [
            0,
            1e-9,
            -1e-9,
            1e-8,
            -1e-8,
            1.01e-8,
            1,
            np.nan,
            np.inf,
            -np.inf,
        ]
    slope = rng.uniform(0, 1, hand.shape).astype(dtype)
    catchments = rng.choice([10, 20], hand.shape).astype(np.int32)
    # Third catchment is empty; second column is deliberately not positional.
    hydroids = np.array([[10, 90], [20, 80], [30, 70]], dtype=np.int32)
    heights = np.array([0, 0.5, 1, 2, 3], dtype=height_dtype)
    return hand, catchments, slope, hydroids, heights, 1.31


def assert_parity(args):
    expected = reference(*args)
    actual = process_cells(*args)
    for i, (candidate, golden) in enumerate(zip(actual, expected)):
        assert candidate.shape == golden.shape
        assert candidate.dtype == golden.dtype
        if i == 0:
            np.testing.assert_array_equal(candidate, golden)
        else:
            np.testing.assert_allclose(
                candidate, golden, rtol=1e-6, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 67])
@pytest.mark.parametrize(
    "dtype,height_dtype",
    [
        (np.float32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float32),
        (np.float64, np.float64),
    ],
)
def test_fixed_numba_parity(rows, dtype, height_dtype):
    assert_parity(inputs(rows=rows, dtype=dtype, height_dtype=height_dtype))


@pytest.mark.parametrize("area", [np.float32(1.31), np.float64(1.31), 2])
def test_scalar_promotion(area):
    args = list(inputs(dtype=np.float32, height_dtype=np.float32))
    args[-1] = area
    assert_parity(args)


def test_strided_inputs_and_no_mutation():
    args = list(inputs())
    args[:3] = [array[::-2, ::-2] for array in args[:3]]
    before = [array.copy() for array in args[:-1]]
    assert_parity(args)
    for actual, original in zip(args[:-1], before):
        np.testing.assert_array_equal(actual, original)


def test_empty_stages_and_empty_grid():
    args = list(inputs())
    args[4] = np.array([], dtype=np.float64)
    assert_parity(args)
    args = list(inputs(rows=0))
    args[3] = np.empty((0, 2), dtype=np.int32)
    for result in process_cells(*args):
        assert result.shape == (5, 0)


def test_searchsorted_insertion_index():
    args = list(inputs())
    args[1][:] = 15  # insertion index 1, even without an exact hydroid match
    assert_parity(args)


@pytest.mark.parametrize("bad_table", [np.array([10, 20]), np.ones((2, 3))])
def test_invalid_lookup_shape(bad_table):
    args = list(inputs())
    args[3] = bad_table
    with pytest.raises(ValueError, match="shape"):
        process_cells(*args)


def test_invalid_grid_shape():
    args = list(inputs())
    args[2] = args[2][:-1]
    with pytest.raises(ValueError, match="shapes must match"):
        process_cells(*args)


def test_out_of_range_active_hydroid_is_safe():
    args = list(inputs())
    args[1][:] = 100
    with pytest.raises(ValueError, match="outside the lookup table"):
        process_cells(*args)


@pytest.mark.parametrize("hydroids", [[20.0, 10.0], [10.0, np.nan]])
def test_native_boundary_validates_lookup(hydroids):
    grid = np.zeros((1, 1))
    with pytest.raises(ValueError, match="finite and sorted"):
        _rust.process_cells(grid, grid, grid, np.array(hydroids), np.array([1.0]), 1.0)


def test_rayon_thread_counts_are_bitwise_identical(tmp_path):
    script = """
import sys
import numpy as np
from pygeoflood.tools import process_cells
rng = np.random.default_rng(42)
hand = rng.random((67, 43), dtype=np.float32)
slope = rng.random(hand.shape, dtype=np.float32)
segments = rng.choice([10, 20], hand.shape).astype(np.int32)
hydroids = np.array([[10, 0], [20, 1]], dtype=np.int32)
heights = np.arange(0, 3, .1)
result = process_cells(hand, segments, slope, hydroids, heights, 1.31)
for i, array in enumerate(result):
    np.save(f'{sys.argv[1]}/{i}.npy', array)
"""
    reference_arrays = None
    for threads in [1, 2, 4]:
        directory = tmp_path / str(threads)
        directory.mkdir()
        subprocess.run(
            [sys.executable, "-c", script, str(directory)],
            env=dict(os.environ, RAYON_NUM_THREADS=str(threads)),
            check=True,
        )
        arrays = [(directory / f"{i}.npy").read_bytes() for i in range(4)]
        if reference_arrays is None:
            reference_arrays = arrays
        assert arrays == reference_arrays
