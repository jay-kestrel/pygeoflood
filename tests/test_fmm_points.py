"""Regression tests for the vectorized get_fmm_points (review art_SjgX1Jzb F3).

The replaced numba prange kernel appended to Python lists from parallel
workers: outputs were nondeterministic and the process crashed with SIGSEGV
or heap corruption at ambient thread counts. Reproduced at HEAD 0b2fcd8
with numba 0.68.1: exit -11 at 8 threads on both a 2000x2000/40-label and a
500x500/25-label domain ("free(): chunks in smallbin corrupted"), and 2
distinct outputs across 20 identical calls in a surviving run. The
vectorized selection is a pure function of the inputs; these tests pin the
exact selection semantics and thread-count invariance.
"""

import os
import subprocess
import sys

import numpy as np

from pygeoflood.tools import get_fmm_points


def seeded_domain(size=64):
    """Deterministic raster: one 50% basin, one 12.5% basin, small ones."""
    rng = np.random.default_rng(11)
    basins = rng.integers(3, 6, (size, size)).astype(np.int32)
    basins[: size // 2] = 1  # 50% of the domain
    basins[size // 2 :, : size // 4] = 2  # 12.5%
    return basins


def outlets_for(basins, n=8):
    """(2, N) candidate outlet coordinates; label L selects basin id L+1."""
    return np.transpose(np.argwhere(basins > 0)[:n])


def serial_reference(basins, outlets, basin_elements, area_threshold):
    """The review's selection math, serial and label-ordered."""
    n_pixels = basins.size
    fmm_y, fmm_x = [], []
    for label in range(outlets.shape[1]):
        numelements = int(np.sum(basins.ravel() == label + 1))
        percent = numelements * 100.00001 / n_pixels
        if percent > area_threshold and numelements > basin_elements:
            fmm_y.append(outlets[0, label])
            fmm_x.append(outlets[1, label])
    return np.array([fmm_y, fmm_x])


def test_matches_serial_reference():
    basins = seeded_domain()
    outlets = outlets_for(basins)
    for basin_elements, area_threshold in [
        (2, 0.1),
        (100, 0.1),
        (2, 13.0),
        (10, 50.0),
    ]:
        np.testing.assert_array_equal(
            get_fmm_points(basins, outlets, basin_elements, area_threshold),
            serial_reference(basins, outlets, basin_elements, area_threshold),
        )


def test_matches_serial_reference_on_shuffled_outlets():
    basins = seeded_domain()
    shuffled = outlets_for(basins)[:, ::-1]
    np.testing.assert_array_equal(
        get_fmm_points(basins, shuffled, 2, 0.1),
        serial_reference(basins, shuffled, 2, 0.1),
    )


def test_element_threshold_is_strict():
    """A basin with exactly basin_elements cells is excluded (strict >)."""
    basins = np.zeros((10, 10), dtype=np.int32)
    basins[0, 0:3] = 1  # exactly 3 cells
    basins[1, 0:4] = 2  # 4 cells
    outlets = np.array([[0, 1], [0, 1]])
    result = get_fmm_points(basins, outlets, 3, 0.0)
    assert result.shape == (2, 1)
    assert list(result[0]) == [1]  # fmmY: only the 4-cell basin's row
    assert list(result[1]) == [1]  # fmmX


def test_nan_and_background_basins_are_never_selected():
    basins = seeded_domain().astype(np.float64)
    basins[0, 0] = np.nan  # float-raster nodata convention (read_raster)
    basins[0, 1] = 0.0  # background
    outlets = outlets_for(basins)
    result = get_fmm_points(basins, outlets, 2, 0.1)
    np.testing.assert_array_equal(
        result, serial_reference(basins, outlets, 2, 0.1)
    )
    assert result.shape[1] >= 1  # sanity: the large basins still qualify


def test_empty_selection_is_two_by_zero():
    basins = np.zeros((10, 10), dtype=np.int32)
    basins[0, 0:3] = 1  # too few elements for basin_elements=5
    outlets = np.array([[0, 1, 2], [0, 0, 0]])
    result = get_fmm_points(basins, outlets, 5, 0.1)
    assert result.shape == (2, 0)


INVAR_SCRIPT = """
import sys
import numpy as np
from pygeoflood.tools import get_fmm_points

rng = np.random.default_rng(11)
size = 64
basins = rng.integers(3, 6, (size, size)).astype(np.int32)
basins[: size // 2] = 1
basins[size // 2:, : size // 4] = 2
outlets = np.transpose(np.argwhere(basins > 0)[:8])
sigs = [get_fmm_points(basins, outlets, 2, 0.1).tobytes() for _ in range(5)]
assert len(set(sigs)) == 1, "output varies across calls at a fixed thread count"
open(sys.argv[1], "wb").write(sigs[0])
"""


def test_selection_is_invariant_across_thread_counts(tmp_path):
    """The same domain must select the same points at 1 and 8 numba threads.

    On the replaced prange kernel this crashed the interpreter (SIGSEGV,
    exit -11) at 8 threads — reproduced at HEAD 0b2fcd8 on both a 2000x2000
    and a 500x500 domain — and produced 2 distinct outputs across 20
    identical calls when it survived.
    """
    digests = {}
    for threads in ("1", "8"):
        out = tmp_path / f"points-{threads}.bin"
        subprocess.run(
            [sys.executable, "-c", INVAR_SCRIPT, str(out)],
            check=True,
            env=dict(
                os.environ, NUMBA_NUM_THREADS=threads, OMP_NUM_THREADS=threads
            ),
            timeout=300,
        )
        digests[threads] = out.read_bytes()
    assert digests["1"] == digests["8"]
