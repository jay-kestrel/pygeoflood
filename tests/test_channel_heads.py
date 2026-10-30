"""Channel-head kernel regression tests (PG-1 findings F5, F6).

F5: jit_channel_heads preallocates a (max_channel_heads, 2) buffer but the
old capacity check ran *after* the write and never stopped the scan — once
the cap was reached, every further head was written out of bounds (numba
performs no bounds checking) and a warning was printed per excess head.

F6: get_channel_heads indexed bin_edges[2], which needs num_bins >= 2 —
fewer than 4 connected components (num_bins = floor(sqrt(n))) cannot fill
that index and raised IndexError.
"""

import sys

import numpy as np
import pytest

from pygeoflood.tools import get_channel_heads

# One dominant 20x20 block (400 cells) plus five 2x4 blobs (8 cells each),
# placed with 8-connectivity clearance so each is its own component:
# num_labels = 6, so the histogram threshold path is exercised.
BLOCK_ORIGIN = (10, 10)
BLOB_POSITIONS = [(2, 2), (2, 20), (2, 32), (30, 2), (30, 32)]


def make_skeleton(with_block=True, blobs=BLOB_POSITIONS, shape=(40, 40)):
    skeleton = np.zeros(shape, dtype=np.int32)
    if with_block:
        r0, c0 = BLOCK_ORIGIN
        skeleton[r0 : r0 + 20, c0 : c0 + 20] = 1
    for r, c in blobs:
        skeleton[r : r + 2, c : c + 4] = 1
    return skeleton


def test_tiny_cap_stops_before_writing_and_warns_once(capfd):
    """max_channel_heads=350 on a 400-cell block: the kernel must stop at
    the cap (no out-of-bounds write) and warn exactly once.

    The unpatched kernel wrote past the buffer for every excess head and
    printed a warning per excess head (50 warnings on this input)."""
    skeleton = make_skeleton()
    geodesic = np.ones(skeleton.shape, dtype=np.float64)

    ch_rows, ch_cols = get_channel_heads(
        skeleton,
        geodesic,
        channel_head_median_dist=1,
        max_channel_heads=350,
    )

    assert ch_rows is not None and ch_cols is not None
    assert ch_rows.shape == ch_cols.shape
    assert ch_rows.size == 350
    # Deterministic heads: row-major scan of the block.
    assert list(zip(ch_rows[:5].tolist(), ch_cols[:5].tolist())) == [
        (10, 10),
        (10, 11),
        (10, 12),
        (10, 13),
        (10, 14),
    ]

    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    assert captured.out.count("Warning: max_channel_heads") == 1


@pytest.mark.parametrize("n_blobs, expected_heads", [(1, 8), (2, 16), (3, 24)])
def test_get_channel_heads_survives_up_to_three_components(
    n_blobs, expected_heads
):
    """A skeleton with fewer than 4 connected components cannot provide
    bin_edges[2] (num_bins = floor(sqrt(n)) < 2) — the old code raised
    IndexError for 1-3 components. A defined fallback threshold (median
    component size) must keep the endpoint scan working."""
    skeleton = make_skeleton(with_block=False, blobs=BLOB_POSITIONS[:n_blobs])
    geodesic = np.ones(skeleton.shape, dtype=np.float64)

    ch_rows, ch_cols = get_channel_heads(
        skeleton,
        geodesic,
        channel_head_median_dist=1,
        max_channel_heads=100,
    )

    assert ch_rows is not None and ch_cols is not None
    assert ch_rows.shape == ch_cols.shape
    # Constant geodesic distance makes every qualifying skeleton cell a
    # channel head: 8 cells per blob, median fallback lets all qualify.
    assert ch_rows.size == expected_heads
