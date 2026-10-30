"""fast_marching degenerate-basin regression test (PG-1 finding F4).

skfmm raises ValueError for a degenerate masked speed field — probed on
scikit-fmm 2025.6.23, an all-zero speed field leaves phi without a zero
contour and travel_time raises ValueError. The old ``except ValueError``
branch printed and fell through to the assignment, so ``travel_time`` was
unbound (first basin -> UnboundLocalError) or still held the previous
basin's solution (later basins -> silent wrong-basin distances written into
geodesic_distance).

Pinned behavior: re-raise with basin context (fail fast, name the basin).
"""

import numpy as np
import pytest

from pygeoflood.tools import fast_marching


def test_fast_marching_raises_with_basin_context_for_degenerate_basin():
    # float32 to match the real pipeline, where basins comes from read_raster
    basins = np.ones((8, 8), dtype=np.float32)
    fmm_start_points = np.array([[4], [4]], dtype=np.int64)
    mfd_fac = np.full((8, 8), 100.0, dtype=np.float32)
    # Degenerate: the masked speed field is all zero.
    cost_function = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(RuntimeError, match="basin"):
        fast_marching(fmm_start_points, basins, mfd_fac, cost_function)
