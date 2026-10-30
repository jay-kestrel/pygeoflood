# SPDX-License-Identifier: GPL-3.0-or-later
# Frozen verbatim from tools.py at 1da71e8 (83ab7d3 chunk reduction).
# Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
# Mark Wang, and Matthew Preisser. Test oracle only.
import numpy as np
from numba import jit, prange


PROCESS_CELLS_MAX_CHUNKS = 32  # fixed chunk budget: keeps private buffers bounded


@jit(nopython=True, parallel=True)
def process_cells(
    hand,
    segment_catchments,
    slope,
    hid_dict_array,
    heights,
    cell_area,
):
    # Race-free rewrite: each parallel worker accumulates into a private
    # per-chunk buffer, and the buffers are folded serially in fixed order.
    # The original accumulated directly into shared per-catchment arrays from
    # prange workers — a non-atomic read-modify-write race whenever one
    # catchment spans multiple rows, so totals lost updates and varied
    # from run to run.
    nheight = len(heights)
    ncatch = len(hid_dict_array[:, 0])
    nrow = hand.shape[0]

    # The chunk partition is a pure function of the input shape (never of
    # thread count or scheduling), so per-chunk partial sums and the final
    # fold order are deterministic: bitwise-identical output for any
    # NUMBA_NUM_THREADS.
    nchunks = max(1, min(nrow, PROCESS_CELLS_MAX_CHUNKS))
    chunk_counts = np.zeros((nchunks, nheight, ncatch), dtype=np.int32)
    chunk_surf = np.zeros((nchunks, nheight, ncatch), dtype=np.float32)
    chunk_bed = np.zeros((nchunks, nheight, ncatch), dtype=np.float32)
    chunk_vol = np.zeros((nchunks, nheight, ncatch), dtype=np.float32)

    # Contiguous, size-balanced row ranges: chunk c owns
    # [c * base + min(c, rem), stop).
    base = nrow // nchunks
    rem = nrow % nchunks

    # loop through each chunk of rows; workers touch disjoint buffer slices
    for c in prange(nchunks):
        start = c * base + min(c, rem)
        stop = start + base + (1 if c < rem else 0)

        # chunk-private accumulators (disjoint slices across chunks)
        counts_c = chunk_counts[c]
        surf_c = chunk_surf[c]
        bed_c = chunk_bed[c]
        vol_c = chunk_vol[c]

        for i in range(start, stop):
            for j in range(hand.shape[1]):
                hydroid = segment_catchments[i, j]
                hydroid_index = np.searchsorted(hid_dict_array[:, 0], hydroid)
                hand_height = hand[i, j]
                slp_value = slope[i, j]
                for h_idx in range(nheight):
                    if hand_height < heights[h_idx] or np.isclose(hand_height, 0.0):
                        counts_c[h_idx, hydroid_index] += 1
                        surf_c[h_idx, hydroid_index] += cell_area
                        incBedArea = cell_area * np.sqrt(1 + slp_value**2)
                        bed_c[h_idx, hydroid_index] += incBedArea
                        incVolume = (heights[h_idx] - hand_height) * cell_area
                        vol_c[h_idx, hydroid_index] += incVolume

    # Deterministic reduction: fold chunks sequentially in fixed index order.
    CellCount = np.zeros((nheight, ncatch), dtype=np.int32)
    SurfaceArea = np.zeros((nheight, ncatch), dtype=np.float32)
    BedArea = np.zeros((nheight, ncatch), dtype=np.float32)
    Volume = np.zeros((nheight, ncatch), dtype=np.float32)
    for c in range(nchunks):
        CellCount += chunk_counts[c]
        SurfaceArea += chunk_surf[c]
        BedArea += chunk_bed[c]
        Volume += chunk_vol[c]

    return CellCount, SurfaceArea, BedArea, Volume

