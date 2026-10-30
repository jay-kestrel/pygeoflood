# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
# Mark Wang, and Matthew Preisser.
"""
PygeofloodKernels — Julia ports of pygeoflood's first-party numeric kernels.

This is a bug-for-bug port of three Numba kernels from
`src/pygeoflood/tools.py`:

- `process_cells` — synthetic-rating-curve accumulation (with the
  deterministic 32-chunk row partition and ordered serial fold that the
  Numba/Rust kernels share, so float sums reduce in the same order),
- `binary_search` + `jit_inun` — inundation depth from a stage lookup,
- `jit_binary_hand` — two-sweep chamfer-style binary HAND transform.

The port replicates current behavior including known defects (the
`searchsorted` insertion-index fold for hydroids missing from the lookup
table, the `|hand| <= 1e-8` near-zero inclusion, and the NaN-DEM binary-HAND
quirk before the driver's nodata post-pass). WhiteboxTools hydrology stages,
raster I/O, and the fast-marching geodesic path are out of scope.

Numerics: every arithmetic step is documented against the Python/Rust
reference in the kernel docstrings — the port promotes to Float64 per
operation and truncates per store exactly where NumPy's weak-scalar
promotion does, so the golden rtol 1e-6 gates hold reduction-order-identically.
"""
module PygeofloodKernels

export process_cells, jit_inun, jit_binary_hand, binary_search,
       PROCESS_CELLS_MAX_CHUNKS

"""Upper bound on the row-chunk partition of `process_cells`
(`tools.py:1544`; also `PROCESS_CELLS_MAX_CHUNKS` in `rust/src/cells.rs`)."""
const PROCESS_CELLS_MAX_CHUNKS = 32

# ---------------------------------------------------------------------------
# process_cells
# ---------------------------------------------------------------------------

"""
    process_cells(hand, segment_catchments, slope, hid_dict_array, heights,
                  cell_area)
      -> (counts::Matrix{Int32}, surface::Matrix{Float32},
          bed::Matrix{Float32}, volume::Matrix{Float32})

Synthetic-rating-curve accumulation, bug-for-bug port of the `process_cells`
wrapper (`tools.py:1545–1616`) and its Rust kernel (`rust/src/cells.rs`).

For every cell and every stage height the kernel accumulates, into
per-catchment columns: the cell count, the surface area (`cell_area`), the
bed area (`cell_area * sqrt(1 + slope^2)`), and the volume
(`(height - hand) * cell_area`).

Reference semantics replicated here:

- A cell is active for a stage when `hand < height` **or** `|hand| <= 1e-8`
  (the near-zero inclusion uses NumPy's f64 comparison, so a Float32 HAND is
  promoted before the test).
- Stage heights index rows and catchments index columns; a cell whose
  catchment label is absent from `hid_dict_array[:, 1]` folds into the
  `searchsorted` insertion slot (bug-for-bug; no bounds check).
- Output precision follows the wrapper's dtype-precision flags derived from
  the argument eltypes (slope f32, delta f32, volume f32): accumulators are
  Float32 regardless, but intermediate products are rounded through Float32
  per update where the corresponding flag is set.
- The row partition (`<= 32` contiguous chunks, a pure function of the row
  count) and the ordered serial fold reproduce the Numba/Rust reduction
  order, which is what the golden rtol 1e-6 SRC gate depends on.
"""
function process_cells(hand::AbstractMatrix{<:Real},
                       segment_catchments::AbstractMatrix{<:Real},
                       slope::AbstractMatrix{<:Real},
                       hid_dict_array::AbstractMatrix{<:Real},
                       heights::AbstractVector{<:Real},
                       cell_area::Real)
    size(hid_dict_array, 2) == 2 ||
        throw(ArgumentError("hid_dict_array must have shape (ncatchments, 2)"))
    size(hand) == size(segment_catchments) == size(slope) || throw(DimensionMismatch(
        "hand, segment_catchments, and slope must have identical shapes"))

    hydroids = Float64.(view(hid_dict_array, :, 1))
    (all(isfinite, hydroids) && issorted(hydroids)) || throw(ArgumentError(
        "hydroid lookup must be finite and sorted ascending"))

    # Precision flags mirror the Python wrapper's dtype introspection.
    slope_f32 = eltype(slope) === Float32
    delta_f32 = eltype(hand) === Float32 && eltype(heights) === Float32
    volume_f32 = delta_f32 && cell_area isa Float32

    return _process_cells(
        Float64.(hand), Float64.(segment_catchments), Float64.(slope),
        hydroids, Float64.(heights), Float64(cell_area),
        slope_f32, delta_f32, volume_f32)
end

function _process_cells(hand::Matrix{Float64},
                        catchments::Matrix{Float64},
                        slope::Matrix{Float64},
                        hydroids::Vector{Float64},
                        heights::Vector{Float64},
                        cell_area::Float64,
                        slope_f32::Bool, delta_f32::Bool, volume_f32::Bool)
    nheight = length(heights)
    ncatch = length(hydroids)
    nrow, ncol = size(hand)

    counts = zeros(Int32, nheight, ncatch)
    surface = zeros(Float32, nheight, ncatch)
    bed = zeros(Float32, nheight, ncatch)
    volume = zeros(Float32, nheight, ncatch)
    nheight == 0 && return counts, surface, bed, volume
    nrow == 0 && return counts, surface, bed, volume

    nchunks = clamp(nrow, 1, PROCESS_CELLS_MAX_CHUNKS)
    counts_bufs = [zeros(Int32, nheight, ncatch) for _ in 1:nchunks]
    surface_bufs = [zeros(Float32, nheight, ncatch) for _ in 1:nchunks]
    bed_bufs = [zeros(Float32, nheight, ncatch) for _ in 1:nchunks]
    volume_bufs = [zeros(Float32, nheight, ncatch) for _ in 1:nchunks]

    try
        Threads.@threads for chunk in 1:nchunks
            start0, stop0 = _row_range(nrow, nchunks, chunk - 1)  # 0-based, half-open
            _accumulate_chunk!(
                counts_bufs[chunk], surface_bufs[chunk], bed_bufs[chunk],
                volume_bufs[chunk], hand, catchments, slope, hydroids, heights,
                cell_area, slope_f32, delta_f32, volume_f32,
                start0, stop0, ncol, ncatch, nheight)
        end
    catch err
        # surface the kernel's own error, not the threading wrapper's
        # (@threads failures arrive as TaskFailedException or CompositeException
        # depending on thread count/schedule)
        inner = err
        while true
            if inner isa TaskFailedException
                inner = inner.task.exception
            elseif inner isa CompositeException && length(inner.exceptions) == 1
                inner = inner.exceptions[1]
            else
                break
            end
        end
        inner === err ? rethrow() : throw(inner)
    end

    # Ordered serial fold — the reduction order the golden gates rely on.
    @inbounds for chunk in 1:nchunks
        counts .+= counts_bufs[chunk]
        surface .+= surface_bufs[chunk]
        bed .+= bed_bufs[chunk]
        volume .+= volume_bufs[chunk]
    end
    return counts, surface, bed, volume
end

"""Deterministic row partition — verbatim port of `row_range`
(`rust/src/cells.rs`): chunk `c` (0-based) covers
`start0:(stop0 - 1)` rows 0-based, `stop0` exclusive."""
function _row_range(rows::Int, chunks::Int, chunk::Int)
    base, rem = divrem(rows, chunks)
    start0 = chunk * base + min(chunk, rem)
    stop0 = start0 + base + (chunk < rem ? 1 : 0)
    return start0, stop0
end

"""Per-chunk accumulation — verbatim port of the Rust `accumulate` body.
Pixels are visited in row-major order (outer rows, inner columns), matching
the reference flat-index iteration; Float64 arithmetic is truncated through
Float32 at exactly the reference's store points."""
function _accumulate_chunk!(counts::Matrix{Int32}, surface::Matrix{Float32},
                            bed::Matrix{Float32}, volume::Matrix{Float32},
                            hand::Matrix{Float64}, catchments::Matrix{Float64},
                            slope::Matrix{Float64}, hydroids::Vector{Float64},
                            heights::Vector{Float64}, cell_area::Float64,
                            slope_f32::Bool, delta_f32::Bool, volume_f32::Bool,
                            start0::Int, stop0::Int,
                            ncol::Int, ncatch::Int, nheight::Int)
    @inbounds begin
        for row0 in start0:stop0-1
            for col0 in 0:ncol-1
                hand_v = hand[row0 + 1, col0 + 1]
                near_zero = abs(hand_v) <= 1e-8
                hydroid_v = catchments[row0 + 1, col0 + 1]
                catchment = _partition_point(hydroids, hydroid_v)  # 0-based
                slope_v = slope[row0 + 1, col0 + 1]
                square = slope_f32 ?
                    Float64(Float32(slope_v) * Float32(slope_v)) :
                    slope_v * slope_v
                bed_inc = cell_area * sqrt(1.0 + square)
                for stage0 in 0:nheight-1
                    height = heights[stage0 + 1]
                    if hand_v < height || near_zero
                        catchment == ncatch && throw(ErrorException(
                            "active cell hydroid is outside the lookup table"))
                        # index0 = stage0 * ncatch + catchment is the reference's
                        # ROW-MAJOR flat index; these Julia matrices are
                        # (stage, catchment), so index the dims directly
                        si = stage0 + 1
                        ci = catchment + 1
                        delta = delta_f32 ?
                            Float64(Float32(height) - Float32(hand_v)) :
                            height - hand_v
                        volume_inc = volume_f32 ?
                            Float64(Float32(delta) * Float32(cell_area)) :
                            delta * cell_area
                        counts[si, ci] += Int32(1)
                        surface[si, ci] =
                            Float32(Float64(surface[si, ci]) + cell_area)
                        bed[si, ci] =
                            Float32(Float64(bed[si, ci]) + bed_inc)
                        volume[si, ci] =
                            Float32(Float64(volume[si, ci]) + volume_inc)
                    end
                end
            end
        end
    end
    return nothing
end

"""`np.searchsorted(hydroids, value)` — leftmost insertion point (0-based).
Mirrors the Rust kernel: no bounds check, so a value greater than every
hydroid returns `length(hydroids)` (the fold-into-last-slot defect), and NaN
catchment values sort after all finite IDs (so an active NaN cell errors)."""
function _partition_point(hydroids::Vector{Float64}, value::Float64)
    lo, hi = 0, length(hydroids)
    while lo < hi
        mid = (lo + hi) >>> 1
        if hydroids[mid + 1] < value || isnan(value)
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

# ---------------------------------------------------------------------------
# binary_search + jit_inun
# ---------------------------------------------------------------------------

"""Binary search — verbatim port of `binary_search` (`tools.py:1786–1806`).
Returns the 0-based index of `x` in `arr`, or `-1` when absent."""
function binary_search(arr::AbstractVector{Float64}, x::Float64)
    low, high = 0, length(arr) - 1
    while low <= high
        mid = (low + high) ÷ 2
        mid_val = arr[mid + 1]
        if mid_val < x
            low = mid + 1
        elseif mid_val > x
            high = mid - 1
        else
            return mid
        end
    end
    return -1
end

"""
    jit_inun(hand, seg_catch, hydroids, stage_m) -> Matrix{Float32}

Inundation depth — bug-for-bug port of `jit_inun` (`tools.py:1808–1824`).
Per cell: binary-search its catchment label in the sorted `hydroids` table
and, when the stage exceeds the HAND value, write `stage - hand` (Float64
difference, Float32 store); otherwise leave the cell at Float32 NaN. An
unmapped label searches to `-1` and receives the sentinel stage `-32768.0`,
which can never exceed a real HAND value — the cell stays dry. `hydroids`
must be sorted ascending; nothing validates that (bug-for-bug).
"""
function jit_inun(hand::AbstractMatrix{<:AbstractFloat},
                  seg_catch::AbstractMatrix{<:Real},
                  hydroids::AbstractVector{Float64},
                  stage_m::AbstractVector{Float64})
    size(hand) == size(seg_catch) || throw(DimensionMismatch(
        "hand and seg_catch must have identical shapes"))
    nrows, ncols = size(hand)
    inun = fill(Float32(NaN), nrows, ncols)  # np.empty_like(hand, float32) + NaN
    hydroids_c = collect(Float64, hydroids)
    stage_c = collect(Float64, stage_m)
    Threads.@threads for row in 1:nrows  # cell writes are independent
        @inbounds for col in 1:ncols
            hydroid = Float64(seg_catch[row, col])
            hand_h = Float64(hand[row, col])
            hydroid_idx = binary_search(hydroids_c, hydroid)
            h = hydroid_idx != -1 ? stage_c[hydroid_idx + 1] : -32768.0
            if h > hand_h
                inun[row, col] = Float32(h - hand_h)
            end
        end
    end
    return inun
end

# ---------------------------------------------------------------------------
# jit_binary_hand
# ---------------------------------------------------------------------------

"""
    jit_binary_hand(dem, flowline_raster, nodata) -> Matrix{Int}

Binary HAND — bug-for-bug port of `jit_binary_hand` (`tools.py:1092–1200`),
a two-sweep chamfer-style distance transform. Sweep 1 walks rows ascending
and columns ascending with the 4 neighborhood offsets 0–3 (weights 1, 1.414,
1, 1.414); sweep 2 walks both descending with offsets 4–7 (same weights).
Flowline cells seed `distance = 0` and `allocation = dem`; every other cell
adopts the distance-weighted best neighbor's distance and inherits its
`allocation` (the source cell's elevation). The kernel ignores the
`parallel=True` race in the Python reference and implements the sequential
two-pass semantics — the behavior the golden suite actually pins
(`NUMBA_NUM_THREADS=1`).

Returns 0/1 per cell: `allocation < dem ? 0 : 1` in Float64, so a NaN DEM
cell compares false and yields 1 (bug-for-bug quirk; the driver's nodata
post-pass overwrites those cells downstream of the kernel).
"""
function jit_binary_hand(dem::AbstractMatrix{<:AbstractFloat},
                         flowline_raster::AbstractMatrix{<:Integer},
                         nodata::Real)
    size(dem) == size(flowline_raster) || throw(DimensionMismatch(
        "dem and flowline_raster must have identical shapes"))
    nrows, ncols = size(dem)
    nodata_f = Float64(nodata)

    distance = Array{Float64}(undef, nrows, ncols)
    allocation = Array{Float64}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        if flowline_raster[i, j] == 1
            distance[i, j] = 0.0
            allocation[i, j] = Float64(dem[i, j])
        else
            distance[i, j] = Inf
            allocation[i, j] = Inf
        end
    end

    # 0-based neighbor offsets (dx, dy) 0..7, weight per offset — from tools.py.
    d_x = (-1, -1, 0, 1, 1, 1, 0, -1)
    d_y = (0, -1, -1, -1, 0, 1, 1, 1)
    h_w = (1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414)

    distance_c = distance
    allocation_c = allocation
    @inbounds begin
        # Sweep 1: forward rows and columns, neighbor offsets 0..3.
        for row in 0:nrows-1
            for col in 0:ncols-1
                z = distance_c[row + 1, col + 1]
                if z != 0
                    z_min = Inf
                    which_cell = 0
                    for i in 0:3
                        x = col + d_x[i + 1]
                        y = row + d_y[i + 1]
                        if x >= 0 && x < ncols && y >= 0 && y < nrows
                            z2 = distance_c[y + 1, x + 1]
                            if z2 != nodata_f
                                z2 += h_w[i + 1]
                                if z2 < z_min
                                    z_min = z2
                                    which_cell = i
                                end
                            end
                        end
                    end
                    if z_min < z
                        distance_c[row + 1, col + 1] = z_min
                        x = col + d_x[which_cell + 1]
                        y = row + d_y[which_cell + 1]
                        allocation_c[row + 1, col + 1] =
                            allocation_c[y + 1, x + 1]
                    end
                end
            end
        end
        # Sweep 2: reverse rows and columns, neighbor offsets 4..7.
        for row in nrows-1:-1:0
            for col in ncols-1:-1:0
                z = distance_c[row + 1, col + 1]
                if z != 0
                    z_min = Inf
                    which_cell = 0
                    for i in 4:7
                        x = col + d_x[i + 1]
                        y = row + d_y[i + 1]
                        if x >= 0 && x < ncols && y >= 0 && y < nrows
                            z2 = distance_c[y + 1, x + 1]
                            if z2 != nodata_f
                                z2 += h_w[i + 1]
                                if z2 < z_min
                                    z_min = z2
                                    which_cell = i
                                end
                            end
                        end
                    end
                    if z_min < z
                        distance_c[row + 1, col + 1] = z_min
                        x = col + d_x[which_cell + 1]
                        y = row + d_y[which_cell + 1]
                        allocation_c[row + 1, col + 1] =
                            allocation_c[y + 1, x + 1]
                    end
                end
            end
        end
    end

    # nodata DEM cells carry their nodata allocation (np.where pass)
    @inbounds for i in 1:nrows, j in 1:ncols
        if Float64(dem[i, j]) == nodata_f
            allocation[i, j] = nodata_f
        end
    end

    # np.where(allocation < dem, 0, 1) — Float64 comparison
    binary = Array{Int, 2}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        binary[i, j] = allocation[i, j] < Float64(dem[i, j]) ? 0 : 1
    end
    return binary
end

end # module
