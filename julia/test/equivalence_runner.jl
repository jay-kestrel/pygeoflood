# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
# Mark Wang, and Matthew Preisser.
#
# Golden-corpus equivalence runner: validates the Julia kernel ports against
# the same fixtures the Python golden suite gates on (rtol 1e-6 CSV columns,
# exact binary outputs), mirroring tests/test_golden_equivalence.py's gate
# functions plus a discrimination check proving the gates can fail.
#
# Usage:
#   julia --project=julia julia/test/equivalence_runner.jl <fixtures-dir> [--report=<path.json>]
#
# Fixtures are produced by julia/tools/dump_kernel_fixtures.py from a golden
# corpus run (tests/golden/generate_golden_fixtures.py); the runner locates
# the corpus's SRC CSV relative to the fixtures directory. Exit code 0 = all
# gates pass, 1 = any gate or discrimination check fails.

using Printf
using PygeofloodKernels

include(joinpath(@__DIR__, "Npy.jl"))
using .Npy

const SRC_COLUMNS = ["HYDROID", "Stage_m", "NumCells", "SurfaceArea_m2",
    "BedArea_m2", "Volume_m3", "Area_km2", "Slope", "Length_km", "COMID",
    "StreamOrder", "Roughness", "TopWidth_m", "WettedPerimeter_m",
    "WetArea_m2", "HydraulicRadius_m", "Discharge_cms", "FloodAreaRatio"]
const RTOL = 1e-6   # tolerance table from tests/test_golden_equivalence.py
const FIM_NODATA = -32768.0

"""np.allclose(a, b; rtol, atol=0, equal_nan=True): |a-b| <= rtol*|b|."""
allclose(a::AbstractFloat, b::AbstractFloat) =
    (isnan(a) && isnan(b)) || abs(a - b) <= RTOL * abs(b)

function allclose(candidate::AbstractMatrix{<:AbstractFloat},
                  golden::AbstractMatrix{<:AbstractFloat})
    size(candidate) == size(golden) || return false
    return all(map(allclose, candidate, golden))
end

"""Parse a plain CSV into (header, rows-of-strings)."""
function read_csv(path::AbstractString)
    lines = split(read(path, String), "\n"; keepempty=false)
    isempty(lines) && error("$path: empty CSV")
    header = split(strip(lines[1]), ",")
    rows = [split(strip(line), ",") for line in lines[2:end]]
    return header, rows
end

function parse_failsafe(s::AbstractString, what::AbstractString)
    v = tryparse(Float64, s)
    v === nothing && error("cannot parse $what: '$s'")
    return v
end

"""Gate: kernel accumulators vs the golden SRC CSV — identifiers exact,
NumCells exact, area/bed/volume columns at rtol 1e-6. Returns
(passed::Bool, detail::String)."""
function gate_src_csv(csv_path, hydroids, heights, counts, surface, bed, volume)
    nheight, ncatch = size(counts)
    header, rows = read_csv(csv_path)
    header == SRC_COLUMNS ||
        return false, "column mismatch: got $(join(header, ","))"
    length(rows) == nheight * ncatch ||
        return false, "row count $(length(rows)) != expected $nheight*$ncatch"
    col = Dict(name => i for (i, name) in enumerate(header))

    for (k, row) in enumerate(rows)
        k0 = k - 1
        catchment0, stage0 = divrem(k0, nheight)   # hydroid-major row order
        hydroid_csv = parse_failsafe(row[col["HYDROID"]], "HYDROID")
        hydroid_csv == hydroids[catchment0 + 1] || return false,
            "row $k: HYDROID $hydroid_csv != table $(hydroids[catchment0 + 1])"
        stage_csv = parse_failsafe(row[col["Stage_m"]], "Stage_m")
        height = heights[stage0 + 1]
        abs(stage_csv - height) <= 5e-5 || return false,
            "row $k: Stage_m $stage_csv vs height $height"
        numcells = tryparse(Int, row[col["NumCells"]])
        numcells === nothing && return false, "row $k: bad NumCells"
        # kernel arrays are (stage, catchment)
        numcells != counts[stage0 + 1, catchment0 + 1] && return false,
            "row $k: NumCells $numcells != " *
            "$(counts[stage0 + 1, catchment0 + 1]) (exact gate)"
        surface_csv = parse_failsafe(row[col["SurfaceArea_m2"]], "SurfaceArea_m2")
        bed_csv = parse_failsafe(row[col["BedArea_m2"]], "BedArea_m2")
        volume_csv = parse_failsafe(row[col["Volume_m3"]], "Volume_m3")
        passes = allclose(surface[stage0 + 1, catchment0 + 1], surface_csv) &&
                 allclose(bed[stage0 + 1, catchment0 + 1], bed_csv) &&
                 allclose(volume[stage0 + 1, catchment0 + 1], volume_csv)
        passes || return false,
            "row $k: area/bed/volume outside rtol $RTOL (candidate " *
            "$(surface[stage0 + 1, catchment0 + 1])/" *
            "$(bed[stage0 + 1, catchment0 + 1])/" *
            "$(volume[stage0 + 1, catchment0 + 1]) vs " *
            "$surface_csv/$bed_csv/$volume_csv)"
    end
    return true, "$(length(rows)) rows: NumCells exact, floats within rtol $RTOL"
end

"""The `inundate` driver's NaN -> -32768 fill before raster write."""
function fill_nodata!(a::Matrix{Float32})
    @inbounds for i in eachindex(a)
        isnan(a[i]) && (a[i] = Float32(FIM_NODATA))
    end
end

"""Gate: inundation mask exact + depths rtol 1e-6 (mirrors
gate_inundation_map). NOTE: fills its arguments' NaNs in place (mirrors the
driver's write-time fill). Returns (passed::Bool, detail::String)."""
function gate_inundation_map!(candidate::Matrix{Float32}, golden::Matrix{Float32})
    size(candidate) == size(golden) || return false, "shape mismatch"
    fill_nodata!(candidate)
    fill_nodata!(golden)
    mask_c = candidate .== FIM_NODATA
    mask_g = golden .== FIM_NODATA
    mask_c == mask_g ||
        return false, "mask mismatch at $(count(mask_c .!= mask_g)) cells"
    allclose(candidate, golden) || return false, "depths outside rtol $RTOL"
    return true, "mask exact, depths within rtol $RTOL"
end

"""Gate: binary HAND exact after the driver's nodata post-pass
(binary_hand[dem == nodata] = -32768; binary_hand[isnan(dem)] = -32768).
Returns (passed::Bool, detail::String)."""
function gate_binary_hand(candidate::AbstractMatrix{<:Integer},
                          golden::AbstractMatrix{<:Integer},
                          dem::Matrix{Float32}, nodata::Float64)
    size(candidate) == size(golden) == size(dem) || return false, "shape mismatch"
    final = Int.(candidate)
    @inbounds for i in eachindex(dem)
        d = dem[i]
        (Float64(d) == nodata || isnan(d)) && (final[i] = -32768)
    end
    n_bad = count(final .!= Int.(golden))
    n_bad == 0 || return false, "$n_bad cells differ (exact gate)"
    return true, "exact match after nodata post-pass"
end

function main()
    fixtures = length(ARGS) >= 1 ? ARGS[1] :
        error("usage: julia equivalence_runner.jl <fixtures-dir> [--report=path]")
    report_path = nothing
    for arg in ARGS[2:end]
        m = match(r"--report=(.+)", arg)
        m === nothing || (report_path = m.captures[1])
    end

    meta = read(joinpath(fixtures, "fixtures.json"), String)
    ld(name) = Npy.read(joinpath(fixtures, "$name.npy"))
    hand = ld("hand")
    catchments = ld("segment_catchments")
    slope = ld("slope")
    hydroids = ld("hydroids")
    heights = ld("heights")
    inun_hydroids = ld("inun_hydroids")
    inun_stages = ld("inun_stages")
    dem = ld("dem")
    flowline_raster = ld("flowline_raster")
    golden_fim = ld("golden_fim")
    golden_binary_hand = ld("golden_binary_hand")

    cell_area = parse(Float64,
        match(r""""cell_area"\s*:\s*([0-9.eE+-]+)""", meta).captures[1])
    nodata = parse(Float64,
        match(r""""nodata"\s*:\s*([0-9.eE+-]+)""", meta).captures[1])
    flags = (
        slope_f32 = occursin("\"slope_f32\": true", meta),
        delta_f32 = occursin("\"delta_f32\": true", meta),
        volume_f32 = occursin("\"volume_f32\": true", meta),
    )
    corpus_csv = normpath(joinpath(fixtures, "..", "..", "..", "tests",
        "golden", "data", "run1", "OC1mTestCrop_src.csv"))

    println("== PygeofloodKernels golden equivalence ==")
    println("Julia $(VERSION), threads=$(Threads.nthreads()), " *
            "cell_area=$cell_area, nodata=$nodata")
    println("precision flags (from fixtures.json): $flags")
    derived = (slope_f32 = eltype(slope) === Float32,
               delta_f32 = eltype(hand) === Float32 && eltype(heights) === Float32,
               volume_f32 = false)  # JSON-parsed cell_area is always Float64
    flags == derived ||
        error("flag mismatch: fixtures say $flags, runner derives $derived")

    # --- kernels (same call shapes as the Python wrappers) -------------------
    println("[1/4] process_cells (SRC accumulators)...")
    hid_dict_array = hcat(hydroids, Float64.(0:length(hydroids)-1))
    t_start = time()
    counts, surface, bed, volume = process_cells(
        hand, catchments, slope, hid_dict_array, heights, cell_area)
    println("      done in $(round(time() - t_start; digits=2))s")

    println("[2/4] jit_inun (inundation depth)...")
    t_start = time()
    inun = jit_inun(hand, catchments, inun_hydroids, inun_stages)
    println("      done in $(round(time() - t_start; digits=2))s")

    println("[3/4] jit_binary_hand (two-sweep chamfer)...")
    t_start = time()
    binary_hand = PygeofloodKernels.jit_binary_hand(dem, flowline_raster, nodata)
    println("      done in $(round(time() - t_start; digits=2))s")

    # --- gates ----------------------------------------------------------------
    println("[4/4] gates (mirrors tests/test_golden_equivalence.py)...")
    results = Dict{String, Tuple{Bool, String}}()
    results["src_csv"] = gate_src_csv(
        corpus_csv, hydroids, heights, counts, surface, bed, volume)
    results["inundation_map"] = gate_inundation_map!(inun, golden_fim)
    results["binary_hand"] = gate_binary_hand(
        binary_hand, golden_binary_hand, dem, nodata)

    # --- discrimination: each perturbed candidate must FAIL its gate ---------
    disc = Dict{String, Bool}()
    let v = copy(volume)
        v[2, 1] = Float32(Float64(v[2, 1]) * (1.0 + 1e-4))
        disc["src_csv"] = !gate_src_csv(
            corpus_csv, hydroids, heights, counts, surface, bed, v)[1]
    end
    let perturbed = copy(inun)
        wet = findall(x -> x != Float32(FIM_NODATA) && !isnan(x), perturbed)
        if !isempty(wet)
            perturbed[wet[1]] =
                Float32(Float64(perturbed[wet[1]]) * 1.001)
        end
        disc["inundation_map"] = !gate_inundation_map!(perturbed, golden_fim)[1]
    end
    let perturbed = copy(binary_hand)
        i = findfirst(==(0), perturbed)
        i === nothing && (i = findfirst(==(1), perturbed))
        perturbed[i] = perturbed[i] == 0 ? 1 : 0
        disc["binary_hand"] = !gate_binary_hand(
            perturbed, golden_binary_hand, dem, nodata)[1]
    end

    # --- report ---------------------------------------------------------------
    all_pass = all(v[1] for v in values(results))
    all_disc = all(values(disc))
    for (name, (passed, detail)) in sort(collect(results); by=first)
        @printf("%-18s %s — %s\n", name, passed ? "PASS" : "FAIL", detail)
    end
    for (name, ok) in sort(collect(disc); by=first)
        @printf("%-18s discrimination %s\n", name, ok ? "ok" : "FAILED")
    end
    println(all_pass && all_disc ? "EQUIVALENCE: PASS" : "EQUIVALENCE: FAIL")

    if report_path !== nothing
        open(report_path, "w") do io
            write(io, "{\n")
            write(io, "  \"julia_version\": \"$(VERSION)\",\n")
            write(io, "  \"julia_threads\": $(Threads.nthreads()),\n")
            for (name, (passed, detail)) in sort(collect(results); by=first)
                detail_json = escape_string(detail)
                write(io,
                    "  \"$name\": {\"pass\": $passed, \"detail\": \"$detail_json\"},\n")
            end
            for (name, ok) in sort(collect(disc); by=first)
                write(io, "  \"discrimination_$name\": $ok,\n")
            end
            write(io, "  \"equivalence_pass\": $(all_pass && all_disc)\n}\n")
        end
        println("report: $report_path")
    end

    (all_pass && all_disc) || exit(1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
