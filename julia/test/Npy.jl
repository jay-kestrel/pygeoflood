# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
# Mark Wang, and Matthew Preisser.
"""Minimal .npy reader for the equivalence runners (stdlib only).

Supports the NumPy 1.0 writer output produced by `np.save`: version 1-3
headers, little-endian numeric dtypes, C-order (fortran_order=false) 1-D and
2-D arrays — the only shapes the fixture dumper emits.
"""
module Npy

const DESCRIPTORS = Dict{String, DataType}(
    "<f4" => Float32,
    "<f8" => Float64,
    "<i1" => Int8,
    "<i2" => Int16,
    "<i4" => Int32,
    "<i8" => Int64,
    "<u1" => UInt8,
    "<u2" => UInt16,
    "<u4" => UInt32,
    "<u8" => UInt64,
    "|b1" => Bool,
)

"""Read a .npy file into a Julia `Vector` (1-D) or `Matrix` (2-D, row-major
file order transposed into Julia's column-major layout)."""
function read(path::AbstractString)
    raw = open(path, "r") do io
        magic = String(Base.read(io, 6))
        magic == "\x93NUMPY" || error("$path: not an npy file")
        major = Base.read(io, UInt8)
        Base.read(io, UInt8)  # minor
        header_len = major >= 3 ? Base.read(io, UInt64) :
                     major == 2 ? Base.read(io, UInt32) : Base.read(io, UInt16)
        header = String(Base.read(io, header_len))
        descr, fortran, shape = _parse_header(header, path)
        flat = Vector{eltype_type(descr)}(undef, prod(shape))
        Base.read!(io, flat)  # little-endian file, little-endian host
        return (descr, fortran, shape, flat)
    end
    descr, fortran, shape, flat = raw
    if length(shape) == 1
        return reshape(flat, shape[1])
    elseif length(shape) == 2
        fortran && error("$path: fortran-order arrays unsupported")
        nrows, ncols = shape
        # file is row-major: row i, col j lives at flat[(i-1)*ncols + j]
        return permutedims(reshape(flat, ncols, nrows))  # (nrows, ncols)
    end
    error("$path: only 1-D and 2-D arrays supported (got ndim=$(length(shape)))")
end

function eltype_type(descr::AbstractString)
    t = get(DESCRIPTORS, String(descr), nothing)
    t isa DataType || error("unsupported npy descriptor '$descr'")
    return t
end

"""Parse the npy header dict — `{'descr': '<f4', 'fortran_order': False,
'shape': (2000, 2000), }` (Python literal syntax; enough for this writer)."""
function _parse_header(header::String, path::AbstractString)
    descr_m = match(r"'descr'\s*:\s*'([^']+)'", header)
    descr_m === nothing && error("$path: missing descr in npy header")
    descr = descr_m.captures[1]
    fortran = occursin(r"'fortran_order'\s*:\s*True", header)
    shape_m = match(r"'shape'\s*:\s*\(([^)]*)\)", header)
    shape_m === nothing && error("$path: missing shape in npy header")
    dims = strip(shape_m.captures[1])
    shape = dims == "" ? Int[] :
        [parse(Int, strip(d)) for d in split(dims, ",") if strip(d) != ""]
    # single-element tuples write as "(n,)" — trailing comma already split off
    return descr, fortran, shape
end

end # module
