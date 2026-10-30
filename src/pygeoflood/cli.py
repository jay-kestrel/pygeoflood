"""CLI for pygeoflood quality-of-life helpers.

    pygeoflood version                 # package, Python, dependency, and WhiteboxTools versions
    pygeoflood list-options            # every method/option PGF_Config recognizes
    pygeoflood check-inputs DEM.tif [--flowline F.shp] [--catchment C.shp]

    python -m pygeoflood <command> ...  # equivalent

Commands are read-only: check-inputs validates inputs and never writes.
Heavy imports live inside command bodies; note the console script itself
still pays the pygeoflood package import (src/pygeoflood/__init__.py
pulls the geo stack) — making that import lazy is outside this PR's
additive scope.

Style mirrors benchmarks/__main__.py (usage docstring, parse_args(argv),
main(argv) -> int, sys.exit(main())).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

# Same library set the benchmark harness records in every run manifest
# (benchmarks/manifest.py PACKAGES), so `pygeoflood version` answers the
# same "what exact environment?" question without a benchmark run. The
# harness list lives outside the installed package, so it is mirrored here
# and a test (benchmarks/tests/test_cli.py) guards against drift.
PACKAGES = (
    "pygeoflood",
    "numpy",
    "numba",
    "scipy",
    "scikit-image",
    "scikit-fmm",
    "rasterio",
    "geopandas",
    "pandas",
    "xarray",
    "netcdf4",
    "fiona",
    "shapely",
    "whitebox",
    "psutil",
)


def package_versions() -> dict[str, str | None]:
    """Installed version per PACKAGES entry; None when absent.

    Mirrors benchmarks/manifest.py package_versions().
    """
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def wbt_lines() -> list[str]:
    """WhiteboxTools status lines, degrading gracefully when absent.

    Mirrors benchmarks/manifest.py wbt_versions(): the binary is *run* for
    its version, and WBT_PATH is honored the way
    pygeoflood.tools.get_WhiteboxTools honors it, since the resolved
    binary is load-bearing (equivalence CI pins v2.4.0 via WBT_PATH).
    """
    wbt_path_env = os.getenv("WBT_PATH")
    lines = [f"  WBT_PATH: {wbt_path_env if wbt_path_env else 'not set'}"]
    try:
        from whitebox import WhiteboxTools
    except ImportError:
        lines.append("  package: not installed")
        lines.append("  binary: unavailable (whitebox package missing)")
        return lines

    wbt = WhiteboxTools()
    if wbt_path_env:
        wbt.set_whitebox_dir(wbt_path_env)  # mirror tools.get_WhiteboxTools
    try:
        exe_path = str(Path(wbt.exe_path))
    except Exception as error:  # attribute/layout drift across whitebox versions
        exe_path = f"not resolved: {error}"
    # version() runs the binary and echoes its banner via print(); keep
    # only the returned version line so the report stays clean.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            returned = wbt.version()
        # broken-binary runs return the exception object instead of raising
        if isinstance(returned, Exception):
            binary_version = f"unavailable: {returned}"
        else:
            binary_version = str(returned).strip().splitlines()[0]
    except Exception as error:  # missing/broken binary is a status, not a crash
        binary_version = f"unavailable: {error}"
    lines.append(f"  package: {metadata.version('whitebox')}")
    lines.append(f"  binary: {binary_version}")
    lines.append(f"  binary path: {exe_path}")
    return lines


def cmd_version(args: argparse.Namespace) -> int:
    """Print environment versions from package metadata — pygeoflood has
    no __version__ attribute, so importlib.metadata is the source of truth."""
    versions = package_versions()
    print(f"pygeoflood {versions['pygeoflood']}")
    print(f"Python {platform.python_version()} ({platform.platform()})")
    print()
    print("Dependencies (same set as benchmarks/manifest.py PACKAGES):")
    for name, version in versions.items():
        print(f"  {name:<16} {version if version else 'not installed'}")
    print()
    print("WhiteboxTools:")
    for line in wbt_lines():
        print(line)
    return 0


def option_lines() -> list[str]:
    """Every method PGF_Config recognizes, with its recognized options.

    pgf_params is derived from the pyGeoFlood class at import time
    (pygeoflood.py), so this command needs the package import. Keeping it
    out of module scope is what lets `pygeoflood --help` skip the geo
    stack.
    """
    from .pygeoflood import pgf_params

    lines = [
        "Methods and options PGF_Config recognizes"
        " (unrecognized names are dropped with only a print warning):",
        "",
    ]
    for method, params in pgf_params.items():
        lines.append(f"  {method}: {', '.join(params) if params else '(no options)'}")
    lines.append("")
    lines.append(
        f"{len(pgf_params)} methods. Use these names as PGF_Config(...) keys."
    )
    return lines


def cmd_list_options(args: argparse.Namespace) -> int:
    for line in option_lines():
        print(line)
    return 0


def check_dem(dem_path: Path) -> tuple[list[str], dict | None]:
    """Validate the DEM (rasterio, read-only).

    Returns report lines plus a context dict (transform/bounds/shape/crs)
    for the vector checks, or None when the DEM is unusable so downstream
    checks are skipped instead of producing garbage.
    """
    import rasterio

    if not dem_path.is_file():
        return [f"ERROR: DEM not found: {dem_path}"], None
    try:
        with rasterio.open(dem_path) as src:
            context = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "shape": (src.height, src.width),
            }
            nodata = src.nodata
    except Exception as error:
        return [f"ERROR: DEM failed to open: {dem_path} ({error})"], None

    height, width = context["shape"]
    lines = [
        f"DEM: {dem_path}",
        f"  {height}x{width} pixels, bounds"
        f" {tuple(round(value, 2) for value in context['bounds'])}",
        f"  CRS: {context['crs']}",
        f"  nodata: {nodata}",
    ]
    if context["crs"] is None:
        lines.append(
            "ERROR: DEM has no CRS — vector checks need georeferencing to mean anything"
        )
        return lines, None
    if nodata is None:
        lines.append("WARNING: DEM has no explicit nodata value")
    return lines, context


def flowline_endpoint_lines(
    gdf: "gpd.GeoDataFrame",  # noqa: F821 - geopandas is imported lazily by callers
    dem_context: dict,
) -> list[str]:
    """Endpoints must convert to a cell inside the DEM grid.

    Endpoint extraction mirrors pygeoflood.tools.get_endpoints (explode
    MultiLineStrings, take first/last coordinate). The workflow routes
    least-cost paths from the converted cells: an endpoint exactly on the
    grid edge converts to a row/col one cell outside the cost array and
    crashes the run (tests/golden/README.md), so such endpoints fail here.
    """
    from rasterio.transform import rowcol

    # get_endpoints only explodes homogeneous MultiLineString frames; a
    # mixed frame would crash it mid-run, so explode unconditionally here.
    if any(geometry.geom_type == "MultiLineString" for geometry in gdf.geometry):
        gdf = gdf.explode()

    transform = dem_context["transform"]
    height, width = dem_context["shape"]
    lines: list[str] = []
    endpoint_count = 0
    for feature_idx, geometry in enumerate(gdf.geometry, start=1):
        if geometry is None or geometry.geom_type not in ("LineString", "LinearRing"):
            kind = "missing geometry" if geometry is None else geometry.geom_type
            lines.append(
                f"ERROR: flowline feature {feature_idx} has {kind} —"
                " expected (Multi)LineString"
            )
            continue
        # index, not unpack: committed flowlines carry Z/M coordinates
        for label, coord in (
            ("start", geometry.coords[0]),
            ("end", geometry.coords[-1]),
        ):
            endpoint_count += 1
            x, y = coord[0], coord[1]
            if not (math.isfinite(x) and math.isfinite(y)):
                lines.append(
                    f"ERROR: flowline feature {feature_idx} {label} endpoint has"
                    f" non-finite coordinates ({x}, {y})"
                )
                continue
            row, col = rowcol(transform, x, y)
            if not (0 <= row < height and 0 <= col < width):
                lines.append(
                    f"ERROR: flowline feature {feature_idx} {label} endpoint"
                    f" ({x:.2f}, {y:.2f}) maps outside the DEM grid"
                    f" (row {row}, col {col} in a {height}x{width} grid) —"
                    " endpoints exactly on the grid edge crash least-cost"
                    " routing (see tests/golden/README.md)"
                )
    if not any(line.startswith("ERROR:") for line in lines):
        lines.append(
            f"  flowline endpoints: all {endpoint_count} inside the DEM grid"
        )
    return lines


def check_vector(
    vector_path: Path,
    kind: str,
    dem_context: dict | None,
) -> list[str]:
    """Validate a vector input (geopandas, read-only).

    Mirrors tests/test_functionality.py validate_vector_file: opens the
    file and requires at least one feature. The endpoint check runs only
    when the vector's CRS matches the DEM's — converting coordinates
    across CRSes would be garbage, and the mismatch itself is what the
    user needs to fix first.
    """
    import geopandas as gpd

    if not vector_path.is_file():
        return [f"ERROR: {kind} not found: {vector_path}"]
    try:
        gdf = gpd.read_file(vector_path)
    except Exception as error:
        return [f"ERROR: {kind} failed to open: {vector_path} ({error})"]

    lines = [f"{kind}: {vector_path}"]
    lines.append(f"  {len(gdf)} feature(s), CRS: {gdf.crs}")
    if len(gdf) == 0:
        lines.append(f"ERROR: {kind} has no features")
        return lines
    if dem_context is None:
        return lines
    if gdf.crs is None:
        lines.append(
            f"WARNING: {kind} has no CRS — endpoint check skipped"
            " (pygeoflood assumes vectors share the DEM CRS)"
        )
        return lines
    if gdf.crs != dem_context["crs"]:
        lines.append(
            f"WARNING: {kind} CRS ({gdf.crs}) differs from DEM CRS"
            f" ({dem_context['crs']}) — pygeoflood assumes they match;"
            " endpoint check skipped"
        )
        return lines
    if kind == "flowline":
        lines.extend(flowline_endpoint_lines(gdf, dem_context))
    return lines


def check_inputs(
    dem_path: Path,
    flowline: Path | None = None,
    catchment: Path | None = None,
) -> list[str]:
    """Full pre-flight report; lines prefixed 'ERROR:'/'WARNING:'.

    Read-only by construction: every check only opens its input.
    """
    lines, dem_context = check_dem(dem_path)
    for path, kind in ((flowline, "flowline"), (catchment, "catchment")):
        if path is not None:
            lines.extend(check_vector(path, kind, dem_context))
    return lines


def cmd_check_inputs(args: argparse.Namespace) -> int:
    lines = check_inputs(args.dem, flowline=args.flowline, catchment=args.catchment)
    for line in lines:
        print(line)
    errors = sum(1 for line in lines if line.startswith("ERROR:"))
    if errors:
        print(f"check-inputs: FAILED with {errors} problem(s)")
        return 1
    print("check-inputs: OK")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pygeoflood",
        description=(
            "Quality-of-life commands around the pygeoflood FIM workflow."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="command"
    )

    version_parser = subparsers.add_parser(
        "version",
        help="print pygeoflood, Python, dependency, and WhiteboxTools versions",
        description=(
            "Print pygeoflood, Python, and dependency versions (same package"
            " set as benchmarks/manifest.py PACKAGES) plus the resolved"
            " WhiteboxTools binary version and path. The binary is queried,"
            " never the workflow."
        ),
    )
    version_parser.set_defaults(handler=cmd_version)

    options_parser = subparsers.add_parser(
        "list-options",
        help="list every method and option PGF_Config recognizes",
        description=(
            "List every pyGeoFlood method and the option names PGF_Config"
            " recognizes, so typos surface here instead of as one-line"
            " warnings from update_options in a long notebook run."
        ),
    )
    options_parser.set_defaults(handler=cmd_list_options)

    check_parser = subparsers.add_parser(
        "check-inputs",
        help="read-only pre-flight validation for the FIM workflow",
        description=(
            "Validate FIM workflow inputs before the 25-90s+ pipeline: the"
            " DEM opens with a CRS, vectors open with features, and every"
            " flowline endpoint lands inside the DEM grid. Writes nothing."
        ),
    )
    check_parser.add_argument("dem", type=Path, help="path to the DEM raster")
    check_parser.add_argument(
        "--flowline",
        type=Path,
        default=None,
        help="flowline vector; endpoints are checked against the DEM grid",
    )
    check_parser.add_argument(
        "--catchment",
        type=Path,
        default=None,
        help="catchment vector (opened and counted only)",
    )
    check_parser.set_defaults(handler=cmd_check_inputs)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
