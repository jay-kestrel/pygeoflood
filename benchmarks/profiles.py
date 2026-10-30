"""Workload profiles: the full OC1mTest tile and a fixed 2000x2000 crop.

Both profiles run the canonical workload (run_fim_workflow with
custom_Q=500) from a private work directory so no run ever reads another
run's intermediate outputs. The crop profile derives its window and its
flowline deterministically from the input bundle: a 2000x2000 window
centered on the first flowline vertex (clamped into the DEM, snapped to
a 100-cell grid), and the flowline clipped to that window, keeping the
longest resulting piece. The clip is required, not cosmetic: the full
line spans 3360x2252 cells, no 2000x2000 window can contain it, and
extract_channel_network traces the flowline end to end, which requires
both endpoints inside the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema import (
    COBOL_CROP_SIZE,
    CROP_SIZE,
    PROFILE_COBOL,
    PROFILE_CROP,
    PROFILE_FULL,
)

# In-repo input bundle (committed at 3650c9b).
INPUTS_DIR = Path("tests") / "test_data" / "OC1mTest_inputs"
DEM_NAME = "OC1mTest.tif"
FLOWLINE_NAME = "Flowline.shp"
CATCHMENT_NAME = "Catchment.shp"

CROP_STEM = "OC1mTest_crop2000"


@dataclass(frozen=True)
class PreparedProfile:
    """Everything a benchmark run needs after workspace preparation."""

    profile: str
    dem_path: Path
    flowline_path: Path
    catchment_path: Path
    run_dir: Path
    crop: dict | None  # {"origin_row", "origin_col", "height", "width"}


def prepare_profile(
    profile: str, repo_root: Path, work_root: Path
) -> PreparedProfile:
    """Materialize the profile's DEM copy (and crop, for the crop profile)
    under *work_root* and return the prepared paths."""
    repo_root = Path(repo_root)
    inputs = repo_root / INPUTS_DIR
    dem_src = inputs / DEM_NAME
    flowline_src = inputs / FLOWLINE_NAME
    catchment_src = inputs / CATCHMENT_NAME
    for path in (dem_src, flowline_src, catchment_src):
        if not path.is_file():
            raise FileNotFoundError(f"benchmark input missing: {path}")

    profile_dir = work_root / profile
    profile_dir.mkdir(parents=True, exist_ok=True)

    if profile == PROFILE_FULL:
        dem_path = profile_dir / DEM_NAME
        if not dem_path.is_file():
            shutil_copy(dem_src, dem_path)
        return PreparedProfile(
            profile=profile,
            dem_path=dem_path,
            flowline_path=flowline_src,
            catchment_path=catchment_src,
            run_dir=profile_dir,
            crop=None,
        )
    if profile == PROFILE_CROP:
        return _prepare_crop(
            profile_dir=profile_dir,
            dem_src=dem_src,
            flowline_src=flowline_src,
            catchment_src=catchment_src,
        )
    if profile == PROFILE_COBOL:
        return _prepare_crop(
            profile_dir=profile_dir,
            dem_src=dem_src,
            flowline_src=flowline_src,
            catchment_src=catchment_src,
            size=COBOL_CROP_SIZE,
        )
    raise ValueError(f"unknown profile: {profile!r}")


def _prepare_crop(
    profile_dir: Path,
    dem_src: Path,
    flowline_src: Path,
    catchment_src: Path,
    *,
    size: int = CROP_SIZE,
) -> PreparedProfile:
    import rasterio
    from rasterio.windows import Window

    stem = f"OC1mTest_crop{size}"
    dem_path = profile_dir / f"{stem}.tif"
    with rasterio.open(dem_src) as src:
        profile = src.profile

    crop = _compute_crop_window(dem_src, flowline_src, size)
    window = Window(
        col_off=crop["origin_col"],
        row_off=crop["origin_row"],
        width=crop["width"],
        height=crop["height"],
    )

    if not dem_path.is_file():
        with rasterio.open(dem_src) as src:
            data = src.read(
                1,
                window=window,
            )
            out_profile = profile.copy()
            out_profile.update(
                height=crop["height"],
                width=crop["width"],
                transform=src.window_transform(window),
            )
        with rasterio.open(dem_path, "w", **out_profile) as dst:
            dst.write(data, 1)

    flowline = _clip_flowline(
        flowline_src,
        profile_dir / f"{stem}_flowline.shp",
        dem_path,
    )
    crop["flowline_original_vertices"] = flowline["original_vertices"]
    crop["flowline_clipped_vertices"] = flowline["vertices"]

    return PreparedProfile(
        profile=profile_dir.name,
        dem_path=dem_path,
        flowline_path=flowline["path"],
        catchment_path=catchment_src,
        run_dir=profile_dir,
        crop=crop,
    )


def _clip_flowline(src_path: Path, dst_path: Path, dem_path: Path) -> dict:
    """Clip the flowline to the crop window, keeping the longest resulting
    piece as a single-feature LineString shapefile (written once and
    reused across runs). extract_channel_network traces the flowline end
    to end and requires both endpoints inside the grid; the unclipped
    line never fits a 2000x2000 window."""
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    with rasterio.open(dem_path) as src:
        res_x, res_y = src.res
        bounds = src.bounds

    # Inset the clip box by one cell: a vertex sitting exactly on the
    # window boundary rasterizes one past the last index (rasterio rowcol
    # floors the map coordinate), and route_through_array then rejects the
    # endpoint. One cell of inset keeps every clipped vertex inside the
    # grid deterministically.
    window_box = box(
        bounds.left + res_x,
        bounds.bottom + res_y,
        bounds.right - res_x,
        bounds.top - res_y,
    )

    gdf = gpd.read_file(src_path)
    original_vertices = 0
    lines = []
    for geom in gdf.geometry:
        original_vertices += len(getattr(geom, "coords", []))
        clipped = geom.intersection(window_box)
        if clipped.is_empty:
            continue
        pieces = clipped.geoms if clipped.geom_type == "MultiLineString" else [clipped]
        lines.extend(g for g in pieces if g.geom_type == "LineString")
    if not lines:
        raise RuntimeError(
            "crop window does not intersect the flowline — cannot build the crop profile"
        )
    longest = max(lines, key=lambda g: g.length)
    out = gpd.GeoDataFrame(geometry=[longest], crs=gdf.crs)
    out.to_file(dst_path)
    return {
        "path": dst_path,
        "vertices": len(longest.coords),
        "original_vertices": original_vertices,
    }


def _compute_crop_window(
    dem_path: Path, flowline_path: Path, size: int
) -> dict:
    """Deterministic 2000x2000 window centered on the first flowline
    vertex, clamped into the DEM and snapped to a 100-cell grid so the
    origin is stable across machines reading identical inputs."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(dem_path) as src:
        height, width = src.height, src.width
        transform = src.transform

    gdf = gpd.read_file(flowline_path)
    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiLineString":
        geom = geom.geoms[0]
    # The source is a measured 3D LineString — pyogrio hands back (x, y, z)
    # tuples; the window origin only needs the first two ordinates.
    x, y = geom.coords[0][:2]
    row, col = rowcol(transform, x, y)

    def clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(int(value), maximum))

    origin_row = clamp(row - size // 2, 0, height - size)
    origin_col = clamp(col - size // 2, 0, width - size)
    origin_row = (origin_row // 100) * 100
    origin_col = (origin_col // 100) * 100
    return {
        "origin_row": origin_row,
        "origin_col": origin_col,
        "height": size,
        "width": size,
    }


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)
