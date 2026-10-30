# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
# Mark Wang, and Matthew Preisser.
"""Dump kernel-boundary fixtures for the Julia equivalence runners.

Reads a generated golden corpus directory (tests/golden/data/run<N>, produced
by tests/golden/generate_golden_fixtures.py) plus the corpus's cropped inputs,
and writes the exact arrays the ported kernels consume and must reproduce:

  process_cells   hand, segment_catchments, slope (unfiltered-DEM gradient),
                  hydroid table, heights, cell_area, dtype-precision flags
  jit_inun        hand, segment_catchments, HYDROID/stage vectors from the
                  golden flood-stage CSV, plus the golden inundation map
  jit_binary_hand DEM (read_raster NaN convention), rasterized flowline,
                  nodata sentinel, plus the golden binary HAND raster

Everything lands as .npy (byte-exact dtypes) plus fixtures.json describing
flags, cell_area, nodata, shapes, and per-file sha256 hashes.

This script is a test oracle: it imports pygeoflood tools read-only and never
modifies package source. Usage:

    python julia/tools/dump_kernel_fixtures.py \
        --corpus-dir tests/golden/data/run1 --out julia/test/fixtures
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Single-thread reference condition, enforced before numba is imported.
if os.environ.get("NUMBA_NUM_THREADS", "1") != "1":
    raise SystemExit(
        "NUMBA_NUM_THREADS must be 1 for fixture dumping (got "
        f"{os.environ['NUMBA_NUM_THREADS']!r}): the golden corpus is a "
        "single-thread reference."
    )
os.environ["NUMBA_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402

from pygeoflood.tools import (  # noqa: E402
    compute_dem_slope,
    rasterize_flowline,
    read_raster,
)

CROP_STEM = "OC1mTestCrop"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(out_dir, name, array):
    path = out_dir / f"{name}.npy"
    np.save(path, array)
    return {
        "name": name,
        "dtype": str(np.asarray(array).dtype),
        "shape": list(np.asarray(array).shape),
        "sha256": sha256_file(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    corpus = args.corpus_dir.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- rasters through the package's own reader (NaN/nodata semantics) ----
    dem_path = corpus / f"{CROP_STEM}.tif"
    dem, dem_profile = read_raster(dem_path)
    if dem.ndim != 2:
        raise SystemExit("expected a single-band DEM crop")
    nodata = float(dem_profile["nodata"])

    segment_catchments, _ = read_raster(corpus / f"{CROP_STEM}_segment_catchments.tif")
    hand, _ = read_raster(corpus / f"{CROP_STEM}_HAND.tif")

    # --- SRC kernel boundary (mirrors calculate_src in pygeoflood.py) -------
    unfilt_slope = compute_dem_slope(dem, dem_profile["transform"].a, verbose=False)
    unfilt_slope = np.nan_to_num(unfilt_slope, nan=0)
    heights = np.arange(0, 20 + 0.1, 0.1)
    cell_area = abs(
        dem_profile["transform"].a * dem_profile["transform"].e
    )
    # hydroid table: unique raster values minus the profile nodata sentinel,
    # matching get_river_attributes + catchhydrogeo (sorted ascending; the
    # second column of hid_dict_array is never read by the kernel)
    hydroid_values, _ = np.unique(segment_catchments, return_counts=True)
    hydroids = hydroid_values[hydroid_values != dem_profile["nodata"]].astype(
        np.float64
    )
    slope_f32 = unfilt_slope.dtype == np.float32
    delta_f32 = hand.dtype == np.float32 and heights.dtype == np.float32
    volume_f32 = delta_f32 and isinstance(cell_area, np.float32)

    # --- inundation kernel boundary (mirrors inundate in pygeoflood.py) -----
    flood_df_lines = (
        corpus / f"{CROP_STEM}_flood_stage.csv"
    ).read_text().splitlines()
    header = flood_df_lines[0].split(",")
    hydroid_col = header.index("HYDROID")
    stage_col = header.index("Stage_m")
    rows = [line.split(",") for line in flood_df_lines[1:] if line]
    order = sorted(range(len(rows)), key=lambda k: float(rows[k][hydroid_col]))
    inun_hydroids = np.array(
        [float(rows[k][hydroid_col]) for k in order], dtype=np.float64
    )
    inun_stages = np.array(
        [float(rows[k][stage_col]) for k in order], dtype=np.float64
    )

    # --- binary-HAND kernel boundary (mirrors get_binary_hand) --------------
    flowline_gdf = gpd.read_file(corpus / f"{CROP_STEM}_flowline.shp")
    flowline_raster = rasterize_flowline(
        flowline_gdf=flowline_gdf, ref_profile=dem_profile
    )

    files = [
        save(out_dir, "hand", hand),
        save(out_dir, "segment_catchments", segment_catchments),
        save(out_dir, "slope", unfilt_slope),
        save(out_dir, "hydroids", hydroids),
        save(out_dir, "heights", heights),
        save(out_dir, "inun_hydroids", inun_hydroids),
        save(out_dir, "inun_stages", inun_stages),
        save(out_dir, "dem", dem),
        save(out_dir, "flowline_raster", flowline_raster),
    ]

    # golden outputs for the gates
    import rasterio as rio

    def read_golden(name, path):
        with rio.open(path) as src:
            array = src.read(1)
        files.append(save(out_dir, name, array))
        return array

    read_golden("golden_fim", corpus / f"{CROP_STEM}_fim.tif")
    read_golden("golden_binary_hand", corpus / f"{CROP_STEM}_binary_hand.tif")

    meta = {
        "cell_area": cell_area,
        "nodata": nodata,
        "flags": {
            "slope_f32": bool(slope_f32),
            "delta_f32": bool(delta_f32),
            "volume_f32": bool(volume_f32),
        },
        "nheights": int(heights.size),
        "ncatchments": int(hydroids.size),
        "fim_nodata": -32768.0,
        "files": files,
    }
    (out_dir / "fixtures.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"fixtures: {len(files)} arrays + fixtures.json -> {out_dir}")


if __name__ == "__main__":
    main()
