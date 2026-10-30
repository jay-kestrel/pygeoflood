#!/usr/bin/env python3
"""Generate golden reference fixtures for the pygeoflood equivalence suite.

Deliverable D2 of the pygeoflood port spec (art_4Ga5fqvh): deterministic
reference outputs from *unmodified* pygeoflood, produced with
NUMBA_NUM_THREADS=1. The single-thread constraint is load-bearing -- the
parallel ``process_cells`` kernel has a confirmed data race and
thread-count-dependent FP accumulation, so multi-thread SRC output cannot
serve as a golden reference (findings art_0v8nRJ7D, Correctness findings).

Pipeline inputs are a fixed 2000x2000 crop of the bundled OC1mTest DEM
(centered window; see the constants below) with the flowline and catchment
vectors clipped to the same extent, so the goldens exercise the full
``run_fim_workflow`` on a small but real stream reach. Flood stage is
computed from a constant ``custom_Q`` (the canonical example value), which
keeps the fixture self-contained and deterministic without the CONUS-wide
NWM forecast file.

Usage:
    python tests/golden/generate_golden_fixtures.py --output-dir <dir>

The output directory receives the cropped inputs, all workflow outputs, a
``versions.json`` environment record, and ``manifest.sha256`` -- the sha256
manifest of the golden artifact classes, one line per artifact:

    <sha256>  <artifact class>  <filename>

Regeneration must reproduce the pinned ``tests/golden/golden_manifest.sha256``
byte-identically; the equivalence suite (tests/test_golden_equivalence.py)
enforces this, along with the tolerance-table gates from the spec's
Verification section.

Requires the pinned WhiteboxTools v2.4.0 binary, resolved via the WBT_PATH
environment variable (the ``whitebox`` package honors it; see
src/pygeoflood/tools.py). Regeneration fails if the resolved binary is any
other version.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

# The single-thread reference is the point of the exercise. Enforce the
# pin *before* numba is imported (pygeoflood imports it transitively),
# because numba reads NUMBA_NUM_THREADS at import time.
if os.environ.get("NUMBA_NUM_THREADS", "1") != "1":
    raise SystemExit(
        "NUMBA_NUM_THREADS must be 1 for golden generation (got "
        f"{os.environ['NUMBA_NUM_THREADS']!r}): the parallel process_cells "
        "kernel is thread-count sensitive and cannot serve as a "
        "deterministic reference."
    )
os.environ["NUMBA_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import geopandas as gpd  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import rasterio as rio  # noqa: E402
import scipy  # noqa: E402
import skimage  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.geometry import box  # noqa: E402

from pygeoflood import PGF_Config, pyGeoFlood  # noqa: E402

# --- fixed crop window (changing it invalidates the pinned manifest) ------
WINDOW_ROWS = 2000
WINDOW_COLS = 2000
# centered on the OC1mTest DEM: (3378 - 2000) // 2 and (4750 - 2000) // 2
EXPECTED_DEM_SHAPE = (3378, 4750)
ROW_START = 689
COL_START = 1375
# canonical-example constant discharge for calculate_flood_stage
CUSTOM_Q = 500.0
# the flowline is clipped to the window inset by this margin (metres): a
# clipped endpoint lying exactly on the window edge converts to a row/col
# one cell outside the cost array (rasterio rowcol of 2000.0), so the
# least-cost routing in extract_channel_network needs endpoints strictly
# interior to the DEM crop
VECTOR_CLIP_INSET_M = 10.0

INPUT_DIR = REPO_ROOT / "tests" / "test_data" / "OC1mTest_inputs"
CROP_STEM = "OC1mTestCrop"

WBT_PINNED_VERSION = "2.4.0"  # spec: pin WhiteboxTools v2.4.0

# golden artifact classes (spec D2) -> workflow output filename
ARTIFACT_CLASSES = {
    "hand_raster": f"{CROP_STEM}_HAND.tif",
    "inundation_map": f"{CROP_STEM}_fim.tif",
    "src_csv": f"{CROP_STEM}_src.csv",
    "flood_stage_csv": f"{CROP_STEM}_flood_stage.csv",
    "wbt_d8_fdr": f"{CROP_STEM}_d8_fdr.tif",
    "wbt_mfd_fac": f"{CROP_STEM}_mfd_fac.tif",
    "wbt_basins": f"{CROP_STEM}_basins.tif",
    "wbt_segment_catchments": f"{CROP_STEM}_segment_catchments.tif",
    "binary_hand": f"{CROP_STEM}_binary_hand.tif",
    "channel_network_raster": f"{CROP_STEM}_channel_network_raster.tif",
}


def sha256_file(path):
    """Return the hex sha256 digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment(output_dir):
    """Verify pinned numba threading and WhiteboxTools version."""
    if numba.get_num_threads() != 1:
        # env var pins the default at import; this guards an in-process
        # caller that imported numba earlier with a different default
        numba.set_num_threads(1)
    if numba.get_num_threads() != 1:
        raise SystemExit("numba thread pool could not be pinned to 1 thread")

    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt_path = os.getenv("WBT_PATH")
    if wbt_path:
        wbt.set_whitebox_dir(wbt_path)
    else:
        print(
            "WARNING: WBT_PATH is not set; the whitebox package will "
            "resolve (and possibly download) its own binary."
        )
    version_banner = wbt.version() or ""
    first_line = version_banner.strip().splitlines()[0] if version_banner.strip() else ""
    if f"v{WBT_PINNED_VERSION}" not in first_line:
        raise SystemExit(
            f"WhiteboxTools v{WBT_PINNED_VERSION} is pinned for golden "
            f"generation but the resolved binary reports: {first_line!r}. "
            "Point WBT_PATH at the pinned binary."
        )
    # environment record (informational sidecar; not part of the manifest)
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numba_threads": numba.get_num_threads(),
        "whitebox_tools": first_line,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "numba": numba.__version__,
        "rasterio": rio.__version__,
        "geopandas": gpd.__version__,
    }
    (output_dir / "versions.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n"
    )


def prepare_inputs(output_dir):
    """Crop the OC1m DEM to the fixed window and clip vectors to match.

    Inputs are derived deterministically from the committed test data, so
    they are rebuilt on every run rather than stored.
    """
    dem_in = INPUT_DIR / "OC1mTest.tif"
    crop_dem = output_dir / f"{CROP_STEM}.tif"
    with rio.open(dem_in) as src:
        if (src.height, src.width) != EXPECTED_DEM_SHAPE:
            raise SystemExit(
                f"unexpected DEM shape {(src.height, src.width)}; the fixed "
                f"crop window assumes {EXPECTED_DEM_SHAPE}"
            )
        window = Window(COL_START, ROW_START, WINDOW_COLS, WINDOW_ROWS)
        window_bounds = src.window_bounds(window)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(height=WINDOW_ROWS, width=WINDOW_COLS,
                       transform=src.window_transform(window))
        with rio.open(crop_dem, "w", **profile) as dst:
            dst.write(data, 1)

    extent = box(*window_bounds)
    inset = VECTOR_CLIP_INSET_M
    vector_extent = box(
        extent.bounds[0] + inset,
        extent.bounds[1] + inset,
        extent.bounds[2] - inset,
        extent.bounds[3] - inset,
    )
    for src_name, out_name in [
        ("Flowline.shp", f"{CROP_STEM}_flowline.shp"),
        ("Catchment.shp", f"{CROP_STEM}_catchment.shp"),
    ]:
        gdf = gpd.read_file(INPUT_DIR / src_name)
        clipped = gdf.clip(vector_extent)
        if clipped.empty:
            raise SystemExit(f"clipped {src_name} is empty for the crop window")
        clipped.to_file(output_dir / out_name)
    return crop_dem


def run_workflow(crop_dem, output_dir):
    """Run the unmodified full FIM workflow on the crop."""
    pgf = pyGeoFlood(dem_path=crop_dem, project_dir=output_dir)
    pgf.flowline_path = output_dir / f"{CROP_STEM}_flowline.shp"
    pgf.catchment_path = output_dir / f"{CROP_STEM}_catchment.shp"
    # mirrors tests/test_functionality.py: a nonexistent custom flowline so
    # run_fim_workflow skips rasterize_custom_flowline
    pgf.custom_flowline_path = output_dir / "custom_flowline"
    pgf.config = PGF_Config(
        {"calculate_flood_stage": {"custom_Q": CUSTOM_Q}}
    )
    pgf.run_fim_workflow()


def write_manifest(output_dir):
    """Hash every golden artifact class into manifest.sha256."""
    lines = []
    total_bytes = 0
    for artifact_class in sorted(ARTIFACT_CLASSES):
        filename = ARTIFACT_CLASSES[artifact_class]
        path = output_dir / filename
        if not path.is_file():
            raise SystemExit(
                f"missing golden artifact {artifact_class!r}: {path}"
            )
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{sha256_file(path)}  {artifact_class}  {filename}\n")
    manifest_path = output_dir / "manifest.sha256"
    manifest_path.write_text("".join(lines))
    return total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference fixtures (spec D2)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving cropped inputs, workflow outputs, and the "
        "sha256 manifest",
    )
    parser.add_argument(
        "--check-manifest",
        type=Path,
        default=None,
        help="fail unless the generated manifest equals this pinned "
        "manifest byte-for-byte",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    check_environment(output_dir)
    print("Preparing cropped inputs ...")
    crop_dem = prepare_inputs(output_dir)
    print(f"Running unmodified pygeoflood run_fim_workflow on {crop_dem} ...")
    run_workflow(crop_dem, output_dir)
    total_bytes = write_manifest(output_dir)

    print(
        f"Golden artifacts: {len(ARTIFACT_CLASSES)} files, "
        f"{total_bytes / 1e6:.1f} MB total"
    )
    manifest_path = output_dir / "manifest.sha256"
    print(f"Manifest written to {manifest_path}")

    if args.check_manifest is not None:
        pinned = Path(args.check_manifest).read_bytes()
        generated = manifest_path.read_bytes()
        if pinned != generated:
            pinned_lines = pinned.decode().splitlines()
            generated_lines = generated.decode().splitlines()
            print(
                "MANIFEST MISMATCH: regenerated outputs are not byte-identical "
                "to the pinned manifest.",
                file=sys.stderr,
            )
            for line in pinned_lines:
                if line not in generated_lines:
                    print(f"  pinned-only: {line}", file=sys.stderr)
            for line in generated_lines:
                if line not in pinned_lines:
                    print(f"  generated-only: {line}", file=sys.stderr)
            sys.exit(1)
        print("Manifest matches the pinned reference byte-identically.")


if __name__ == "__main__":
    main()
