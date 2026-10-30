"""CLI tests for the pygeoflood QoL entry point.

These run the CLI in-process (no subprocess) and need pygeoflood itself
importable — CI installs the package (`pip install -e ".[bench]"` on
Python 3.12) before running this directory. The check-inputs tests read
the committed OC1mTest inputs; they open metadata only, so they stay
fast despite the 67 MB DEM.
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import LineString

from benchmarks import manifest
from pygeoflood import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
OC1M_INPUTS = REPO_ROOT / "tests" / "test_data" / "OC1mTest_inputs"
OC1M_DEM = OC1M_INPUTS / "OC1mTest.tif"
OC1M_FLOWLINE = OC1M_INPUTS / "Flowline.shp"
OC1M_CATCHMENT = OC1M_INPUTS / "Catchment.shp"


def test_cli_package_list_matches_benchmark_manifest():
    # `version` mirrors benchmarks/manifest.py PACKAGES by value, not by
    # import (benchmarks/ is not part of the wheel) — drift must fail loudly.
    assert cli.PACKAGES == tuple(manifest.PACKAGES)


def test_version_reports_environment(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out
    assert "pygeoflood" in out
    assert "Python" in out
    assert "WhiteboxTools" in out


def test_list_options_lists_methods_and_options(capsys):
    assert cli.main(["list-options"]) == 0
    out = capsys.readouterr().out
    # one known method and one of its recognized options
    assert "calculate_flood_stage" in out
    assert "custom_Q" in out


def test_check_inputs_happy_path_on_committed_oc1m_inputs(capsys):
    assert OC1M_DEM.is_file(), "committed OC1mTest inputs missing"
    exit_code = cli.main(
        [
            "check-inputs",
            str(OC1M_DEM),
            "--flowline",
            str(OC1M_FLOWLINE),
            "--catchment",
            str(OC1M_CATCHMENT),
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "check-inputs: OK" in out


def test_check_inputs_flags_out_of_bounds_flowline(tmp_path, capsys):
    # Endpoints must land inside the grid: an endpoint exactly on the
    # edge converts one cell outside and crashes least-cost routing
    # (tests/golden/README.md). Synthetic flowline in the DEM's own CRS,
    # far outside its bounds.
    with rasterio.open(OC1M_DEM) as src:
        bounds = src.bounds
        dem_crs = src.crs
    outside_x = bounds.right + 10 * (bounds.right - bounds.left)
    outside_y = bounds.bottom - 10 * (bounds.top - bounds.bottom)
    flowline = tmp_path / "out_of_bounds.gpkg"
    gpd.GeoDataFrame(
        geometry=[
            LineString(
                [
                    (
                        (bounds.left + bounds.right) / 2,
                        (bounds.bottom + bounds.top) / 2,
                    ),
                    (outside_x, outside_y),
                ]
            )
        ],
        crs=dem_crs,
    ).to_file(flowline, driver="GPKG")

    exit_code = cli.main(["check-inputs", str(OC1M_DEM), "--flowline", str(flowline)])
    out = capsys.readouterr().out
    assert exit_code == 1, out
    assert "outside the DEM grid" in out
    assert "FAILED" in out


def test_check_inputs_missing_dem_fails(tmp_path, capsys):
    assert cli.main(["check-inputs", str(tmp_path / "nope.tif")]) == 1
    out = capsys.readouterr().out
    assert "ERROR: DEM not found" in out
