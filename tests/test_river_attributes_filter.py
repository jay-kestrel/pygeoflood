"""Junk-hydroid SRC rows (review art_SjgX1Jzb F2, second half).

get_river_attributes builds its table from np.unique(segment_catchments) and
only filtered the nodata sentinel, so a rasterized background (HYDROID 0)
survived into the river-attributes table and — downstream, through
catchhydrogeo — into the SRC CSV as a junk row with NaN COMID. Real WBT
basin/catchment ids are positive: filter non-positive and non-finite rows
at the source.
"""

import numpy as np
import rasterio as rio
from geopandas import GeoDataFrame
from shapely.geometry import LineString, box

from pygeoflood.tools import get_river_attributes

CRS = "EPSG:32614"


def write_dem(tmp_path):
    """A flat 10x10 one-metre DEM southwest corner (500000, 3999990)."""
    dem_path = tmp_path / "dem.tif"
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "nodata": None,
        "crs": rio.crs.CRS.from_epsg(32614),
        "transform": rio.transform.Affine(
            1.0, 0.0, 500000.0, 0.0, -1.0, 4000000.0
        ),
    }
    with rio.open(dem_path, "w", **profile) as ds:
        ds.write(np.full((10, 10), 100.0, dtype=np.float32), 1)
    return dem_path


def test_river_attributes_drops_background_hydroid_rows(tmp_path):
    dem_path = write_dem(tmp_path)

    # catchments: real catchment 1, HYDROID=0 background, nodata margin
    segment_catchments = np.array(
        [[1, 1, 0], [1, 0, -32768], [0, -32768, -32768]], dtype=np.int32
    )
    profile = {
        "nodata": -32768,
        "transform": rio.transform.Affine(
            1.0, 0.0, 500000.0, 0.0, -1.0, 4000000.0
        ),
        "crs": rio.crs.CRS.from_epsg(32614),
    }

    segments = GeoDataFrame(
        {"HYDROID": [1]},
        geometry=[LineString([(500000.5, 3999999.5), (500000.9, 3999999.9)])],
        crs=CRS,
    )
    nwm_catchments = GeoDataFrame(
        {"FEATUREID": [100]},
        geometry=[box(499999, 3999998, 500002, 4000001)],
        crs=CRS,
    )

    ra = get_river_attributes(
        dem_path,
        segment_catchments,
        nwm_catchments,
        segments,
        profile,
        1e-6,  # min_slope below the flat-DEM slope so no clamping noise
    )

    assert 1 in ra["HYDROID"].values
    assert 0 not in ra["HYDROID"].values  # junk background row dropped
    assert -32768 not in ra["HYDROID"].values  # nodata row (pre-existing)
