# pygeoflood benchmark — crop (commit 3650c9b) — numba cache=True variant

Schema version: 1 · profile: `crop` · DEM sha256: `9c9a304b664e…`
Crop window: origin (row 1200, col 300), 2000x2000 cells

## End-to-end

| mode | iteration | wall clock (s) |
|---|---|---|
| cold | 1 | 19.43 |
| cold | 2 | 18.45 |
| warm | 1 | 19.91 |
| warm | 2 | 18.10 |

## Run-to-run variance

Warm-run spread 9.52% (mean 19.00 s, 2 runs) — within the 10% guidance.

## Per-stage timings (seconds)

Bypassed in upstream `run_fim_workflow()` at 3650c9b (geodesic path, upstream issue #12): calculate_geodesic_distance, identify_channel_heads. `rasterize_custom_flowline` is conditional upstream and does not run in the canonical workload (no custom flowline input).

| stage | cold | warm (median) | status |
|---|---|---|---|
| apply_nonlinear_filter | 4.43 | 4.21 | timed |
| calculate_slope | 0.40 | 0.38 | timed |
| calculate_curvature | 0.46 | 0.46 | timed |
| fill_depressions | 3.01 | 3.00 | timed |
| calculate_mfd_flow_accumulation | 2.05 | 2.10 | timed |
| calculate_d8_flow_direction | 1.14 | 1.14 | timed |
| find_outlets | 0.08 | 0.08 | timed |
| delineate_basins | 0.55 | 0.54 | timed |
| define_skeleton | 0.32 | 0.30 | timed |
| find_endpoints | 0.02 | 0.01 | timed |
| calculate_binary_hand | 0.47 | 0.20 | timed |
| extract_channel_network | 0.65 | 0.62 | timed |
| calculate_hand | 1.79 | 1.79 | timed |
| segment_channel_network | 0.25 | 0.19 | timed |
| delineate_segment_catchments | 0.74 | 0.76 | timed |
| calculate_src | 2.35 | 3.01 | timed |
| calculate_flood_stage | 0.00 | 0.00 | timed |
| inundate | 0.22 | 0.21 | timed |

## WhiteboxTools calls (C++ subprocesses)

| tool | host stage | calls/run | cold (s) | warm (s) |
|---|---|---|---|---|
| fill_depressions | fill_depressions | 1 | 3.00 | 2.99 |
| quinn_flow_accumulation | calculate_mfd_flow_accumulation | 1 | 2.04 | 2.09 |
| d8_pointer | calculate_d8_flow_direction | 1 | 0.98 | 0.97 |
| basins | delineate_basins | 1 | 0.55 | 0.54 |
| elevation_above_stream | calculate_hand | 1 | 1.78 | 1.78 |
| watershed | delineate_segment_catchments | 1 | 0.66 | 0.69 |

## Python/numba kernels

Cold timings include the one-off numba JIT compile; the cold-vs-warm delta is the JIT tax that an AOT port removes.

| kernel | host stage | cold (s) | warm (s) |
|---|---|---|---|
| anisodiff | apply_nonlinear_filter | 4.08 | 3.91 |
| jit_binary_hand | calculate_binary_hand | 0.35 | 0.11 |
| process_cells | calculate_src | 2.07 | 2.73 |
| jit_inun | inundate | 0.01 | 0.00 |

## Raster/CSV/vector I/O

| kind | warm calls/run | cold (s) | warm (s) |
|---|---|---|---|
| raster_read | 40 | 0.77 | 0.76 |
| raster_write | 20 | 1.08 | 1.07 |
| csv_write | 4 | 0.01 | 0.01 |
| vector_write | 2 | 0.01 | 0.01 |

## Environment manifest

```json
{
  "cpu_count": 8,
  "cpu_model": "Intel(R) Xeon(R) Processor @ 2.60GHz",
  "crop": {
    "flowline_clipped_vertices": 13,
    "flowline_original_vertices": 68,
    "height": 2000,
    "origin_col": 300,
    "origin_row": 1200,
    "width": 2000
  },
  "dem": {
    "bounds": [
      567951.0,
      3333015.0,
      569951.0,
      3335015.0
    ],
    "cell_count": 4000000,
    "crs": "EPSG:6343",
    "dtype": "float32",
    "nodata": -3.4028234663852886e+38,
    "path": "/home/user/work/bench-runs/work-cache/crop/OC1mTest_crop2000.tif",
    "resolution": [
      1.0,
      1.0
    ],
    "shape": [
      2000,
      2000
    ]
  },
  "dem_sha256": "9c9a304b664e31226edd04abb4bce5a66d43a819ea53ded0987bf32417756b98",
  "generated_utc": "2026-10-30T18:18:52Z",
  "git_commit": "3650c9bb66802fd67839819aec21cbe241797cd0",
  "numba_cache": true,
  "numba_cache_kernels": {
    "anisodiff": "not-numba",
    "get_fmm_points": "cached",
    "jit_binary_hand": "cached",
    "jit_inun": "cached",
    "process_cells": "cached"
  },
  "numba_env_threads": "8",
  "numba_num_threads": 8,
  "packages": {
    "fiona": "1.10.1",
    "geopandas": "1.2.0",
    "netcdf4": "1.7.5.1.post1",
    "numba": "0.68.1",
    "numpy": "2.5.4",
    "pandas": "3.1.0",
    "psutil": "7.2.2",
    "pygeoflood": "0.0.3a8",
    "rasterio": "1.5.2",
    "scikit-fmm": "2025.6.23",
    "scikit-image": "0.26.0",
    "scipy": "1.18.1",
    "shapely": "2.2.0",
    "whitebox": "2.3.6",
    "xarray": "2026.9.0"
  },
  "platform": "Linux-6.1.158+-x86_64-with-glibc2.41",
  "python_version": "3.12.16 (main, Oct 26 2026, 14:08:45) [Clang 23.1.3 ]",
  "ram_total_bytes": 8342290432,
  "wbt_binary_version": "WhiteboxTools v2.4.0 (c) Dr. John Lindsay 2017-2023\n\nWhiteboxTools is an advanced geospatial data analysis platform developed at\nthe University of Guelph's Geomorphometry and Hydrogeomatics Research \nGroup (GHRG). See www.whiteboxgeo.com for more details.\n",
  "wbt_max_procs": 1,
  "wbt_package_version": "2.3.6"
}
```
