# pygeoflood benchmark — full tile (commit 3650c9b) — numba cache=True variant

Schema version: 1 · profile: `full` · DEM sha256: `d2dad4d4a1a7…`

## End-to-end

| mode | iteration | wall clock (s) |
|---|---|---|
| cold | 1 | 85.74 |
| cold | 2 | 83.02 |
| cold | 3 | 82.47 |

## Run-to-run variance

Fewer than two warm runs — no variance computed for this invocation.

## Per-stage timings (seconds)

Bypassed in upstream `run_fim_workflow()` at 3650c9b (geodesic path, upstream issue #12): calculate_geodesic_distance, identify_channel_heads. `rasterize_custom_flowline` is conditional upstream and does not run in the canonical workload (no custom flowline input).

| stage | cold | warm (median) | status |
|---|---|---|---|
| apply_nonlinear_filter | 22.58 | — | timed |
| calculate_slope | 1.63 | — | timed |
| calculate_curvature | 2.06 | — | timed |
| fill_depressions | 11.87 | — | timed |
| calculate_mfd_flow_accumulation | 7.87 | — | timed |
| calculate_d8_flow_direction | 3.87 | — | timed |
| find_outlets | 0.29 | — | timed |
| delineate_basins | 1.65 | — | timed |
| define_skeleton | 1.25 | — | timed |
| find_endpoints | 0.06 | — | timed |
| calculate_binary_hand | 1.21 | — | timed |
| extract_channel_network | 3.08 | — | timed |
| calculate_hand | 8.11 | — | timed |
| segment_channel_network | 0.98 | — | timed |
| delineate_segment_catchments | 2.28 | — | timed |
| calculate_src | 12.67 | — | timed |
| calculate_flood_stage | 0.01 | — | timed |
| inundate | 0.85 | — | timed |

## WhiteboxTools calls (C++ subprocesses)

| tool | host stage | calls/run | cold (s) | warm (s) |
|---|---|---|---|---|
| fill_depressions | fill_depressions | 1 | 11.86 | — |
| quinn_flow_accumulation | calculate_mfd_flow_accumulation | 1 | 7.86 | — |
| d8_pointer | calculate_d8_flow_direction | 1 | 3.25 | — |
| basins | delineate_basins | 1 | 1.64 | — |
| elevation_above_stream | calculate_hand | 1 | 8.10 | — |
| watershed | delineate_segment_catchments | 1 | 2.06 | — |

## Python/numba kernels

Cold timings include the one-off numba JIT compile; the cold-vs-warm delta is the JIT tax that an AOT port removes.

| kernel | host stage | cold (s) | warm (s) |
|---|---|---|---|
| anisodiff | apply_nonlinear_filter | 21.19 | — |
| jit_binary_hand | calculate_binary_hand | 0.84 | — |
| process_cells | calculate_src | 11.73 | — |
| jit_inun | inundate | 0.03 | — |

## Raster/CSV/vector I/O

| kind | warm calls/run | cold (s) | warm (s) |
|---|---|---|---|
| raster_read | 0 | 3.12 | — |
| raster_write | 0 | 4.20 | — |
| csv_write | 0 | 0.02 | — |
| vector_write | 0 | 0.02 | — |

## Environment manifest

```json
{
  "cpu_count": 8,
  "cpu_model": "Intel(R) Xeon(R) Processor @ 2.60GHz",
  "crop": null,
  "dem": {
    "bounds": [
      567651.0,
      3332837.0,
      572401.0,
      3336215.0
    ],
    "cell_count": 16045500,
    "crs": "EPSG:6343",
    "dtype": "float32",
    "nodata": -3.4028234663852886e+38,
    "path": "/home/user/work/bench-runs/work-cache/full/OC1mTest.tif",
    "resolution": [
      1.0,
      1.0
    ],
    "shape": [
      3378,
      4750
    ]
  },
  "dem_sha256": "d2dad4d4a1a7b0843c492b4ba7ab9fd8ef4d4480727a95f43176bbbc6908cc4f",
  "generated_utc": "2026-10-30T18:14:09Z",
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
