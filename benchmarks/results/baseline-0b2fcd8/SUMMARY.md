# pygeoflood benchmark — full profile (commit 0b2fcd8f5af5bbd4acd0949dc6a89bbb7eede511)

Schema version: 1 · profile: `full` · DEM sha256: `d2dad4d4a1a7…`

## End-to-end

| mode | iteration | wall clock (s) |
|---|---|---|
| cold | 1 | 58.72 |
| warm | 1 | 50.54 |
| warm | 2 | 50.87 |

## Run-to-run variance

Warm-run spread 0.66% (mean 50.70 s, 2 runs) — within the 10% guidance.

## Per-stage timings (seconds)

Bypassed in upstream `run_fim_workflow()` at 3650c9b (geodesic path, upstream issue #12): calculate_geodesic_distance, identify_channel_heads. `rasterize_custom_flowline` is conditional upstream and does not run in the canonical workload (no custom flowline input).

| stage | cold | warm (median) | status |
|---|---|---|---|
| apply_nonlinear_filter | 3.29 | 3.23 | timed |
| calculate_slope | 1.59 | 1.63 | timed |
| calculate_curvature | 1.94 | 2.03 | timed |
| fill_depressions | 11.59 | 11.77 | timed |
| calculate_mfd_flow_accumulation | 7.74 | 7.77 | timed |
| calculate_d8_flow_direction | 3.72 | 3.70 | timed |
| find_outlets | 0.29 | 0.28 | timed |
| delineate_basins | 1.52 | 1.51 | timed |
| define_skeleton | 1.08 | 1.16 | timed |
| find_endpoints | 0.06 | 0.01 | timed |
| calculate_binary_hand | 8.13 | 1.10 | timed |
| extract_channel_network | 2.94 | 2.79 | timed |
| calculate_hand | 7.75 | 7.93 | timed |
| segment_channel_network | 0.89 | 0.91 | timed |
| delineate_segment_catchments | 2.19 | 2.20 | timed |
| calculate_src | 1.75 | 1.80 | timed |
| calculate_flood_stage | 0.01 | 0.01 | timed |
| inundate | 2.26 | 0.87 | timed |

## WhiteboxTools calls (C++ subprocesses)

| tool | host stage | calls/run | cold (s) | warm (s) |
|---|---|---|---|---|
| fill_depressions | fill_depressions | 1 | 11.59 | 11.76 |
| quinn_flow_accumulation | calculate_mfd_flow_accumulation | 1 | 7.74 | 7.77 |
| d8_pointer | calculate_d8_flow_direction | 1 | 3.14 | 3.14 |
| basins | delineate_basins | 1 | 1.51 | 1.51 |
| elevation_above_stream | calculate_hand | 1 | 7.75 | 7.92 |
| watershed | delineate_segment_catchments | 1 | 1.98 | 1.98 |

## Python/numba kernels

Cold timings include the one-off numba JIT compile; the cold-vs-warm delta is the JIT tax that an AOT port removes.

| kernel | host stage | cold (s) | warm (s) |
|---|---|---|---|
| anisodiff | apply_nonlinear_filter | 1.94 | 1.93 |
| jit_binary_hand | calculate_binary_hand | 7.75 | 0.76 |
| process_cells | calculate_src | 0.84 | 0.84 |
| jit_inun | inundate | 1.46 | 0.08 |

## Raster/CSV/vector I/O

| kind | warm calls/run | cold (s) | warm (s) |
|---|---|---|---|
| raster_read | 48 | 2.88 | 2.87 |
| raster_write | 20 | 4.24 | 4.24 |
| csv_write | 4 | 0.02 | 0.02 |
| vector_write | 2 | 0.01 | 0.01 |

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
    "path": "/home/user/work/pgf-bench-runs/full/OC1mTest.tif",
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
  "generated_utc": "2026-10-30T20:18:36Z",
  "git_commit": "0b2fcd8f5af5bbd4acd0949dc6a89bbb7eede511",
  "numba_cache": false,
  "numba_env_threads": "1",
  "numba_num_threads": 1,
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
