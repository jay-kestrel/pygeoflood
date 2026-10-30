# pygeoflood benchmark — full tile (commit 3650c9b) — numba-warm variant

Schema version: 1 · profile: `full` · DEM sha256: `d2dad4d4a1a7…`

## End-to-end

| mode | iteration | wall clock (s) |
|---|---|---|
| cold | 1 | 87.40 |
| cold | 2 | 98.24 |
| cold | 3 | 97.92 |
| warm | 1 | 77.10 |
| warm | 2 | 75.43 |

## Run-to-run variance

Warm-run spread 2.18% (mean 76.26 s, 2 runs) — within the 10% guidance.

## Per-stage timings (seconds)

Bypassed in upstream `run_fim_workflow()` at 3650c9b (geodesic path, upstream issue #12): calculate_geodesic_distance, identify_channel_heads. `rasterize_custom_flowline` is conditional upstream and does not run in the canonical workload (no custom flowline input).

| stage | cold | warm (median) | status |
|---|---|---|---|
| apply_nonlinear_filter | 22.52 | 17.92 | timed |
| calculate_slope | 1.61 | 1.57 | timed |
| calculate_curvature | 2.10 | 1.93 | timed |
| fill_depressions | 12.14 | 12.11 | timed |
| calculate_mfd_flow_accumulation | 7.90 | 10.01 | timed |
| calculate_d8_flow_direction | 4.12 | 6.55 | timed |
| find_outlets | 0.29 | 0.25 | timed |
| delineate_basins | 1.61 | 1.57 | timed |
| define_skeleton | 1.34 | 1.19 | timed |
| find_endpoints | 0.06 | 0.12 | timed |
| calculate_binary_hand | 8.28 | 0.91 | timed |
| extract_channel_network | 3.32 | 2.78 | timed |
| calculate_hand | 8.37 | 8.14 | timed |
| segment_channel_network | 0.88 | 0.91 | timed |
| delineate_segment_catchments | 2.28 | 2.21 | timed |
| calculate_src | 18.05 | 7.31 | timed |
| calculate_flood_stage | 0.01 | 0.01 | timed |
| inundate | 2.28 | 0.79 | timed |

## WhiteboxTools calls (C++ subprocesses)

| tool | host stage | calls/run | cold (s) | warm (s) |
|---|---|---|---|---|
| fill_depressions | fill_depressions | 1 | 12.13 | 12.11 |
| quinn_flow_accumulation | calculate_mfd_flow_accumulation | 1 | 7.90 | 9.99 |
| d8_pointer | calculate_d8_flow_direction | 1 | 3.48 | 5.38 |
| basins | delineate_basins | 1 | 1.60 | 1.56 |
| elevation_above_stream | calculate_hand | 1 | 8.36 | 8.12 |
| watershed | delineate_segment_catchments | 1 | 2.06 | 2.04 |

## Python/numba kernels

Cold timings include the one-off numba JIT compile; the cold-vs-warm delta is the JIT tax that an AOT port removes.

| kernel | host stage | cold (s) | warm (s) |
|---|---|---|---|
| anisodiff | apply_nonlinear_filter | 21.15 | 16.52 |
| jit_binary_hand | calculate_binary_hand | 7.88 | 0.59 |
| process_cells | calculate_src | 16.94 | 6.32 |
| jit_inun | inundate | 1.43 | 0.02 |

## Raster/CSV/vector I/O

| kind | warm calls/run | cold (s) | warm (s) |
|---|---|---|---|
| raster_read | 48 | 3.34 | 3.64 |
| raster_write | 20 | 4.21 | 4.07 |
| csv_write | 4 | 0.02 | 0.02 |
| vector_write | 2 | 0.03 | 0.04 |

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
  "generated_utc": "2026-10-30T16:57:58Z",
  "git_commit": "233ee90a582c71bd9a7d71dcf0548471eea7e940",
  "numba_env_threads": null,
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
