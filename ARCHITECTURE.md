# Architecture

As-built architecture of **pygeoflood** on the `main` branch (at commit
`37b957c`, October 2026). This document describes what the code does today,
not what is planned. For usage see the [README](README.md); for contributor
setup and the test matrix see [CONTRIBUTING.md](CONTRIBUTING.md).

## The one-paragraph version

`pyGeoFlood` is a stage-based orchestrator: every analysis step is a method
that reads files written by earlier stages and writes deterministic paths for
the next one. Numerical kernels live in `src/pygeoflood/tools.py`
(NumPy/SciPy/scikit-image/numba/scikit-fmm), with the two hottest kernels
implemented in Rust (`pygeoflood._rust`, built by maturin/PyO3, parallelized
with Rayon). Gridded hydrology — depression filling, flow accumulation and
direction, basin delineation — is delegated to the external WhiteboxTools
binary through the `whitebox` Python wrapper. Per-method options flow through
a `PGF_Config` object with a fixed precedence: explicit call arguments beat
config values, config values beat signature defaults.

## Layer map

```text
┌─────────────────────────────────────────────────────────────────┐
│  pyGeoFlood + PGF_Config          src/pygeoflood/pygeoflood.py  │
│  stage orchestration, deterministic path attributes,            │
│  run_ce_workflow / run_fim_workflow                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │ one method per pipeline stage
┌───────────────────────────────▼─────────────────────────────────┐
│  tools.py                       src/pygeoflood/tools.py         │
│  numerics + I/O (rasterio, scipy, skimage, numba, skfmm),       │
│  decorators (time_it, use_config_defaults), raster/vector I/O,  │
│  WhiteboxTools bootstrap (get_WhiteboxTools, WBT_PATH)          │
└───────────────────┬─────────────────────────────┬───────────────┘
     hot kernels    │                             │  hydrology subprocess
┌───────────────────▼───────────────────┐ ┌───────▼────────────────────────┐
│  pygeoflood._rust        rust/src/    │ │  WhiteboxTools binary          │
│  anisodiff, process_cells             │ │  via the `whitebox` PyPI       │
│  PyO3 0.29, abi3-py310, Rayon         │ │  wrapper; v2.4.0 pinned in CI  │
└───────────────────────────────────────┘ └────────────────────────────────┘
```

## Module map

| Path | Lines | Role |
| --- | --- | --- |
| `src/pygeoflood/pygeoflood.py` | 2,819 | `pyGeoFlood` class (~30 public stage methods + the `run_ce_workflow` / `run_fim_workflow` orchestrators) and `PGF_Config`, the per-method options container. |
| `src/pygeoflood/tools.py` | 1,843 | Decorators (`time_it`, `use_config_defaults`), raster/vector I/O (`read_raster`, `write_raster`, `write_vector_points/lines`, `rasterize_flowline`), WhiteboxTools bootstrap (`get_WhiteboxTools`), and the numerics: `anisodiff` and `process_cells` (both delegate to `_rust`), `simple_gaussian_smoothing`, `compute_dem_slope/curvature`, `get_skeleton`, `fast_marching` (scikit-fmm), `get_channel_heads`, `jit_binary_hand`, `get_channel_network`, `get_river_attributes`, `catchhydrogeo`, `get_flood_stage`, `get_inun`, `get_c_hand`. |
| `src/pygeoflood/cli.py` | 401 | The `pygeoflood` console script (read-only commands: `version`, `list-options`, `check-inputs`). |
| `src/pygeoflood/wrapper.py` | 80 | `run_wbt`, the fail-fast choke point for every WhiteboxTools call. |
| `src/pygeoflood/__main__.py` | 11 | `python -m pygeoflood` — delegates to `cli.main`. |
| `src/pygeoflood/__init__.py` | 1 | Exports exactly two names: `pyGeoFlood`, `PGF_Config`. `pygeoflood._rust` is private. |
| `rust/src/lib.rs` | 144 | The two `#[pyfunction]`s (`anisodiff`, `process_cells`) and their GIL/Rayon handoff. |
| `rust/src/diffusion.rs` | 197 | PeronaMalik1 / PeronaMalik2 / Linear flux kernels with f32/f64 precision toggles. |
| `rust/src/cells.rs` | 188 | Deterministic chunked SRC accumulation (fixed row chunks, ordered fold). |

## The `pyGeoFlood` object

The constructor takes three arguments:

- `dem_path` — path to the DEM GeoTIFF (required);
- `project_dir` — where every output is written (defaults to the directory
  containing the DEM);
- `config` — a `PGF_Config` instance, a plain dict (converted), or `None`.

Everything else is derived. `__init__` builds ~35 path attributes —
`filtered_dem_path`, `slope_path`, `hand_path`, `fim_path`, … — all of the
form `{project_dir}/{dem stem}_{product}.{ext}` (backed by the
`tools.path_property` property factory). Two consequences follow:

1. **Stages compose through the filesystem.** Each stage checks for and
   writes its deterministic file, so a workflow can be resumed or re-run in
   place — and a stage that fails silently would let the next stage read the
   *previous run's* stale intermediate. That hazard is why WhiteboxTools
   calls are fail-fast (see [The WhiteboxTools boundary](#the-whiteboxtools-boundary)).
2. **Instances round-trip.** `to_paths(file_path)` writes
   `attribute="value"` lines for existing files; `from_paths(file_path)`
   reconstructs an instance from them (the `config` attribute must be set
   separately).

Public stage methods are decorated with `@time_it` (wall-clock logging) and
`@use_config_defaults` (config resolution — see
[`PGF_Config`](#pgf_config)). The two workflow orchestrators carry only
`@time_it` because they take no options.

## Workflow data flow

`run_ce_workflow()` runs the channel-extraction pipeline:

```text
apply_nonlinear_filter → calculate_slope → calculate_curvature
→ fill_depressions → calculate_mfd_flow_accumulation
→ calculate_d8_flow_direction → find_outlets → delineate_basins
→ define_skeleton → find_endpoints → calculate_binary_hand
→ [rasterize_custom_flowline, only if custom_flowline_path exists]
→ extract_channel_network
```

`run_fim_workflow()` runs the full fluvial flood-mapping pipeline: the same
front half, then

```text
→ calculate_hand → segment_channel_network → delineate_segment_catchments
→ calculate_src → calculate_flood_stage → inundate
```

Two notes on this shape:

- `calculate_geodesic_distance` and `identify_channel_heads` are **commented
  out of `run_fim_workflow` by upstream design** (upstream issue #12); the
  benchmark harness reports them as `skipped`, never silently absent.
- `rasterize_custom_flowline` is conditional: it runs only when a custom
  flowline shapefile exists at `custom_flowline_path`.

Beyond the two orchestrators, standalone entry points exist for the compound
modes: `fill_spill_merge` (pluvial; lazily imports the
[mdp0023/richdem fork](https://github.com/mdp0023/richdem) and raises a
directed `ImportError` with install instructions when it is missing) and
`c_hand` (coastal HAND).

## `PGF_Config`

`PGF_Config` is the single options container. Its behavior:

- At module import, `pygeoflood.py` introspects every public `pyGeoFlood`
  method signature (`pgf_methods` / `pgf_params`) to build the set of valid
  `(method, option)` pairs. `pygeoflood list-options` prints exactly this
  table.
- `update_options(dict, **kwargs)` accepts either
  `{method: {option: value, ...}, ...}` or keyword form
  `method={option: value, ...}`. Unrecognized methods or options print a
  warning and are **ignored** — never a crash.
- `pyGeoFlood(..., config=...)` accepts a `PGF_Config` or a dict; the
  `config` setter converts dicts.

The `@use_config_defaults` decorator resolves each stage's parameters with a
fixed precedence, and prints the resolved parameter set before every run:

```text
1. explicitly passed arguments        (highest)
2. values from self.config
3. signature defaults                 (lowest)
```

Example — force `custom_Q=500` for the flood-stage step wherever the
workflow runs it:

```python
from pygeoflood import pyGeoFlood, PGF_Config

pgf = pyGeoFlood(
    dem_path="OC1mTest.tif",
    config=PGF_Config({"calculate_flood_stage": {"custom_Q": 500}}),
)
pgf.run_fim_workflow()
```

## The native layer (`rust/`)

Two kernels are implemented in Rust and exposed through a single compiled
module, `pygeoflood._rust`:

- **`anisodiff`** — Perona–Malik (1 and 2) and linear nonlinear diffusion,
  the first stage of both workflows and the most iterated kernel. `rust/
  src/diffusion.rs` replicates NumPy's scalar-promotion semantics
  deliberately: weak Python scalars keep float32 arithmetic, and explicit
  float64 scalars promote only at the operation where they enter — the
  `precision` tuple toggles each threshold/step/gamma to f64 independently.
- **`process_cells`** — the SRC (stage–rating-curve) accumulation from
  `calculate_src`: for each (height × catchment) pair, accumulate cell
  counts, surface area, bed area, and volume. `rust/src/cells.rs` keeps the
  result **bitwise deterministic across thread counts**: work is partitioned
  into fixed row chunks (at most `MAX_CHUNKS = 32`) and the per-chunk
  totals are folded in chunk order, instead of atomically merged in
  completion order. Inputs are validated (matching raster shapes, finite and
  sorted hydroids, overflow-checked accumulator size), and the Python side
  (`tools.process_cells`) preserves the Numba kernel's f32 rounding points
  before promoting to f64 for the call.

Both functions follow the same concurrency pattern in `lib.rs`:

1. Borrow the NumPy arrays as read-only views (zero-copy).
2. **Snapshot into owned buffers while holding the GIL** — Python aliases
   could mutate a "read-only" array once the GIL is released.
3. Run the kernel with `py.detach` / `py.allow_threads` so Rayon can use the
   full core pool off-GIL (`RAYON_NUM_THREADS` limits it).
4. Rebuild owned output arrays and return them; NaN input to `anisodiff`
   raises `PyValueError`.

Build and packaging: maturin is the PEP 517 backend (`[tool.maturin]` sets
`manifest-path = rust/Cargo.toml`, `python-source = "src"`,
`module-name = "pygeoflood._rust"`). Wheels use the abi3-py310 stable ABI —
one build serves Python 3.10–3.12. `tools.py` imports `_rust` at module
scope (line 14), so **`import pygeoflood` requires the compiled extension**;
editable installs must be rebuilt (`pip install -e .`) after Rust edits.

Parity is enforced, not assumed — see the next section.

## Parity enforcement

Three independent guards keep the ports honest:

1. **Golden-value equivalence** (`tests/test_golden_equivalence.py`,
   `tests/golden/`): the deterministic 2000×2000 crop of the OC1m DEM is run
   through `run_fim_workflow`; the corpus (~25 MB/run) is regenerated, not
   committed. The suite regenerates it **twice** and requires byte-identical
   manifests, checks both against the pinned `tests/golden/
   golden_manifest.sha256`, then applies per-class tolerance gates — HAND
   raster and CSVs at `rtol 1e-6`, inundation **mask positions exact** with
   depths at `rtol 1e-6`, WBT integer rasters / binary HAND / channel-network
   raster — and finally a discrimination check: perturbations of `1e-3` must
   fail the gates.
2. **Rust-vs-reference kernels** (`tests/test_anisodiff.py` +
   `tests/reference_anisodiff.py`): the native `anisodiff` against the NumPy
   reference implementation.
3. **Determinism regression** (`tests/test_process_cells_determinism.py`):
   `process_cells` results must be bitwise identical across thread counts.
   The test reads the ambient thread count on purpose — pinning
   `NUMBA_NUM_THREADS=1` here would reduce the check to a serial no-op.

## The WhiteboxTools boundary

Hydrological stages are **subprocess calls to the WhiteboxTools CLI** (six
native calls in the canonical workload): depression filling, MFD flow
accumulation, D8 flow direction, outlets/basin delineation, HAND-adjacent
raster products, and catchment delineation. The `whitebox` PyPI package is
just a wrapper around the binary; `tools.get_WhiteboxTools` bootstraps it and
honors `WBT_PATH`.

Reproducibility discipline (mirrored by CI):

- **Version pinning**: equivalence and functionality CI download and verify
  WhiteboxTools **v2.4.0** (cache key `wbt-v2.4.0-linux-x86_64`, version
  `grep` check after download).
- **Thread pinning**: CI sets `NUMBA_NUM_THREADS=1` and `OMP_NUM_THREADS=1`
  for the value-sensitive suites.
- **Fail-fast calls**: the `whitebox 2.3.6` wrapper's `run_tool` returns 0
  even when the tool panics (panics surface only in stdout). Every
  `wbt.*` call site routes through `wrapper.run_wbt`, which raises
  `RuntimeError` on a nonzero exit code or a panic marker (`error`,
  `panic`, `Problem`, `Exception`). Without this, a panicked stage would
  leave the previous run's cached intermediates in place and downstream
  stages would silently consume them.

## Benchmark harness

`python -m benchmarks` (see `benchmarks/README.md`) times the canonical
workload — `run_fim_workflow()` on the OC1mTest DEM with `custom_Q=500` —
under two profiles: `crop` (fixed 2000×2000 window; fast iteration and CI
smoke) and `full` (the whole 4750×3378 tile; headline numbers). Each
invocation runs one cold pass (fresh process, numba JIT cold) plus warm
passes, and reports per-stage, per-WhiteboxTools-subprocess, per-kernel, and
I/O timings. Every run writes a `manifest.json` (CPU/RAM/platform, thread
counts, WhiteboxTools package and binary versions, full dependency versions,
git commit, DEM sha256, grid metadata) so every performance claim carries
its environment. `reproduce_baseline.sh` reproduces the official baseline;
the pinned-above claim chain starts at `benchmarks/results/`.

The harness also ships the only tests bare `pytest` collects —
`benchmarks/tests/` — schema tests for the manifests plus a drift guard
(`test_cli.py`) that keeps `cli.py`'s dependency list identical to
`benchmarks/manifest.py`'s. See the
[`testpaths` quirk](CONTRIBUTING.md#the-bare-pytest-testpaths-quirk) in
CONTRIBUTING.md before running the package suites.

## CLI

`pygeoflood` (console script) and `python -m pygeoflood` are equivalent.
All commands are read-only:

| Command | What it does |
| --- | --- |
| `pygeoflood version` | pygeoflood, Python, platform, dependency versions (same package set the benchmark manifests record), and WhiteboxTools package/binary status honoring `WBT_PATH`. |
| `pygeoflood list-options` | Every `(method, option)` pair `PGF_Config` recognizes, derived from the live signatures. |
| `pygeoflood check-inputs DEM.tif [--flowline F.shp] [--catchment C.shp]` | Read-only pre-flight validation for the FIM workflow: DEM readability, flowline endpoints inside the DEM grid, catchment vector sanity. Never writes. |

Heavy imports are deferred into command bodies, but the console script still
pays the `pygeoflood` package import (and therefore the compiled `_rust`
extension) — see the note in `src/pygeoflood/cli.py`.
