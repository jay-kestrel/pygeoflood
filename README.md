# pygeoflood

Flood mapping from high-resolution terrain analysis — the maintained
successor of GeoFlood/GeoNet. Implements the Height Above Nearest Drainage
(HAND) fluvial flood-inundation method, channel extraction, and coastal
(`c_hand`) and pluvial (`fill_spill_merge`) compound flooding modes.

## About this fork

This is the Kestrel fork of
[passaH2O/pygeoflood](https://github.com/passah2O/pygeoflood) — ahead of
upstream, with no upstream changes unmerged.

> ⚠️ **`pip install pygeoflood` installs upstream's PyPI release, not this
> fork.** To get this fork, install from GitHub (below).

What the fork adds on top of upstream (see [CHANGELOG.md](CHANGELOG.md)):

- **Native Rust kernels** — `anisodiff` and the `process_cells` SRC
  accumulation, compiled once for Python 3.10–3.12 (abi3) and parallelized
  with Rayon.
- **A `pygeoflood` CLI** — read-only `version`, `list-options`, and
  `check-inputs` commands.
- **A benchmark harness** with an official recorded baseline
  (`benchmarks/README.md`).
- **A golden-value equivalence suite** pinning every kernel to deterministic
  reference outputs (`tests/golden/README.md`).
- **Fail-fast WhiteboxTools calls** — a panicked WBT stage can no longer let
  downstream stages read stale intermediates.
- **Documentation** — as-built architecture ([ARCHITECTURE.md](ARCHITECTURE.md)),
  contributor guide ([CONTRIBUTING.md](CONTRIBUTING.md)),
  [CHANGELOG.md](CHANGELOG.md), and a repaired Sphinx reference (`doc/`).

## Installation

We recommend installing pygeoflood into a conda environment. Python 3.10,
3.11, or 3.12 is required (3.13 is not supported).

```bash
# upstream's PyPI release — NOT this fork
pip install pygeoflood

# this fork, from source (requires a C linker and Rust 1.85.1+;
# CI uses 1.85.1)
pip install git+https://github.com/jay-kestrel/pygeoflood
```

Maturin (the PEP 517 build backend) is installed automatically and builds
the `pygeoflood._rust` extension; do not run setuptools separately. Built
wheels use CPython's abi3 (3.10+) interface.

If you plan to run `fill_spill_merge` (pluvial flooding), you also need a
fork of the richdem package:

```bash
git clone https://github.com/mdp0023/richdem.git richdem
cd richdem/wrappers/pyrichdem
pip install "setuptools<81"
pip install "numpy==2.3.5"
pip install --no-deps .
```

For local development of this fork, install in editable mode:

```bash
git clone https://github.com/jay-kestrel/pygeoflood
cd pygeoflood
pip install -e .
```

Editable installs do not recompile the Rust extension — re-run
`pip install -e .` after editing `rust/`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full development environment and
test matrix.

## Quickstart

```python
from pygeoflood import pyGeoFlood, PGF_Config

# full fluvial FIM workflow
pgf = pyGeoFlood(dem_path="DEM.tif", project_dir="output")
pgf.run_fim_workflow()

# per-method options via PGF_Config
pgf = pyGeoFlood(
    dem_path="DEM.tif",
    config=PGF_Config({"calculate_flood_stage": {"custom_Q": 500}}),
)
pgf.run_fim_workflow()

# channel extraction only
pgf = pyGeoFlood(dem_path="DEM.tif")
pgf.run_ce_workflow()
```

Every stage prints its resolved parameters and writes deterministic outputs
under `project_dir`, named `{dem-stem}_{product}.{ext}` — the how and why
are in [ARCHITECTURE.md](ARCHITECTURE.md).

The `pygeoflood` CLI (equivalently `python -m pygeoflood`) provides
read-only helpers:

```bash
pygeoflood version        # package, Python, dependency, WBT versions
pygeoflood list-options   # every method/option PGF_Config recognizes
pygeoflood check-inputs DEM.tif --flowline flowline.shp --catchment catchment.shp
```

## Documentation

| Document | Contents |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | As-built architecture: orchestration → numerics → Rust kernels → WhiteboxTools, workflow data flow, `PGF_Config`, parity enforcement. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev environment (Python/Rust/WhiteboxTools), the test matrix, conventions. |
| [CHANGELOG.md](CHANGELOG.md) | Fork history atop upstream v0.0.3a8. |
| `doc/` | Sphinx docs — Getting Started, Background, User Guide, API reference. Build with `cd doc && make html`. |
| [benchmarks/README.md](benchmarks/README.md) | The benchmark harness protocol and the official baseline. |
| [tests/golden/README.md](tests/golden/README.md) | The golden-corpus regeneration and gating protocol. |

The upstream documentation site
([passah2o.github.io/pygeoflood](https://passah2o.github.io/pygeoflood/)) is
built from the upstream repository; this fork's docs render locally from
`doc/`.

## Testing

Quick note before you run anything: bare **`pytest` collects only
`benchmarks/tests/`** (the harness schema tests) due to the `testpaths`
setting — the package suites are opt-in by path. The full matrix, including
the golden-equivalence and determinism suites and their WhiteboxTools
requirements, is in [CONTRIBUTING.md](CONTRIBUTING.md#the-test-matrix).

## Native development checks

Run from the repository root:

```bash
cargo fmt --manifest-path rust/Cargo.toml --check
PYO3_PYTHON=python cargo clippy --manifest-path rust/Cargo.toml --locked --all-targets -- -D warnings
PYO3_PYTHON=python cargo test --manifest-path rust/Cargo.toml --locked
pytest tests/test_anisodiff.py
```

If Cargo cannot find the Python shared library on Linux, set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
```

Golden-value verification remains:

```bash
WBT_PATH=/path/to/WBT pytest tests/test_golden_equivalence.py
```

with the cached WhiteboxTools v2.4.0 binary and the pins from
`tests/golden/requirements-pins.txt`.

## Julia kernel-track development checks

`julia/` is an experimental, additive subproject (a sibling of `rust/`) that
ports the first-party numeric kernels to Julia — currently `process_cells`,
its callees `binary_search`/`jit_inun`, and `jit_binary_hand` — and verifies
them against the same golden corpus the Python equivalence suite gates on
(SRC CSV at rtol 1e-6, inundation mask exact + depths rtol 1e-6, binary HAND
exact). It reads the corpus's rasters directly as `.npy` fixtures (no
WhiteboxTools, no GDAL, no Python in the loop) and is invoked as a
subprocess — the same integration shape WhiteboxTools itself uses.

Install Julia 1.10.9 (pinned in CI) with
[juliaup](https://github.com/JuliaLang/juliaup) or the tarball:

```bash
curl -sL https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.9-linux-x86_64.tar.gz -o /tmp/julia.tar.gz
echo "5a2d2c5224594b683c97e7304cb72407fbcf0be4a0187789cba1a2f73f0cbf09  /tmp/julia.tar.gz" | sha256sum -c -
mkdir -p /opt/julia && tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1
export PATH=/opt/julia/bin:$PATH
```

Run from the repository root (fixtures are generated, not committed —
they are ~136 MB of corpus arrays):

```bash
python julia/tools/dump_kernel_fixtures.py \
  --corpus-dir tests/golden/data/run1 --out julia/test/fixtures
julia --project=julia -e 'using Pkg; Pkg.test()'
julia --project=julia julia/test/equivalence_runner.jl julia/test/fixtures
```

The equivalence runner re-derives the kernels' outputs from the corpus and
applies the gates plus a discrimination check (a perturbed input must fail).
CI runs the same steps in `.github/workflows/julia.yml`.

## Citing

When using pygeoflood, please cite the following papers:

Zheng, X., D. Maidment, D. Tarboton, Y. Liu, P. Passalacqua (2018), GeoFlood: Large scale flood inundation mapping based on high resolution terrain analysis, Water Resources Research, 54, 12, 10013-10033, doi:10.1029/2018WR023457.

Passalacqua, P., T. Do Trung, E. Foufoula-Georgiou, G. Sapiro, W. E. Dietrich (2010), A geometric framework for channel network extraction from lidar: Nonlinear diffusion and geodesic paths, Journal of Geophysical Research Earth Surface, 115, F01002, doi:10.1029/2009JF001254.

Sangireddy, H., R. A. Carothers, C.P. Stark, P. Passalacqua (2016), Controls of climate, topography, vegetation, and lithology on drainage density extracted from high resolution topography data, Journal of Hydrology, 537, 271-282, doi:10.1016/j.jhydrol.2016.02.051.

## License

This project is licensed under GPL-3.0 — see [LICENSE](LICENSE).
