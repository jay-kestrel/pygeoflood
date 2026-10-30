# GeoFlood

Flood mapping program based on high-resolution terrain analyses.

## Documentation and examples

https://passah2o.github.io/pygeoflood/

## Installation

We recommend installing `pygeoflood` into a conda environment. Python 3.10, 3.11, or 3.12 is required:

```bash
$ pip install pygeoflood
```

For the bleeding-edge version, install from GitHub:

```bash
$ pip install git+https://github.com/passah2o/pygeoflood
```

If you plan on running the Fill-Spill-Merge function to estimate pluvial flooding, you must also install a fork of the richdem package using the following commands:

```bash
$ git clone https://github.com/mdp0023/richdem.git richdem
$ cd richdem/wrappers/pyrichdem
$ pip install "setuptools<81"
$ pip install "numpy==2.3.5"
$ pip install --no-deps .
```

For local development of this fork, install in editable mode. Source and
editable installs require a C linker and Rust 1.85.1 or newer (`cargo`,
`rustc`); CI uses 1.85.1. Maturin is installed automatically by pip as the
PEP 517 build backend — do not run setuptools separately. Built wheels use
CPython's abi3 (3.10+) interface; the package still supports Python 3.10–3.12.

```bash
$ git clone https://github.com/jay-kestrel/pygeoflood
$ cd pygeoflood
$ pip install -e .
```

The private `pygeoflood._rust` extension currently accelerates only
`tools.anisodiff`; all other kernels and geospatial dependencies remain in
Python. The public filter signature, float32 output, and directional NaN
masking are unchanged. Rayon uses the available CPU pool; set
`RAYON_NUM_THREADS` before the first native call to limit it.

Native development checks (run from the repository root):

```bash
$ cargo fmt --manifest-path rust/Cargo.toml --check
$ PYO3_PYTHON=python cargo clippy --manifest-path rust/Cargo.toml --locked --all-targets -- -D warnings
$ PYO3_PYTHON=python cargo test --manifest-path rust/Cargo.toml --locked
$ pytest tests/test_anisodiff.py
```

For Python installations whose shared library is not on the dynamic linker
search path, set `LD_LIBRARY_PATH` to `python -c 'import sysconfig;
print(sysconfig.get_config_var("LIBDIR"))'` when running Cargo unit tests on
Linux. Re-run `pip install -e .` after editing Rust; editable imports do not
recompile it automatically.

The native CI workflow tests editable installs, abi3 wheels, and sdist
rebuilds on Python 3.10, 3.11, and 3.12. Golden-value verification remains
`WBT_PATH=/path/to/WBT pytest tests/test_golden_equivalence.py` with the
cached WhiteboxTools v2.4.0 binary and `tests/golden/requirements-pins.txt`.

To run the test suite:

```bash
$ pip install pytest
$ pytest tests/test_functionality.py -v
```

## Citing

When using GeoFlood, please cite the following paper:

Zheng, X., D. Maidment, D. Tarboton, Y. Liu, P. Passalacqua (2018), GeoFlood: Large scale flood inundation mapping based on high resolution terrain analysis, Water Resources Research, 54, 12, 10013-10033, doi:10.1029/2018WR023457.

When using GeoNet, please cite the following papers:

Passalacqua, P., T. Do Trung, E. Foufoula-Georgiou, G. Sapiro, W. E. Dietrich (2010), A geometric framework for channel network extraction from lidar: Nonlinear diffusion and geodesic paths, Journal of Geophysical Research Earth Surface, 115, F01002, doi:10.1029/2009JF001254.

Sangireddy, H., R. A. Carothers, C.P. Stark, P. Passalacqua (2016), Controls of climate, topography, vegetation, and lithology on drainage density extracted from high resolution topography data, Journal of Hydrology, 537, 271-282, doi:10.1016/j.jhydrol.2016.02.051.
