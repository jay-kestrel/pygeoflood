# Contributing to pygeoflood

Thanks for contributing. This guide covers the development environment, the
test matrix (including two traps that surprise everyone), and the commit
conventions. For architecture see [ARCHITECTURE.md](ARCHITECTURE.md); for
usage see the [README](README.md).

## Development environment

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | >= 3.10, **< 3.13** | `pyproject.toml` `requires-python`; 3.13 is not supported. |
| Rust | **1.85.1** | CI pins exactly this toolchain; `rust-version = "1.85"` is the MSRV. |
| C linker | any | Needed to link the extension module. |
| maturin | >= 1.8, < 2 | Installed automatically — it is the PEP 517 build backend. Do not run setuptools separately. |
| WhiteboxTools | **v2.4.0** | External binary for hydrology stages; required by the equivalence and functionality suites. |

```bash
rustup toolchain install 1.85.1 --profile minimal
rustup default 1.85.1

git clone https://github.com/jay-kestrel/pygeoflood
cd pygeoflood
pip install -e ".[bench]"     # builds the Rust extension via maturin
```

Two gotchas when working on the Rust side:

- **Editable installs do not recompile Rust.** After editing `rust/`, re-run
  `pip install -e .` (or `maturin develop`) or your changes will not be
  visible to `import pygeoflood`.
- **Cargo unit tests may need `LD_LIBRARY_PATH`.** On Linux, if
  `PYO3_PYTHON=python cargo test --manifest-path rust/Cargo.toml --locked`
  cannot find the Python shared library, set it:

  ```bash
  export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
  ```

### WhiteboxTools setup

The value-sensitive test suites need the **WhiteboxTools v2.4.0** binary.
Point `WBT_PATH` at the extracted directory's `WBT` folder:

```bash
curl -sL https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip -o /tmp/wbt.zip
unzip -q /tmp/wbt.zip -d /opt/wbt
export WBT_PATH=/opt/wbt/WhiteboxTools_linux_amd64/WBT
"$WBT_PATH/whitebox_tools" --version        # must report v2.4.0
```

For byte-identical golden regeneration, also install the exact pins from
`tests/golden/requirements-pins.txt` (the equivalence workflow does:
`pip install -c tests/golden/requirements-pins.txt -e .`).

### richdem fork (only for `fill_spill_merge`)

Pluvial flood mapping needs the
[mdp0023/richdem fork](https://github.com/mdp0023/richdem), imported lazily
by `fill_spill_merge` (missing → a directed `ImportError` with these
instructions). The fork still imports `pkg_resources`, which setuptools 81+
dropped, so install an older setuptools first:

```bash
pip install "setuptools<81"
git clone https://github.com/mdp0023/richdem.git richdem
cd richdem/wrappers/pyrichdem
pip install --no-deps .
```

## The test matrix

⚠️ **Read this before running anything.** `[tool.pytest.ini_options]` sets
`testpaths = ["benchmarks/tests"]`:

> ### The bare-`pytest` `testpaths` quirk
> Running bare **`pytest` collects only `benchmarks/tests/`** — the harness
> schema tests and the CLI drift guard. It does **not** run the package test
> suites. `pytest` reporting "all passed" on a fresh checkout means the
> schema tests passed, nothing more. Always name the suite you intend to run.

| Suite | Command | What it covers | Needs |
| --- | --- | --- | --- |
| Harness schema + CLI drift | `pytest` (i.e. `benchmarks/tests/`) | Benchmark profile/schema validation; `cli.py` package list matches `benchmarks/manifest.py`. | Nothing heavy. |
| Rust anisodiff parity | `pytest tests/test_anisodiff.py` | `_rust.anisodiff` vs the NumPy reference (`tests/reference_anisodiff.py`). | Built extension. |
| `process_cells` determinism | `pytest tests/test_process_cells_determinism.py` | Bitwise-identical results across thread counts. **Run at ambient thread count — do not pin `NUMBA_NUM_THREADS=1`**, which reduces the check to a serial no-op. | Built extension. |
| Golden equivalence | `pytest tests/test_golden_equivalence.py` | Regenerates the golden corpus twice (byte-identical manifests), checks against the pinned sha256 manifest, applies per-class tolerance gates, runs the 1e-3 discrimination check. | `WBT_PATH` (v2.4.0), pins from `tests/golden/requirements-pins.txt`, single-thread env (`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`). |
| Functionality smoke | `pytest tests/test_functionality.py` | The 97-test structural suite on the committed OC1mTest fixtures. ⚠️ **Mutates `tests/test_data/OC1mTest/` at import time** (copies inputs). Three tests need the richdem fork. | `WBT_PATH` (v2.4.0), single-thread env; richdem fork for 3/97. |
| Rust checks | `cargo fmt --manifest-path rust/Cargo.toml --check` · `PYO3_PYTHON=python cargo clippy --manifest-path rust/Cargo.toml --locked --all-targets -- -D warnings` · `PYO3_PYTHON=python cargo test --manifest-path rust/Cargo.toml --locked` | Formatting, clippy clean, Rust unit tests. | Rust 1.85.1 (+ the `LD_LIBRARY_PATH` note above). |
| Docs build | `cd doc && make html` | Sphinx HTML, including the autodoc API reference. | An **installed** pygeoflood (autodoc imports the package, including `_rust`); `sphinx`, `nbsphinx`, `furo`, `IPython` (see `doc/pgfdocs-env.yml`). |

Quick reference for the common cases:

```bash
# fast loop while editing Python
pytest tests/test_anisodiff.py -v

# before committing Rust changes
cargo fmt --manifest-path rust/Cargo.toml --check
PYO3_PYTHON=python cargo clippy --manifest-path rust/Cargo.toml --locked --all-targets -- -D warnings
PYO3_PYTHON=python cargo test --manifest-path rust/Cargo.toml --locked
pip install -e .        # rebuild the extension

# golden verification (the value-level gate)
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 WBT_PATH=/opt/wbt/WhiteboxTools_linux_amd64/WBT \
  pytest tests/test_golden_equivalence.py -v
```

## What CI runs (for context)

Four workflows trigger on every push/PR to `main` — there are **no path
filters**, so docs-only PRs pay the full bill (~40–70 min wall):

| Workflow | Jobs |
| --- | --- |
| `native` | Rust fmt/clippy/test; packaging matrix on 3.10/3.11/3.12 (editable parity, abi3 wheel + sdist builds with content assertions, wheel install + pytest outside the checkout). |
| `equivalence` | The golden suite above with cached WhiteboxTools v2.4.0. |
| `benchmarks` | Schema tests + a crop-profile cold/warm smoke run. |
| `tests` | Functionality smoke suite (97 tests) and the `process_cells` determinism regression (at ambient thread count). |

`builddocs.yml` exists but is `workflow_dispatch`-only and has never run in
this fork — the Sphinx docs are validated locally, not in CI.

## Conventions

- **Conventional Commits** with the scopes already in use in this fork:
  `feat(native):`, `fix(deps):`, `fix(parallel):`, `bench:`, `test:`,
  `docs:`.
- **Squash merges**, one commit per PR, matching repo history.
- Keep PRs scoped: packaging (`pyproject.toml` / `rust/Cargo.toml`) and
  workflow changes belong in their own PRs.
- Dependency management: `pyproject.toml` is the source of truth;
  `requirements.txt` is a hand-maintained mirror (keep it in sync — it has
  drifted before). Versions are hand-synced between `pyproject.toml`
  (`0.0.3a8`) and `rust/Cargo.toml` (`0.0.3-alpha.8`); a version bump means
  both, and no bump without a release decision.

## License

This project is GPL-3.0 (`LICENSE`); Rust sources carry
`SPDX-License-Identifier: GPL-3.0-or-later` headers. By contributing you
agree your contributions are licensed under GPL-3.0.
