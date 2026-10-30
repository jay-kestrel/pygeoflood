# Changelog

All notable changes in this repository are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The fork
has no tagged releases yet; versions reference `pyproject.toml`
(hand-synced with `rust/Cargo.toml`).

## [Unreleased]

Fork history on top of upstream
[pygeoflood v0.0.3a8](https://github.com/passah2O/pygeoflood/releases)
(commit `3650c9b`). Everything below merged on **2026-10-30**.

### Added

- **Golden-fixture generation and value-level equivalence suite** (#1):
  `tests/golden/generate_golden_fixtures.py`, the pinned
  `tests/golden/golden_manifest.sha256`, and
  `tests/test_golden_equivalence.py` gating HAND rasters, inundation maps,
  SRC and flood-stage CSVs, WBT integer rasters, binary HAND, and
  channel-network rasters against deterministic reference values.
- **Benchmark harness** (#3): `python -m benchmarks` with `crop`/`full`
  profiles, cold/warm timing, per-stage and per-WhiteboxTools-subprocess
  timings, and environment manifests — plus schema tests under
  `benchmarks/tests/`.
- **Official benchmark baseline** (#7): recorded at `3650c9b` with a
  one-command reproduction script (`benchmarks/reproduce_baseline.sh`).
- **Native Rust extension (`pygeoflood._rust`)** (#5, #8): abi3-py310 PyO3
  build with the `anisodiff` kernel ported to Rayon-parallel Rust, extended
  with a deterministic port of the `process_cells` SRC accumulation (fixed
  row chunks, ordered fold — bitwise-stable across thread counts).
- **`pygeoflood` CLI** (#6): console script and `python -m pygeoflood` with
  read-only `version`, `list-options`, and `check-inputs` commands.

### Fixed

- `process_cells` race condition (#2): parallel chunk reduction is now
  deterministic (bitwise-identical results, with a dedicated regression
  test).
- `psutil` is declared as a dependency (#4); it was imported undeclared.
- WhiteboxTools calls fail fast (#10): `wrapper.run_wbt` turns panicked or
  failed tool calls into exceptions instead of silently returning 0 —
  downstream stages can no longer consume the previous run's stale
  intermediates. Channel-head kernels are guarded as well.
- `pyo3` bumped to 0.29.3 to close five published security advisories
  (#11).

### Known issue

- After #11, `rust/src/lib.rs` does not compile against the locked pyo3
  0.29 API (`error[E0599]: no method named 'allow_threads'`). The break was
  caught by a local build during the 2026-10-30 documentation effort;
  CI could not catch it because GitHub Actions was disabled for the
  organization that day. Fixing it is a code change, out of scope for the
  documentation PR.

### Notes for fork users

- `pip install pygeoflood` installs **upstream's** PyPI release, not this
  fork — see the README for fork install instructions.
