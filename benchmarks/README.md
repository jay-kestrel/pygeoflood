# pygeoflood benchmark harness

Deliverable D1 of the approved port spec (*Spec: pygeoflood optimized port*,
art_4Ga5fqvh): time the canonical FIM workload so every later claim about
speedups is measured against an honest baseline.

## What it measures

The canonical workload — `pyGeoFlood.run_fim_workflow()` on the OC1mTest
DEM with `custom_Q=500` (findings §6.1) — end to end and per stage, under
two profiles:

| profile | grid | purpose |
|---|---|---|
| `crop` | fixed 2000×2000 window (origin derived deterministically from the input bundle; see below) | fast iteration, equivalence trials, CI smoke |
| `full` | the full 4750×3378 tile (16,045,500 cells @ 1 m) | headline numbers |

Each invocation runs **one cold pass** (first run in a fresh process —
includes the one-off numba JIT compile) and then `--warm-runs` warm passes
(repeated in-process, JIT warm). Cold-vs-warm per-stage deltas are the JIT
tax an AOT port removes.

Per-stage timing separates (spec requirement for deliverable D4):

- every pipeline stage in `run_fim_workflow()` order,
- each **WhiteboxTools** subprocess call (the six native C++ calls),
- the Python/numba kernels: `anisodiff`, `get_fmm_points` (FMM point
  selection), `jit_binary_hand`, `process_cells`, `jit_inun`,
- raster reads, raster writes, CSV writes, and vector writes.

Instrumentation wraps live callables at runtime (instance attributes,
module attributes, delegating dataset proxies) — the measured source is
byte-identical to upstream. Two stages are **bypassed by upstream design**
at 3650c9b: `calculate_geodesic_distance` and `identify_channel_heads`
(commented out of `run_fim_workflow`, upstream issue #12), and
`rasterize_custom_flowline` is conditional (no custom flowline in the
canonical workload). Results report them as `skipped`, never silently
absent.

## Environment manifest

Every run writes `manifest.json` and embeds it in each result:
CPU count and model, total RAM, platform, Python version, numba thread
count (actual and `NUMBA_NUM_THREADS`), WhiteboxTools binary and package
versions, versions of the full dependency set, the checkout's git commit,
the DEM sha256, and grid metadata.

## Running

```bash
# fast profile (crop), cold + 2 warm
python -m benchmarks --profile crop --warm-runs 2

# headline profile (full tile)
python -m benchmarks --profile full --warm-runs 2
```

One documented command reproduces the official baseline from a clean venv
on the pinned sandbox (8 vCPU / 7.77 GB — the manifest records actuals):

```bash
bash benchmarks/reproduce_baseline.sh
```

The script creates a Python 3.12 venv (via `uv` when available, else
`python3.12 -m venv`), installs `-e ".[bench]"`, and runs both profiles
with cold + 2 warm passes. Scratch copies of the DEM and all pipeline
outputs go to `<repo>/../pgf-bench-runs/` (override with `--workdir`) —
the checkout stays clean except for the results you explicitly copy in.

WhiteboxTools downloads its binary on first use; set `WBT_PATH` to a
cached installation to pin it.

## Results

Per invocation: `run-<profile>-<mode>-<n>.json` (one per run, full schema),
`manifest.json`, and `SUMMARY.md` (deterministic markdown — end-to-end
table, run-to-run variance, per-stage table, WBT calls, kernels, I/O,
manifest). Committed baselines live in `results/` — the official one at
`results/baseline-3650c9b/`.

The result schema is contract-tested: `pytest benchmarks/tests/` validates
required keys, the canonical stage list, and the deterministic markdown
writer. That test runs in CI (`.github/workflows/benchmarks.yml`) together
with a crop-profile smoke run whose two warm passes give the run-to-run
variance evidence.

## Variance policy

Two consecutive end-to-end runs agree within 10% of their mean, or the
report states the measured spread explicitly — the variance block in
`SUMMARY.md` and the CLI output always state which of the two applies.
Nothing is silently assumed stable.

## Scope guard

Per the spec's D1 constraint, this PR changes only `benchmarks/` and the
`pyproject.toml` extras (plus the CI workflow that makes the schema test
run in CI, per the D1 verification gate). No pygeoflood source, test, or
data file is modified — that is what makes these numbers a valid baseline
for the port.
