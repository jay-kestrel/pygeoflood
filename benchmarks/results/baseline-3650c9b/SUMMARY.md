# Official baseline — pygeoflood at commit 3650c9b (spec deliverable D1)

Reference numbers for the optimized-port comparison (deliverable D4). The
fairness rule from the spec: baselines are the **numba-warm** variant for
JIT-free comparisons and the **numba cache=True** variant for fresh-process
comparisons — never JIT-tax-inflated cold numbers alone.

## Protocol

- Pinned source: upstream commit `3650c9b` (v0.0.3a8), clean checkout, no
  source modifications. The first full-tile invocation recorded commit
  `233ee90`; its `src/` tree is byte-identical to `3650c9b` (audited), so
  all runs measure the same code.
- Environment: clean Python 3.12.16 venv, `pip install -e ".[bench]"`;
  full package versions in each `manifest.json` (numpy 2.5.4, numba
  0.68.1, rasterio 1.5.2, geopandas 1.2.0, …).
- WhiteboxTools v2.4.0, binary pinned via `WBT_PATH`;
  `wbt_max_procs=1` in every manifest (upstream FillDepressions threading
  race, whitebox-tools issue #476; also makes WBT stages machine-stable).
- `NUMBA_NUM_THREADS=8`; 8-vCPU / 7.77 GB sandbox (actuals in manifests).
- One benchmark process per invocation; each result JSON embeds the
  manifest of the process that recorded it. Harness: this repo at
  `692087b`.
- Reproduce: `bash benchmarks/reproduce_baseline.sh` (standard variant),
  `python -m benchmarks --profile <p> --warm-runs <n> --numba-cache`
  (cache variant). Verify against the committed data:
  `pytest benchmarks/tests/test_baseline_results.py`.

## Directories

| directory | variant | profile | runs |
|---|---|---|---|
| `full/` | numba-warm | full tile | 3 cold + 2 warm |
| `crop/` | numba-warm | 2000×2000 crop | 4 cold + 5 warm |
| `full-cache/` | cache=True | full tile | 3 cold |
| `crop-cache/` | cache=True | 2000×2000 crop | 2 cold + 2 warm |

## Headline medians (end to end, wall clock)

| variant | profile | cold median (s) | warm median (s) | warm spread |
|---|---|---|---|---|
| numba-warm | full | 97.92 | 76.26 | 2.18% ✓ |
| numba-warm | crop | 28.93 | 18.68 | 22.89% ⚠ see variance |
| cache=True | full | 83.02 | — | — |
| cache=True | crop | 18.94 | 19.00 | 9.52% ✓ |

## Per-kernel medians (crop profile)

The kernel-level comparison basis for D4. Cold medians include the
one-off numba JIT compile; cold − warm is the JIT tax per kernel.

| kernel | host stage | cold median std (s) | warm median std (s) | cold median cache (s) |
|---|---|---|---|---|
| anisodiff | apply_nonlinear_filter | 3.92 | 4.04 | 4.08 |
| jit_binary_hand | calculate_binary_hand | 7.22 | 0.12 | 0.35 |
| process_cells | calculate_src | 3.95 | 1.76 | 2.07 |
| jit_inun | inundate | 1.25 | 0.00 | 0.01 |

(Warm medians are over the five standard-variant warm reps; warm medians
per kernel: anisodiff 4.040 s, jit_binary_hand 0.124 s, process_cells
1.762 s, jit_inun 0.002 s.)

## Run-to-run variance

Within-window agreement is tight; the crop profile's aggregate warm spread
exceeds the 10% guidance because of machine-load drift between run
windows, stated here explicitly per the variance policy:

- All runs were recorded 2026-10-30 UTC in two windows: the original
  committed runs at 16:57–17:33, the top-up runs at 18:14–18:26.
- **Within-window spreads:** full-tile warms (one invocation, two
  passes) 2.18%; full-cache colds (three invocations, 18:14–18:17)
  3.96%; crop-cache warms 9.52%; the original crop warms 3.70%.
- **Cross-window drift:** every stage measured in the second window ran
  ~12–17% slower than the first. Crop warms moved 15.66/16.25 s →
  18.68–19.74 s (+~17%); full-tile colds moved 87.40 s → 98.24/97.92 s
  (+12%); the anisodiff warm kernel moved 2.44/2.58 s → 4.04–4.31 s.
  This is environment drift on a shared sandbox, not harness noise —
  hence the crop profile's aggregate 22.89% warm spread
  (15.66 → 19.74 s, mean 17.83 s).
- **Consequence:** cross-window medians carry roughly ±8% window
  uncertainty; same-window comparisons are tight. The cache-variant runs
  (18:14–18:20) and the standard crop top-up runs (18:22–18:24) are
  near-contemporaneous, so the variant comparison below is a fair
  same-window comparison. The D4 comparison should treat baseline
  medians with this ±8% band unless it re-runs the baseline protocol in
  the same window as the ported build.

## Variant comparison: numba-warm vs cache=True (cold process)

With `numba cache=True`, dispatchers cache compiled machine code on disk,
so a fresh process skips the per-process JIT recompile:

- Full tile, cold: std 98.24/97.92 s (18:24–18:26) vs cache
  82.47–85.74 s (18:14–18:17) — **13–16% faster** end to end.
- Crop, cold: std 28.66–29.28 s vs cache 18.45–19.43 s — **34–37%
  faster**; a cache cold costs the same as a warm pass
  (18.45–19.43 s cold vs 18.10–19.91 s warm), i.e. a fresh process with
  a warm disk cache behaves like a warm process.
- Warm-in-process times are unaffected by the flag, as expected: crop
  std warm median 18.68 s vs crop-cache warm median 19.00 s
  (overlapping ranges).

For D4 this means: the AOT port should be compared against numba-warm
numbers for in-process comparisons and against cache=True colds for
fresh-process comparisons. A ported build that eliminates JIT entirely
should land at or below the cache=True cold numbers.

## Upstream drift since the pin (relevant to the D4 comparison)

`main` has moved past the pin: `83ab7d3` (process_cells deterministic
chunk reduction) and `eb8bf63` (psutil declared) landed on `main` after
`3650c9b`. This baseline deliberately measures the pinned `3650c9b`
source. The ported-build benchmark run will record its own
commit SHA and manifest; the D4 comparison must account for the
deterministic-reduction change to `process_cells` (its behavior and cost
at `3650c9b` are what the kernel table above records) so that differences
are attributed to the port rather than to the pin boundary.
