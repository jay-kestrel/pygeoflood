# Golden reference fixtures (spec D2)

Deterministic reference outputs from **unmodified pygeoflood**, gating every
ported kernel against value-level correctness (deliverable D2 of the
pygeoflood optimized-port spec, `art_4Ga5fqvh`). Before this corpus existed,
the repo's test suite validated structure only — shape/CRS/dtype/non-emptiness
— which would let a numerically wrong port ship silently.

## What is here

| File | Role |
| --- | --- |
| `generate_golden_fixtures.py` | Regenerates the full golden corpus: crops the OC1m DEM, clips the vectors, runs the unmodified `run_fim_workflow`, writes the sha256 manifest |
| `golden_manifest.sha256` | **Pinned reference manifest** — one `<sha256>  <class>  <filename>` line per golden artifact class |
| `data/` | Gitignored scratch where generated corpora land (`run1/`, `run2/`) |

## Golden artifact classes

Every class from the spec's tolerance table:

1. **HAND raster** (`*_HAND.tif`)
2. **Inundation map** (`*_fim.tif`) — mask positions are load-bearing
3. **SRC CSV** (`*_src.csv`)
4. **Flood-stage CSV** (`*_flood_stage.csv`)
5. **WBT integer rasters** — D8 flow direction (`*_d8_fdr.tif`), MFD flow
   accumulation (`*_mfd_fac.tif`), basins (`*_basins.tif`), segment
   catchments (`*_segment_catchments.tif`)
6. **Binary HAND** (`*_binary_hand.tif`)
7. **Channel-network raster** (`*_channel_network_raster.tif`)

## Storage decision

The generated corpus is ~25 MB per run — far above the "a few MB" threshold —
so the fixtures are **not committed**. This directory commits the regeneration
script and the pinned manifest instead; CI regenerates the corpus before the
equivalence tests run (see `.github/workflows/equivalence.yml`). Regeneration
takes well under a minute on the 2000x2000 crop.

## How the corpus is generated

- **Inputs:** a fixed 2000x2000 window of `tests/test_data/OC1mTest_inputs/OC1mTest.tif`,
  centered on the DEM (window origin row 689, col 1375). The flowline and
  catchment vectors are clipped to the same window, inset by 10 m so clipped
  flowline endpoints stay strictly inside the DEM grid (an endpoint exactly on
  the edge converts to a row/col one cell outside the cost array and crashes
  least-cost routing).
- **Reference conditions:** `NUMBA_NUM_THREADS=1`. The parallel `process_cells`
  kernel (SRC accumulation) has a confirmed data race and thread-count-dependent
  floating-point accumulation order (findings `art_0v8nRJ7D`), so only the
  single-thread run is a valid reference. The generator enforces the pin before
  numba is imported.
- **Flood stage:** constant `custom_Q=500` (the canonical example value), so
  the fixture is self-contained without the CONUS-wide NWM forecast file.
- **WhiteboxTools:** pinned to **v2.4.0**, resolved via `WBT_PATH`. The
  generator fails if the resolved binary is any other version.

## Pinned environment (byte-identical regeneration)

Byte-identical manifests depend on pinned library versions, not just pinned
code. The pins used for the current manifest are recorded in
`tests/golden/requirements-pins.txt`; install with:

```bash
pip install -c tests/golden/requirements-pins.txt -e . pytest
```

Key pins: numpy 2.5.4, scipy 1.18.1, scikit-image 0.26.0, numba 0.68.1,
rasterio 1.5.2, geopandas 1.2.0, shapely 2.2.0, pandas 3.1.0, fiona 1.10.1,
whitebox 2.3.6, Python 3.12. scikit-image is pinned deliberately:
`route_through_array` tie-breaking is version-sensitive (findings
`art_OZ9DYcjn` §6), and the channel-network gate compares cell sets.

Each generation also writes a `versions.json` sidecar (environment record,
not hashed) next to the manifest.

## Regenerating and re-pinning

```bash
# regenerate twice; manifests must be byte-identical
python tests/golden/generate_golden_fixtures.py --output-dir tests/golden/data/run1
python tests/golden/generate_golden_fixtures.py --output-dir tests/golden/data/run2
diff tests/golden/data/run1/manifest.sha256 tests/golden/data/run2/manifest.sha256

# re-pin (only when inputs, window, or pinned versions deliberately change)
cp tests/golden/data/run1/manifest.sha256 tests/golden/golden_manifest.sha256
```

Changing the crop window, the input crop, the flood-stage mode, or any pinned
version invalidates the pinned manifest — regenerate and re-pin in the same
change, and say so in the PR description.

The equivalence suite (`tests/test_golden_equivalence.py`) enforces all of
this: it runs the generator twice, requires byte-identical manifests equal to
the pinned reference, applies the tolerance-table gates per artifact class,
and proves discrimination — a deliberately perturbed output must fail each
gate.
