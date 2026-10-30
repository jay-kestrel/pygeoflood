"""Golden-value equivalence suite for the pygeoflood port (spec D2).

The golden reference corpus is generated from unmodified pygeoflood at
NUMBA_NUM_THREADS=1 on a fixed 2000x2000 crop of the OC1mTest DEM (see
tests/golden/generate_golden_fixtures.py). This suite:

1. runs the generator twice and requires byte-identical sha256 manifests
   (determinism of the reference pipeline),
2. requires both manifests to equal the pinned
   tests/golden/golden_manifest.sha256 byte-for-byte (the reference pin),
3. applies the tolerance table from the spec's Verification section, per
   artifact class, comparing candidate outputs (run 1) against golden
   outputs (run 2),
4. proves discrimination: a deliberately perturbed output must FAIL each
   gate (the current structure-only suite's blind spot),
5. re-applies the tolerance gates to a reduced-window pipeline pair
   produced with the COBOL anisodiff backend (PYGEOFLOOD_COBOL=1) — the
   spec D1 COBOL row. The window is reduced (see cobol_run) because
   GnuCOBOL 3.2's decimal arithmetic makes the pinned-window stage a
   multi-hour computation; the row compares the COBOL run against a
   Python baseline generated at the same reduced window, with the
   pipeline parameters otherwise identical.

Tolerance table (spec art_4Ga5fqvh, Verification):

    WBT integer rasters    byte-identical
    HAND raster            allclose rtol 1e-6
    inundation map         mask exact + depth values allclose rtol 1e-6
    SRC CSV                rtol 1e-6 vs single-thread golden
    flood-stage CSV        rtol 1e-6
    channel-network raster cell set identical (pinned skimage)

plus binary HAND as exact equality (a binary mask; any EDT-for-chamfer
variant must fail, per the spec's failure-mode note).

Generation runs in two subprocesses so the NUMBA_NUM_THREADS=1 pin is
honored at numba import time; the gates themselves only read files.
"""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# generation is subprocess-driven; pin here too for any in-process numba use
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "golden"
GENERATOR = GOLDEN_DIR / "generate_golden_fixtures.py"
PINNED_MANIFEST = GOLDEN_DIR / "golden_manifest.sha256"
DATA_DIR = GOLDEN_DIR / "data"
RUN1 = DATA_DIR / "run1"  # candidate outputs
RUN2 = DATA_DIR / "run2"  # golden reference outputs
RUN3 = DATA_DIR / "run3"  # candidate outputs, COBOL anisodiff backend
RUN2_SMALL = DATA_DIR / "run2_small"  # python baseline at the COBOL row's
                                      # reduced window (generated fresh)
# the COBOL row's reduced window (see cobol_run): GnuCOBOL 3.2's decimal
# arithmetic makes the pinned 2000x2000 x 50-iteration anisodiff stage a
# multi-hour computation; the row keeps the full pipeline parameters and
# shrinks only the window
COBOL_WINDOW = 500
COBOL_LIBRARY = TESTS_DIR.parent / "cobol" / "lib" / "pygeoflood_cobol.so"

# tolerance from the spec's Verification section
RTOL = 1e-6
# inundate() fills non-inundated/NaN cells with this value on disk
# (src/pygeoflood/pygeoflood.py: "set nan values to" -32768); the mask of
# these positions is the load-bearing NaN topology of the inundation map
FIM_NODATA = -32768.0


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_golden_fixtures", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLDEN = _load_generator_module()
ARTIFACT_CLASSES = GOLDEN.ARTIFACT_CLASSES

WBT_INTEGER_CLASSES = [
    "wbt_d8_fdr",
    "wbt_mfd_fac",
    "wbt_basins",
    "wbt_segment_catchments",
]


def file_sha256(path):
    """Return the hex sha256 digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raster(path):
    """Read a single-band raster as (array, profile)."""
    with rio.open(path) as src:
        return src.read(1), src.profile.copy()


def parse_manifest(path):
    """Parse a manifest file into {artifact_class: (sha256, filename)}."""
    entries = {}
    for line in path.read_text().splitlines():
        digest, artifact_class, filename = line.split("  ")
        entries[artifact_class] = (digest, filename)
    return entries


def artifact_path(run_dir, artifact_class):
    return run_dir / ARTIFACT_CLASSES[artifact_class]


# --------------------------------------------------------------------------
# tolerance-table gates (pure functions; reused by the port's equivalence
# runs, where candidate outputs come from the ported pipeline)
# --------------------------------------------------------------------------


def gate_byte_identical(candidate_path, golden_path):
    """Byte-identical artifact (file sha256)."""
    return file_sha256(candidate_path) == file_sha256(golden_path)


def gate_hand_raster(candidate_path, golden_path):
    """HAND raster: allclose rtol 1e-6."""
    candidate, candidate_profile = read_raster(candidate_path)
    golden, golden_profile = read_raster(golden_path)
    if candidate.shape != golden.shape:
        return False
    if candidate.dtype != golden.dtype:
        return False
    if candidate_profile.get("nodata") != golden_profile.get("nodata"):
        return False
    return bool(
        np.allclose(candidate, golden, rtol=RTOL, atol=0.0, equal_nan=True)
    )


def gate_inundation_map(candidate_path, golden_path):
    """Inundation map: mask exact + depth values allclose rtol 1e-6."""
    candidate, candidate_profile = read_raster(candidate_path)
    golden, golden_profile = read_raster(golden_path)
    if candidate.shape != golden.shape or candidate.dtype != golden.dtype:
        return False
    if candidate_profile.get("nodata") != golden_profile.get("nodata"):
        return False
    # mask positions (NaN topology on disk: the -32768 fill) must match exactly
    candidate_mask = candidate == FIM_NODATA
    golden_mask = golden == FIM_NODATA
    if not np.array_equal(candidate_mask, golden_mask):
        return False
    return bool(
        np.allclose(candidate, golden, rtol=RTOL, atol=0.0, equal_nan=True)
    )


def gate_src_csv(candidate_path, golden_path):
    """SRC CSV: numeric columns within rtol 1e-6, identifiers exact."""
    candidate = pd.read_csv(candidate_path)
    golden = pd.read_csv(golden_path)
    if list(candidate.columns) != list(golden.columns):
        return False
    keys = ["HYDROID", "Stage_m"]
    candidate = candidate.sort_values(keys).reset_index(drop=True)
    golden = golden.sort_values(keys).reset_index(drop=True)
    for column in candidate.columns:
        if candidate[column].dtype.kind in "iu":
            if not bool((candidate[column] == golden[column]).all()):
                return False
        elif candidate[column].dtype.kind == "f":
            if not np.allclose(
                candidate[column], golden[column],
                rtol=RTOL, atol=0.0, equal_nan=True,
            ):
                return False
        else:
            if not bool((candidate[column] == golden[column]).all()):
                return False
    return True


def gate_flood_stage_csv(candidate_path, golden_path):
    """Flood-stage CSV: deterministic given SRC; rtol 1e-6."""
    candidate = pd.read_csv(candidate_path)
    golden = pd.read_csv(golden_path)
    if list(candidate.columns) != list(golden.columns):
        return False
    candidate = candidate.sort_values("HYDROID").reset_index(drop=True)
    golden = golden.sort_values("HYDROID").reset_index(drop=True)
    for column in ["HYDROID", "COMID"]:
        if not bool((candidate[column] == golden[column]).all()):
            return False
    return bool(
        np.allclose(
            candidate["Stage_m"], golden["Stage_m"],
            rtol=RTOL, atol=0.0, equal_nan=True,
        )
    )


def gate_channel_network_raster(candidate_path, golden_path):
    """Channel-network raster: channel cell set identical (pinned skimage)."""
    candidate, _ = read_raster(candidate_path)
    golden, _ = read_raster(golden_path)
    if candidate.shape != golden.shape:
        return False
    return bool(
        np.array_equal(
            np.transpose(np.nonzero(candidate > 0)),
            np.transpose(np.nonzero(golden > 0)),
        )
    )


def gate_binary_hand(candidate_path, golden_path):
    """Binary HAND: exact equality (0/1 chamfer-threshold mask)."""
    candidate, _ = read_raster(candidate_path)
    golden, _ = read_raster(golden_path)
    if candidate.shape != golden.shape or candidate.dtype != golden.dtype:
        return False
    return bool(np.array_equal(candidate, golden))


# --------------------------------------------------------------------------
# session fixture: run the generator twice
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_runs():
    """Run golden generation twice; return (candidate_dir, golden_dir)."""
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for run_dir in (RUN1, RUN2):
        shutil.rmtree(run_dir, ignore_errors=True)
        subprocess.run(
            [sys.executable, str(GENERATOR), "--output-dir", str(run_dir)],
            check=True,
            env=env,
            cwd=str(TESTS_DIR.parent),
        )
    return RUN1, RUN2


@pytest.fixture(scope="session")
def cobol_run():
    """Generate the reduced-workload COBOL row: two pipeline runs at a
    smaller crop window, one Python and one with the COBOL anisodiff
    backend, and return (baseline_dir, cobol_dir).

    Requires the built backend (make -C cobol lib, GnuCOBOL 3); without it
    the COBOL row is skipped so the suite still runs the reference-pipeline
    gates. Both runs route through the real generator subprocess — the
    COBOL one through the real PYGEOFLOOD_COBOL dispatch, no in-process
    shortcuts.

    The window is reduced (COBOL_WINDOW, default 500) because GnuCOBOL
    3.2 lowers every arithmetic verb to decimal runtime calls: the
    anisodiff stage at the pinned 2000x2000 window and 50 iterations
    measures ~160 minutes on this class of runner, which no CI budget
    survives. At the reduced window the same pipeline (same n_iter,
    option, discharge; only the window shrinks) completes the stage in
    minutes. Gate structure is unchanged; the pinned manifest is not
    consulted on this row (the reduced window has no pin — generator
    fidelity is already pinned by the 2000x2000 run 1/run 2 gates).
    """
    if not COBOL_LIBRARY.is_file():
        pytest.skip("COBOL backend not built — run `make -C cobol lib`")
    for run_dir in (RUN2_SMALL, RUN3):
        shutil.rmtree(run_dir, ignore_errors=True)
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(RUN2_SMALL),
         "--window-size", str(COBOL_WINDOW)],
        check=True,
        env=dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        cwd=str(TESTS_DIR.parent),
    )
    env = dict(
        os.environ,
        NUMBA_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYGEOFLOOD_COBOL="1",
    )
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(RUN3),
         "--window-size", str(COBOL_WINDOW)],
        check=True,
        env=env,
        cwd=str(TESTS_DIR.parent),
    )
    return RUN2_SMALL, RUN3


# --------------------------------------------------------------------------
# determinism + pin tests
# --------------------------------------------------------------------------


class TestGoldenDeterminism:
    def test_regeneration_manifests_byte_identical(self, golden_runs):
        """Two consecutive generations produce identical manifests."""
        run1, run2 = golden_runs
        manifest1 = (run1 / "manifest.sha256").read_bytes()
        manifest2 = (run2 / "manifest.sha256").read_bytes()
        assert manifest1 == manifest2, (
            "golden regeneration is not deterministic; the reference "
            "pipeline must produce byte-identical artifacts"
        )

    def test_manifest_matches_pinned_reference(self, golden_runs):
        """Both manifests equal the pinned manifest byte-for-byte."""
        assert PINNED_MANIFEST.is_file(), (
            f"{PINNED_MANIFEST} is missing: pin the reference manifest "
            "(tests/golden/golden_manifest.sha256) produced by the "
            "generator"
        )
        pinned = PINNED_MANIFEST.read_bytes()
        run1, run2 = golden_runs
        assert (run1 / "manifest.sha256").read_bytes() == pinned
        assert (run2 / "manifest.sha256").read_bytes() == pinned

    def test_manifest_lists_every_spec_artifact_class(self, golden_runs):
        """The manifest covers every artifact class in the spec."""
        run1, _ = golden_runs
        manifest_classes = set(parse_manifest(run1 / "manifest.sha256"))
        assert manifest_classes == set(ARTIFACT_CLASSES)


# --------------------------------------------------------------------------
# tolerance-table equivalence tests (candidate = run1, golden = run2)
# --------------------------------------------------------------------------


class TestToleranceTable:
    @pytest.mark.parametrize("artifact_class", WBT_INTEGER_CLASSES)
    def test_wbt_integer_rasters_byte_identical(
        self, golden_runs, artifact_class
    ):
        run1, run2 = golden_runs
        assert gate_byte_identical(
            artifact_path(run1, artifact_class),
            artifact_path(run2, artifact_class),
        )

    def test_hand_raster_within_rtol(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_hand_raster(
            artifact_path(run1, "hand_raster"),
            artifact_path(run2, "hand_raster"),
        )

    def test_inundation_mask_exact_and_depths_within_rtol(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_inundation_map(
            artifact_path(run1, "inundation_map"),
            artifact_path(run2, "inundation_map"),
        )

    def test_src_csv_within_rtol(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_src_csv(
            artifact_path(run1, "src_csv"),
            artifact_path(run2, "src_csv"),
        )

    def test_flood_stage_csv_within_rtol(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_flood_stage_csv(
            artifact_path(run1, "flood_stage_csv"),
            artifact_path(run2, "flood_stage_csv"),
        )

    def test_channel_network_cell_set_identical(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_channel_network_raster(
            artifact_path(run1, "channel_network_raster"),
            artifact_path(run2, "channel_network_raster"),
        )

    def test_binary_hand_exact(self, golden_runs):
        run1, run2 = golden_runs
        assert gate_binary_hand(
            artifact_path(run1, "binary_hand"),
            artifact_path(run2, "binary_hand"),
        )


# --------------------------------------------------------------------------
# COBOL-backend row (spec D1): the same tolerance gates applied to a
# candidate generated with PYGEOFLOOD_COBOL=1. WBT integer rasters are
# intentionally NOT gated here: those byte-identical checks verify the
# reference pipeline's determinism (run 1 vs run 2), while WBT consumes
# the anisodiff-filtered DEM and its integer outputs may flip under the
# kernel's documented rtol 1e-6 float32 differences.
# --------------------------------------------------------------------------


class TestCobolEquivalence:
    """Tolerance-table gates for the COBOL-routed pipeline run.

    The COBOL row compares a COBOL-routed pipeline generation against a
    Python baseline generated at the same reduced window (the cobol_run
    fixture generates both). WBT integer rasters are intentionally NOT
    gated here: those byte-identical checks verify the reference
    pipeline's determinism (run 1 vs run 2), while WBT consumes the
    anisodiff-filtered DEM and its integer outputs may flip under the
    kernel's documented rtol 1e-6 float32 differences.
    """

    def test_cobol_hand_raster_within_rtol(self, cobol_run):
        baseline, cobol_dir = cobol_run
        assert gate_hand_raster(
            artifact_path(cobol_dir, "hand_raster"),
            artifact_path(baseline, "hand_raster"),
        )

    def test_cobol_inundation_mask_exact_and_depths_within_rtol(
        self, cobol_run
    ):
        baseline, cobol_dir = cobol_run
        assert gate_inundation_map(
            artifact_path(cobol_dir, "inundation_map"),
            artifact_path(baseline, "inundation_map"),
        )

    def test_cobol_src_csv_within_rtol(self, cobol_run):
        baseline, cobol_dir = cobol_run
        assert gate_src_csv(
            artifact_path(cobol_dir, "src_csv"),
            artifact_path(baseline, "src_csv"),
        )

    def test_cobol_flood_stage_csv_within_rtol(self, cobol_run):
        baseline, cobol_dir = cobol_run
        assert gate_flood_stage_csv(
            artifact_path(cobol_dir, "flood_stage_csv"),
            artifact_path(baseline, "flood_stage_csv"),
        )

    def test_cobol_channel_network_cell_set_identical(self, cobol_run):
        baseline, cobol_dir = cobol_run
        assert gate_channel_network_raster(
            artifact_path(cobol_dir, "channel_network_raster"),
            artifact_path(baseline, "channel_network_raster"),
        )

    def test_cobol_binary_hand_exact(self, cobol_run):
        baseline, cobol_dir = cobol_run
        assert gate_binary_hand(
            artifact_path(cobol_dir, "binary_hand"),
            artifact_path(baseline, "binary_hand"),
        )


# --------------------------------------------------------------------------
# discrimination check: a perturbed output must FAIL each gate
# --------------------------------------------------------------------------


def _finite_value_cell(arr, low, high):
    """Deterministically pick a cell with low < value < high, or fail."""
    cells = np.argwhere((arr > low) & (arr < high))
    assert cells.size, "no cell with a value in the requested range"
    return tuple(cells[0])


@pytest.fixture(scope="session")
def perturbed_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("perturbed")


def _write_perturbed_raster(arr, profile, path):
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


class TestDiscrimination:
    """A deliberately wrong output must fail the gates (spec D2)."""

    def test_hand_raster_gate_rejects_perturbation(
        self, golden_runs, perturbed_dir
    ):
        run1, run2 = golden_runs
        candidate, profile = read_raster(artifact_path(run1, "hand_raster"))
        row, col = _finite_value_cell(candidate, 0.0, 1.0)
        candidate[row, col] += 1e-3  # 1 mm bump: far beyond rtol 1e-6
        path = perturbed_dir / "hand.tif"
        _write_perturbed_raster(candidate, profile, path)
        assert not gate_hand_raster(path, artifact_path(run2, "hand_raster"))

    def test_inundation_gate_rejects_mask_and_depth_perturbation(
        self, golden_runs, perturbed_dir
    ):
        run1, run2 = golden_runs
        golden_path = artifact_path(run2, "inundation_map")
        candidate, profile = read_raster(artifact_path(run1, "inundation_map"))
        # flip one masked cell to a (wrong) depth value
        masked_cells = np.argwhere(candidate == FIM_NODATA)
        assert masked_cells.size, "inundation map has no masked cells"
        row, col = tuple(masked_cells[0])
        candidate[row, col] = 0.0
        mask_path = perturbed_dir / "fim_mask.tif"
        _write_perturbed_raster(candidate, profile, mask_path)
        assert not gate_inundation_map(mask_path, golden_path)

        # bump one real depth value beyond tolerance
        candidate2, _ = read_raster(artifact_path(run1, "inundation_map"))
        depth_cell = _finite_value_cell(candidate2, 0.0, 1.0)
        candidate2[depth_cell] += 1e-3
        depth_path = perturbed_dir / "fim_depth.tif"
        _write_perturbed_raster(candidate2, profile, depth_path)
        assert not gate_inundation_map(depth_path, golden_path)

    def test_src_csv_gate_rejects_perturbation(self, golden_runs, perturbed_dir):
        run1, run2 = golden_runs
        candidate = pd.read_csv(artifact_path(run1, "src_csv"))
        # scale a row where the volume is nonzero (row 0 is the Stage 0 row,
        # where every accumulation is 0 and scaling would be a no-op)
        positive_rows = candidate.index[candidate["Volume_m3"] > 0]
        assert len(positive_rows), "SRC has no nonzero Volume_m3 rows"
        candidate.loc[positive_rows[0], "Volume_m3"] *= 1.001
        path = perturbed_dir / "src.csv"
        candidate.to_csv(path, index=False)
        assert not gate_src_csv(path, artifact_path(run2, "src_csv"))

    def test_flood_stage_gate_rejects_perturbation(
        self, golden_runs, perturbed_dir
    ):
        run1, run2 = golden_runs
        candidate = pd.read_csv(artifact_path(run1, "flood_stage_csv"))
        candidate.loc[0, "Stage_m"] *= 1.001
        path = perturbed_dir / "flood_stage.csv"
        candidate.to_csv(path, index=False)
        assert not gate_flood_stage_csv(path, artifact_path(run2, "flood_stage_csv"))

    def test_channel_network_gate_rejects_perturbation(
        self, golden_runs, perturbed_dir
    ):
        run1, run2 = golden_runs
        candidate, profile = read_raster(
            artifact_path(run1, "channel_network_raster")
        )
        channel_cells = np.argwhere(candidate > 0)
        assert channel_cells.size, "channel network raster is empty"
        row, col = tuple(channel_cells[0])
        candidate[row, col] = 0  # drop one channel cell from the set
        path = perturbed_dir / "channel_network.tif"
        _write_perturbed_raster(candidate, profile, path)
        assert not gate_channel_network_raster(
            path, artifact_path(run2, "channel_network_raster")
        )

    def test_binary_hand_gate_rejects_perturbation(
        self, golden_runs, perturbed_dir
    ):
        run1, run2 = golden_runs
        candidate, profile = read_raster(artifact_path(run1, "binary_hand"))
        marked_cells = np.argwhere(candidate > 0)
        assert marked_cells.size, "binary HAND has no marked cells"
        row, col = tuple(marked_cells[0])
        candidate[row, col] = 0
        path = perturbed_dir / "binary_hand.tif"
        _write_perturbed_raster(candidate, profile, path)
        assert not gate_binary_hand(path, artifact_path(run2, "binary_hand"))

    @pytest.mark.parametrize("artifact_class", WBT_INTEGER_CLASSES)
    def test_wbt_raster_gates_reject_perturbation(
        self, golden_runs, perturbed_dir, artifact_class
    ):
        run1, run2 = golden_runs
        candidate, profile = read_raster(artifact_path(run1, artifact_class))
        candidate[tuple(np.array(candidate.shape) // 2)] += 1
        path = perturbed_dir / f"{artifact_class}.tif"
        _write_perturbed_raster(candidate, profile, path)
        assert not gate_byte_identical(path, artifact_path(run2, artifact_class))
