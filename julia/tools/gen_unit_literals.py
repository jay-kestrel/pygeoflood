# SPDX-License-Identifier: GPL-3.0-or-later
"""Print Julia literals for small-case unit tests, computed from the live
pygeoflood kernels (NUMBA_NUM_THREADS=1). Run from repo root:

    NUMBA_NUM_THREADS=1 .venv/bin/python julia/tools/gen_unit_literals.py
"""

import os
import sys
from pathlib import Path

if os.environ.get("NUMBA_NUM_THREADS", "1") != "1":
    raise SystemExit("NUMBA_NUM_THREADS must be 1")
os.environ["NUMBA_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from pygeoflood.tools import jit_binary_hand, jit_inun, process_cells


def jvec(a):
    if a.ndim == 1:
        return "[" + ", ".join(repr(float(v)) for v in a) + "]"
    rows = ["[" + ", ".join(repr(float(v)) for v in row) + "]" for row in a]
    return "[" + ";\n  ".join(rows) + "]"


def jintmat(a):
    rows = ["[" + ", ".join(str(int(v)) for v in row) + "]" for row in a]
    return "[" + ";\n  ".join(rows) + "]"


print("# process_cells: hand-computed crop (mirrors rust/src/cells.rs tests)")
hand = np.array([[0.0, 0.5, 1.0, np.nan]], dtype=np.float32)
catch = np.full((1, 4), 10, dtype=np.int32)
slope = np.zeros((1, 4), dtype=np.float32)
hyd = np.array([[10, 0]], dtype=np.int32)
heights = np.array([0.0, 1.0, 2.0])
counts, surface, bed, volume = process_cells(
    hand, catch, slope, hyd, heights, np.float64(2.0)
)
print("COUNTS =", jintmat(counts.T))
print("SURFACE =", jvec(surface.T))
print("BED =", jvec(bed.T))
print("VOLUME =", jvec(volume.T))

print("\n# insertion-index semantics: catchments=15 folds into slot 2 (1-based)")
catch15 = np.full((1, 4), 15, dtype=np.int32)
hyd2 = np.array([[10, 0], [20, 1]], dtype=np.int32)
c2, s2, b2, v2 = process_cells(hand, catch15, slope, hyd2, heights, np.float64(2.0))
print("COUNTS_15 =", jintmat(c2.T))

print("\n# near-zero boundary: |hand| <= 1e-8 counts even above stage 0")
handz = np.array(
    [[0.0, 1e-9, -1e-9, 1e-8, -1e-8, 1.01e-8, 0.5]], dtype=np.float32
)
catchz = np.full((1, 7), 10, dtype=np.int32)
slopez = np.zeros((1, 7), dtype=np.float32)
cz, sz, bz, vz = process_cells(
    handz, catchz, slopez, hyd, np.array([0.0]), np.float64(1.0)
)
print("NEARZERO_COUNTS =", jintmat(cz.T))
print("NEARZERO_SURFACE =", jvec(sz.T))

print("\n# binary_hand: 3x3, flowline center, dem 10 except center 4")
dem3 = np.full((3, 3), 10.0, dtype=np.float32)
dem3[1, 1] = 4.0
fl3 = np.zeros((3, 3), dtype=np.int16)
fl3[1, 1] = 1
bh3 = jit_binary_hand(dem3, fl3, -32768.0)
print("BINARY_HAND_3x3 =", jintmat(bh3))

print("\n# binary_hand: 4x4 with NaN nodata corner (NaN dem -> 1 quirk pre-driver-fix)")
dem4 = np.full((4, 4), 7.5, dtype=np.float32)
dem4[0, 0] = np.nan
fl4 = np.zeros((4, 4), dtype=np.int16)
fl4[3, 3] = 1
bh4 = jit_binary_hand(dem4, fl4, -32768.0)
print("BINARY_HAND_4x4_NAN =", jintmat(bh4))

print("\n# inun: stage lookup + sentinel")
hand_i = np.array([[1.0, 2.0, 9.0], [0.5, 6.0, 3.0]], dtype=np.float32)
segc = np.array([[1, 2, 99], [2, 1, 7]], dtype=np.int32)
hyd_f = np.array([1.0, 2.0, 7.0])
stage = np.array([5.0, 3.0, 10.0])
inun = jit_inun(hand_i, segc, hyd_f, stage)
print("INUN =", jvec(inun))
