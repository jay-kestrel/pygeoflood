"""COBOL anisodiff against the frozen pre-port NumPy implementation.

Mirrors the Rust port's parity matrix (tests/test_anisodiff.py) through
the real PYGEOFLOOD_COBOL dispatch, per the D1 parity gate: every case
the Rust binding covers, the COBOL binding must cover too. Runs require
`make -C cobol lib` (GnuCOBOL 3) — a missing or broken backend raises
rather than silently changing numbers (spec C2, no silent fallbacks).
"""

import ctypes

import numpy as np
import pytest
from reference_anisodiff import anisodiff as reference

from pygeoflood import _cobol, tools


@pytest.fixture(autouse=True)
def cobol_dispatch(monkeypatch):
    """Route every test in this module through the COBOL backend."""
    monkeypatch.setenv("PYGEOFLOOD_COBOL", "1")


def assert_parity(image, **kwargs):
    before = image.copy()
    with np.errstate(all="ignore"):
        expected = reference(image, **kwargs)
        actual = tools.anisodiff(image, **kwargs)
    assert actual.dtype == np.float32
    assert actual.shape == image.shape
    assert not np.shares_memory(actual, image)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    # libm's exp can round differently from NumPy's SIMD exponential
    # (PeronaMalik1); the tolerance row is documented in the spec.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("option", ["PeronaMalik1", "PeronaMalik2", "unknown"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (7, 11), (0, 4), (4, 0)])
@pytest.mark.parametrize("niter", [-2, 0, 1, 10])
def test_stencil_shapes_and_options(option, shape, niter):
    image = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    assert_parity(image, niter=niter, kappa=2.7, gamma=0.1, step=(1.3, 2.1), option=option)


@pytest.mark.parametrize("option", ["PeronaMalik1", "PeronaMalik2", "unknown"])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", [(0, 0), (3, 4), (6, 10)])
def test_nonfinite_directional_masking(option, special, location):
    image = np.arange(77, dtype=np.float32).reshape(7, 11)
    image[location] = special
    assert_parity(image, niter=8, kappa=2.7, gamma=0.1, option=option)


@pytest.mark.parametrize("option", ["PeronaMalik1", "PeronaMalik2"])
@pytest.mark.parametrize("parameter", ["kappa", "gamma", "south", "east"])
def test_numpy_scalar_promotion(option, parameter):
    image = np.random.default_rng(13).normal(size=(11, 13)).astype(np.float32)
    values = {"kappa": 2.7, "gamma": 0.1, "south": 1.3, "east": 2.1}
    values[parameter] = np.float64(values[parameter])
    assert_parity(image, niter=20, kappa=values["kappa"], gamma=values["gamma"],
                  step=(values["south"], values["east"]), option=option)


@pytest.mark.parametrize("layout", ["fortran", "transpose", "slice", "readonly"])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_layouts_and_input_ownership(layout, dtype):
    image = np.arange(120, dtype=dtype).reshape(10, 12)
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "transpose":
        image = image.T
    elif layout == "slice":
        image = image[::-2, ::3]
    else:
        image.flags.writeable = False
    assert_parity(image, niter=5, kappa=3.1, gamma=0.1)


@pytest.mark.parametrize("option", ["PeronaMalik1", "PeronaMalik2"])
@pytest.mark.parametrize("kappa", [0.0, np.nan, np.inf, 1e-30])
def test_degenerate_thresholds(option, kappa):
    image = np.array([[1, 1, 3], [4, 7, 1]], dtype=np.float32)
    assert_parity(image, niter=5, kappa=kappa, gamma=0.1, option=option)


# -- COBOL-specific behavior (beyond the mirrored Rust matrix) ------------


def test_missing_backend_raises_build_me_error(monkeypatch, tmp_path):
    """Routing on + missing .so raises instead of falling back (spec C2)."""
    monkeypatch.setattr(_cobol, "_kernel", None)  # force a fresh load
    monkeypatch.setattr(
        _cobol, "_library_paths", lambda: (tmp_path / "absent.so",)
    )
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(FileNotFoundError, match="not built"):
        tools.anisodiff(image, niter=1, kappa=2.7, gamma=0.1)


def test_oversized_plane_is_rejected_before_the_call(monkeypatch):
    """The OCCURS ceiling is enforced in the adapter with a clear error."""
    monkeypatch.setattr(_cobol, "_MAX_ELEMENTS", 8)
    image = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(_cobol.CobolKernelError, match="ceiling"):
        tools.anisodiff(image, niter=1, kappa=2.7, gamma=0.1)


def test_kernel_error_codes_raise():
    """Nonzero kernel exit statuses surface with their rc message."""
    plane = np.zeros((2, 2), dtype=np.float32)
    out = np.empty_like(plane)
    kernel = _cobol._Kernel.__new__(_cobol._Kernel)  # no .so load
    kernel._entry = lambda *args: 2  # stub exit status
    with pytest.raises(_cobol.CobolKernelError, match="out of memory"):
        kernel(plane, 2, 2, 1, 0, 1.0, 0.1, 1.0, 1.0, 0, out)
    kernel._entry = lambda *args: 7
    with pytest.raises(_cobol.CobolKernelError, match="unrecognized exit status"):
        kernel(plane, 2, 2, 1, 0, 1.0, 0.1, 1.0, 1.0, 0, out)


def test_dispatch_off_uses_rust(monkeypatch):
    """Without the env var the COBOL kernel is never loaded (AC1)."""
    monkeypatch.delenv("PYGEOFLOOD_COBOL", raising=False)
    calls = []
    real_lib = _cobol._lib

    def _spy():
        calls.append(1)
        return real_lib()

    monkeypatch.setattr(_cobol, "_lib", _spy)
    image = np.random.default_rng(7).normal(size=(5, 5)).astype(np.float32)
    with np.errstate(all="ignore"):
        result = tools.anisodiff(image, niter=3, kappa=2.7, gamma=0.1)
    assert not calls
    expected = reference(image, niter=3, kappa=2.7, gamma=0.1)
    np.testing.assert_array_equal(result, expected)


def test_enabled_reflects_the_dispatch_variable(monkeypatch):
    """enabled() is the single dispatch predicate tools.anisodiff uses."""
    monkeypatch.delenv(_cobol.ENV_VAR, raising=False)
    assert not _cobol.enabled()
    monkeypatch.setenv(_cobol.ENV_VAR, "1")
    assert _cobol.enabled()
    monkeypatch.setenv(_cobol.ENV_VAR, "0")
    assert not _cobol.enabled()  # any value but "1" leaves the default route


def test_kernel_entry_point_signature():
    """The .so exports anisodiff_cobol with the documented 11-arg ABI."""
    _cobol._lib()  # raises FileNotFoundError with build instructions if absent
    entry = _cobol._lib()._entry
    assert entry.restype == ctypes.c_int
    assert len(entry.argtypes) == 11
