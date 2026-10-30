"""Native anisodiff against the frozen pre-port NumPy implementation."""

import inspect
from importlib.resources import files

import numpy as np
import pytest
from reference_anisodiff import anisodiff as reference

from pygeoflood import _rust, tools


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
    # expf can round differently from NumPy's SIMD exponential (PM1).
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


@pytest.mark.parametrize("iterations", [0, 5])
def test_private_nan_error(iterations):
    image = np.array([[1, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="anisodiff input contains NaN"):
        _rust.anisodiff(image, iterations, 2.0, 0.1, checknan=True)
    result = _rust.anisodiff(image, iterations, 2.0, 0.1, checknan=False)
    assert np.isnan(result[0, 1])


def test_public_signature_is_unchanged():
    assert inspect.signature(tools.anisodiff) == inspect.signature(reference)


def test_bundled_roughness_data():
    assert files("pygeoflood").joinpath("data/COMID_Roughness.csv").is_file()


def test_native_boundary_rejects_wrong_dimension_or_dtype():
    for image in [np.ones(3, dtype=np.float32), np.ones((2, 3), dtype=np.float64)]:
        with pytest.raises(TypeError):
            _rust.anisodiff(image, 1, 2.0, 0.1)
