// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
// Mark Wang, and Matthew Preisser; original diffusion algorithm from GeoNet.

mod diffusion;

use diffusion::{Axis, Method, Parameters};
use numpy::{ndarray::Array2, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
#[pyo3(signature = (dem, niter, kappa, gamma, step=(1.0, 1.0), option="PeronaMalik2", checknan=false, precision=(false, false, false, false)))]
#[allow(clippy::too_many_arguments)]
fn anisodiff<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    niter: usize,
    kappa: f64,
    gamma: f64,
    step: (f64, f64),
    option: &str,
    checknan: bool,
    precision: (bool, bool, bool, bool),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = dem.as_array();
    let shape = view.dim();
    // The read-only NumPy view is zero-copy, but Python aliases can mutate it.
    // Snapshot into the required output buffer while holding the GIL; no Python
    // memory is accessed once Rayon starts. This also handles strided arrays.
    let image = view.iter().copied().collect();
    let method = match option {
        "PeronaMalik1" => Method::PeronaMalik1,
        "PeronaMalik2" => Method::PeronaMalik2,
        _ => Method::Linear,
    };
    let axis = |step, step_double| Axis {
        kappa,
        step,
        kappa_double: precision.0,
        step_double,
        method,
    };
    let parameters = Parameters {
        iterations: niter,
        south: axis(step.0, precision.1),
        east: axis(step.1, precision.2),
        gamma,
        gamma_double: precision.3,
    };
    let output = py
        .allow_threads(move || diffusion::anisodiff(image, shape.1, parameters, checknan))
        .map_err(|_| PyValueError::new_err("anisodiff input contains NaN"))?;
    let output = Array2::from_shape_vec(shape, output)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(anisodiff, module)?)?;
    Ok(())
}
