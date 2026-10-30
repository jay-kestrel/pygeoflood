// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
// Mark Wang, and Matthew Preisser; original diffusion algorithm from GeoNet.

mod cells;
mod diffusion;

use diffusion::{Axis, Method, Parameters};
use numpy::{ndarray::Array2, Element, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

type CellArrays<'py> = (
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
);

fn array2<T: Element>(
    py: Python<'_>,
    shape: (usize, usize),
    values: Vec<T>,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let array = Array2::from_shape_vec(shape, values)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array))
}

#[pyfunction]
#[pyo3(signature = (hand, segment_catchments, slope, hydroids, heights, cell_area, precision=(false, false, false)))]
#[allow(clippy::too_many_arguments)]
fn process_cells<'py>(
    py: Python<'py>,
    hand: PyReadonlyArray2<'py, f64>,
    segment_catchments: PyReadonlyArray2<'py, f64>,
    slope: PyReadonlyArray2<'py, f64>,
    hydroids: PyReadonlyArray1<'py, f64>,
    heights: PyReadonlyArray1<'py, f64>,
    cell_area: f64,
    precision: (bool, bool, bool),
) -> PyResult<CellArrays<'py>> {
    let hand = hand.as_array();
    let catchments = segment_catchments.as_array();
    let slope = slope.as_array();
    let hydroids = hydroids.as_array();
    let heights = heights.as_array();
    if hand.dim() != catchments.dim() || hand.dim() != slope.dim() {
        return Err(PyValueError::new_err(
            "hand, segment_catchments and slope shapes must match",
        ));
    }
    if hydroids.iter().any(|id| !id.is_finite())
        || hydroids
            .iter()
            .zip(hydroids.iter().skip(1))
            .any(|(a, b)| a > b)
    {
        return Err(PyValueError::new_err("hydroids must be finite and sorted"));
    }
    let shape = (heights.len(), hydroids.len());
    shape
        .0
        .checked_mul(shape.1)
        .and_then(|size| size.checked_mul(32))
        .ok_or_else(|| PyValueError::new_err("accumulator shape is too large"))?;
    // Snapshot all arrays under the GIL: aliases may be mutated by Python
    // after allow_threads, even though these NumPy borrows are read-only.
    let input = cells::Inputs {
        hand: hand.iter().copied().collect(),
        catchments: catchments.iter().copied().collect(),
        slope: slope.iter().copied().collect(),
        hydroids: hydroids.iter().copied().collect(),
        heights: heights.iter().copied().collect(),
        rows: hand.nrows(),
        cols: hand.ncols(),
        cell_area,
        slope_f32: precision.0,
        delta_f32: precision.1,
        volume_f32: precision.2,
    };
    let totals = py
        .allow_threads(move || cells::process_cells(&input))
        .map_err(PyValueError::new_err)?;
    Ok((
        array2(py, shape, totals.counts)?,
        array2(py, shape, totals.surface)?,
        array2(py, shape, totals.bed)?,
        array2(py, shape, totals.volume)?,
    ))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(anisodiff, module)?)?;
    module.add_function(wrap_pyfunction!(process_cells, module)?)?;
    Ok(())
}
