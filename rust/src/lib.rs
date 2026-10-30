// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
// Mark Wang, and Matthew Preisser; original diffusion algorithm from GeoNet.

use pyo3::prelude::*;

#[pymodule]
fn _rust(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
