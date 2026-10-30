// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
// Mark Wang, and Matthew Preisser; original diffusion algorithm from GeoNet.

use rayon::prelude::*;

#[derive(Clone, Copy)]
pub enum Method {
    PeronaMalik1,
    PeronaMalik2,
    Linear,
}

#[derive(Clone, Copy)]
pub struct Axis {
    pub kappa: f64,
    pub step: f64,
    pub kappa_double: bool,
    pub step_double: bool,
    pub method: Method,
}

impl Axis {
    fn is_double(self) -> bool {
        !matches!(self.method, Method::Linear) && (self.kappa_double || self.step_double)
    }

    fn flux(self, delta: f32) -> f64 {
        if matches!(self.method, Method::Linear) {
            return f64::from(delta);
        }
        // NumPy weak Python scalars keep float32; explicit float64 scalars
        // promote only at the operation where they enter the expression.
        let coefficient = if self.kappa_double {
            let ratio = f64::from(delta) / self.kappa;
            match self.method {
                Method::PeronaMalik1 => (-(ratio * ratio)).exp(),
                _ => 1.0 / (1.0 + ratio * ratio),
            }
        } else {
            let ratio = delta / self.kappa as f32;
            f64::from(match self.method {
                Method::PeronaMalik1 => (-(ratio * ratio)).exp(),
                _ => 1.0 / (1.0 + ratio * ratio),
            })
        };
        if self.is_double() {
            (coefficient / self.step) * f64::from(delta)
        } else {
            f64::from((coefficient as f32 / self.step as f32) * delta)
        }
    }

    fn divergence(self, outgoing: f32, incoming: Option<f32>) -> f32 {
        // NS/EW first receive a float32 copy of the outgoing flux in NumPy.
        let current = self.flux(outgoing) as f32;
        match incoming {
            None => current,
            Some(delta) if self.is_double() => (f64::from(current) - self.flux(delta)) as f32,
            Some(delta) => current - self.flux(delta) as f32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Parameters {
    pub iterations: usize,
    pub south: Axis,
    pub east: Axis,
    pub gamma: f64,
    pub gamma_double: bool,
}

#[derive(Debug, PartialEq)]
pub struct NanInput;

fn directional_sum(south: f32, east: f32) -> f32 {
    match (south.is_nan(), east.is_nan()) {
        (true, true) => f32::NAN,
        (true, false) => east,
        (false, true) => south,
        (false, false) => south + east,
    }
}

fn updated_pixel(image: &[f32], i: usize, columns: usize, p: Parameters) -> f32 {
    let value = image[i];
    let south_delta = if i + columns < image.len() {
        image[i + columns] - value
    } else {
        0.0
    };
    let east_delta = if i % columns + 1 < columns {
        image[i + 1] - value
    } else {
        0.0
    };
    let north_delta = (i >= columns).then(|| value - image[i - columns]);
    let west_delta = (i % columns > 0).then(|| value - image[i - 1]);
    let south = p.south.divergence(south_delta, north_delta);
    let east = p.east.divergence(east_delta, west_delta);
    // Missing derivatives are ignored independently. NaNs can spread on later
    // iterations, so freezing an input validity mask would change the stencil.
    let change = directional_sum(south, east);
    if p.gamma_double {
        (f64::from(value) + p.gamma * f64::from(change)) as f32
    } else {
        value + p.gamma as f32 * change
    }
}

pub fn anisodiff(
    mut image: Vec<f32>,
    columns: usize,
    parameters: Parameters,
    checknan: bool,
) -> Result<Vec<f32>, NanInput> {
    // Validation is once per call, not once per iteration. Directional NaN
    // arithmetic below is fused into each update instead of allocating masks.
    if checknan && image.iter().any(|value| value.is_nan()) {
        return Err(NanInput);
    }
    if image.is_empty() || parameters.iterations == 0 {
        return Ok(image);
    }
    let mut next = vec![0.0; image.len()];
    for _ in 0..parameters.iterations {
        next.par_iter_mut().enumerate().for_each(|(i, out)| {
            *out = updated_pixel(&image, i, columns, parameters);
        });
        std::mem::swap(&mut image, &mut next);
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> Parameters {
        let axis = Axis {
            kappa: 2.0,
            step: 1.0,
            kappa_double: false,
            step_double: false,
            method: Method::PeronaMalik2,
        };
        Parameters {
            iterations: 5,
            south: axis,
            east: axis,
            gamma: 0.1,
            gamma_double: false,
        }
    }

    #[test]
    fn nan_input_is_rejected_even_with_zero_iterations() {
        for iterations in [0, 5] {
            let p = Parameters {
                iterations,
                ..parameters()
            };
            assert_eq!(anisodiff(vec![1.0, f32::NAN], 2, p, true), Err(NanInput));
        }
    }

    #[test]
    fn constant_and_empty_images_are_unchanged() {
        assert_eq!(
            anisodiff(vec![3.0; 9], 3, parameters(), true).unwrap(),
            vec![3.0; 9]
        );
        assert!(anisodiff(vec![], 0, parameters(), true).unwrap().is_empty());
    }

    #[test]
    fn missing_derivatives_are_masked_independently() {
        assert_eq!(directional_sum(f32::NAN, 2.0), 2.0);
        assert_eq!(directional_sum(2.0, f32::NAN), 2.0);
        assert!(directional_sum(f32::NAN, f32::NAN).is_nan());
        assert_eq!(directional_sum(2.0, -1.0), 1.0);
    }

    #[test]
    fn rayon_thread_count_does_not_change_results() {
        let input: Vec<_> = (0..128 * 128).map(|i| (i % 97) as f32).collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| anisodiff(input.clone(), 128, parameters(), false).unwrap())
        };
        assert_eq!(run(1), run(4));
    }
}
