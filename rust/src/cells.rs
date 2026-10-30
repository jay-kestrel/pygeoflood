// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from pygeoflood by Paola Passalacqua, Xing Zheng, Alec Carruthers,
// Mark Wang, and Matthew Preisser.

use rayon::prelude::*;

const MAX_CHUNKS: usize = 32;

pub struct Inputs {
    pub hand: Vec<f64>,
    pub catchments: Vec<f64>,
    pub slope: Vec<f64>,
    pub hydroids: Vec<f64>,
    pub heights: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub cell_area: f64,
    pub slope_f32: bool,
    pub delta_f32: bool,
    pub volume_f32: bool,
}

#[derive(Debug, PartialEq)]
pub struct Totals {
    pub counts: Vec<i32>,
    pub surface: Vec<f32>,
    pub bed: Vec<f32>,
    pub volume: Vec<f32>,
}

impl Totals {
    fn zeros(size: usize) -> Self {
        Self {
            counts: vec![0; size],
            surface: vec![0.0; size],
            bed: vec![0.0; size],
            volume: vec![0.0; size],
        }
    }

    fn fold(&mut self, chunk: Self) {
        for i in 0..self.counts.len() {
            self.counts[i] = self.counts[i].wrapping_add(chunk.counts[i]);
            self.surface[i] += chunk.surface[i];
            self.bed[i] += chunk.bed[i];
            self.volume[i] += chunk.volume[i];
        }
    }
}

fn row_range(rows: usize, chunks: usize, chunk: usize) -> std::ops::Range<usize> {
    let base = rows / chunks;
    let remainder = rows % chunks;
    let start = chunk * base + chunk.min(remainder);
    start..start + base + usize::from(chunk < remainder)
}

fn accumulate_chunk(input: &Inputs, rows: std::ops::Range<usize>) -> Result<Totals, &'static str> {
    let ncatch = input.hydroids.len();
    let mut totals = Totals::zeros(input.heights.len() * ncatch);
    for pixel in rows.start * input.cols..rows.end * input.cols {
        let hand = input.hand[pixel];
        let near_zero = hand.abs() <= 1e-8;
        let hydroid = input.catchments[pixel];
        // NumPy searchsorted(side="left") uses the insertion index, not the
        // hydroid table's second column. NaNs sort after all finite IDs.
        let catchment = input
            .hydroids
            .partition_point(|&id| id < hydroid || hydroid.is_nan());
        let slope = input.slope[pixel];
        let square = if input.slope_f32 {
            ((slope as f32) * (slope as f32)) as f64
        } else {
            slope * slope
        };
        let bed = input.cell_area * (1.0 + square).sqrt();
        for (stage, &height) in input.heights.iter().enumerate() {
            if hand < height || near_zero {
                if catchment == ncatch {
                    return Err("active cell hydroid is outside the lookup table");
                }
                let index = stage * ncatch + catchment;
                let delta = if input.delta_f32 {
                    ((height as f32) - (hand as f32)) as f64
                } else {
                    height - hand
                };
                let volume = if input.volume_f32 {
                    ((delta as f32) * (input.cell_area as f32)) as f64
                } else {
                    delta * input.cell_area
                };
                totals.counts[index] = totals.counts[index].wrapping_add(1);
                // Numba stores each update in float32, but promotes these
                // additions when the increment is float64. Do not reassociate.
                totals.surface[index] = (f64::from(totals.surface[index]) + input.cell_area) as f32;
                totals.bed[index] = (f64::from(totals.bed[index]) + bed) as f32;
                totals.volume[index] = (f64::from(totals.volume[index]) + volume) as f32;
            }
        }
    }
    Ok(totals)
}

pub fn process_cells(input: &Inputs) -> Result<Totals, &'static str> {
    let chunks = input.rows.clamp(1, MAX_CHUNKS);
    // Indexed collect retains chunk order regardless of Rayon scheduling.
    let partials: Vec<_> = (0..chunks)
        .into_par_iter()
        .map(|chunk| accumulate_chunk(input, row_range(input.rows, chunks, chunk)))
        .collect();
    let mut totals = Totals::zeros(input.heights.len() * input.hydroids.len());
    for partial in partials {
        totals.fold(partial?);
    }
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop() -> Inputs {
        Inputs {
            hand: vec![0.0, 0.5, 1.0, f64::NAN],
            catchments: vec![10.0; 4],
            slope: vec![0.0; 4],
            hydroids: vec![10.0],
            heights: vec![0.0, 1.0, 2.0],
            rows: 2,
            cols: 2,
            cell_area: 2.0,
            slope_f32: true,
            delta_f32: false,
            volume_f32: false,
        }
    }

    #[test]
    fn single_catchment_crop() {
        let totals = process_cells(&crop()).unwrap();
        assert_eq!(totals.counts, [1, 2, 3]);
        assert_eq!(totals.surface, [2.0, 4.0, 6.0]);
        assert_eq!(totals.bed, totals.surface);
        assert_eq!(totals.volume, [0.0, 3.0, 9.0]);
    }

    #[test]
    fn empty_catchment() {
        let mut input = crop();
        input.hydroids.push(20.0);
        let totals = process_cells(&input).unwrap();
        assert_eq!(totals.counts, [1, 0, 2, 0, 3, 0]);
        for values in [&totals.surface, &totals.bed, &totals.volume] {
            assert!(values.iter().skip(1).step_by(2).all(|&v| v == 0.0));
        }
    }

    #[test]
    fn balanced_partition_covers_all_rows() {
        for rows in [0, 1, 31, 32, 33, 67, 2000] {
            let chunks = rows.clamp(1, MAX_CHUNKS);
            let actual: Vec<_> = (0..chunks)
                .flat_map(|c| row_range(rows, chunks, c))
                .collect();
            assert_eq!(actual, (0..rows).collect::<Vec<_>>());
        }
    }

    #[test]
    fn reduction_independent_of_thread_count() {
        let mut input = crop();
        input.rows = 67;
        input.hand = (0..134).map(|i| (i % 7) as f64 / 7.0).collect();
        input.catchments = vec![10.0; 134];
        input.slope = vec![0.13; 134];
        let reference = process_cells(&input).unwrap();
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..5 {
                assert_eq!(pool.install(|| process_cells(&input)).unwrap(), reference);
            }
        }
    }
}
