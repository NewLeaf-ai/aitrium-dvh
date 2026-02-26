use ndarray::Array2;

/// Histogram calculation matching numpy.histogram behavior
pub struct HistogramCalculator;

impl HistogramCalculator {
    /// Calculate differential histogram from dose values
    /// Matches numpy.histogram with bins in cGy
    pub fn calculate_histogram(
        dose_plane: &Array2<f32>,
        mask: &Array2<bool>,
        dose_scaling: f64,
        max_dose_cgy: u32,
    ) -> (Vec<f64>, f64) {
        assert_eq!(
            dose_plane.shape(),
            mask.shape(),
            "Dose and mask shapes must match"
        );

        let mut histogram = vec![0.0; max_dose_cgy as usize];
        let mut voxel_count = 0u64;

        // Process each voxel
        for ((r, c), &is_inside) in mask.indexed_iter() {
            if is_inside {
                let dose_value = dose_plane[[r, c]] as f64;
                let dose_cgy = dose_value * dose_scaling * 100.0;

                // Clamp to valid range and floor to get bin index
                let bin_index = if dose_cgy < 0.0 {
                    0
                } else if dose_cgy >= max_dose_cgy as f64 {
                    max_dose_cgy as usize - 1
                } else {
                    dose_cgy.floor() as usize
                };

                histogram[bin_index] += 1.0;
                voxel_count += 1;
            }
        }

        (histogram, voxel_count as f64)
    }

    /// Calculate histogram with direct accumulation (for scanline mode)
    pub fn create_accumulator(
        dose_scaling: f64,
        max_dose_cgy: u32,
    ) -> impl FnMut(f32) -> Option<usize> {
        move |dose_value: f32| {
            let dose_cgy = dose_value as f64 * dose_scaling * 100.0;

            if dose_cgy < 0.0 {
                Some(0)
            } else if dose_cgy >= max_dose_cgy as f64 {
                Some(max_dose_cgy as usize - 1)
            } else {
                Some(dose_cgy.floor() as usize)
            }
        }
    }

    /// Convert differential histogram to cumulative
    /// Matches dvh.DVH.cumulative behavior
    /// Cumulative[i] = volume receiving dose >= i
    pub fn to_cumulative(differential: &[f64]) -> Vec<f64> {
        let mut cumulative = vec![0.0; differential.len()];

        if differential.is_empty() {
            return cumulative;
        }

        // Cumulative DVH: volume receiving >= dose
        // Start from the end and accumulate backwards
        let mut running_sum = 0.0;
        for i in (0..differential.len()).rev() {
            running_sum += differential[i];
            cumulative[i] = running_sum;
        }

        cumulative
    }

    /// Trim trailing zeros from histogram (matches numpy.trim_zeros)
    pub fn trim_zeros(histogram: Vec<f64>) -> Vec<f64> {
        let mut end = histogram.len();

        // Find last non-zero element
        while end > 0 && histogram[end - 1] == 0.0 {
            end -= 1;
        }

        if end == 0 {
            // All zeros, return single zero
            vec![0.0]
        } else {
            histogram[0..end].to_vec()
        }
    }

    /// Calculate voxel volume in cc
    pub fn calculate_voxel_volume(
        pixel_spacing_row_mm: f64,
        pixel_spacing_col_mm: f64,
        thickness_mm: f64,
    ) -> f64 {
        // Volume in mm³ converted to cc (cm³)
        (pixel_spacing_row_mm * pixel_spacing_col_mm * thickness_mm) / 1000.0
    }

    /// Scale histogram to volume
    /// Matches Python's rescaling: hist = hist * volume / sum(hist)
    pub fn scale_to_volume(histogram: &mut [f64], total_volume_cc: f64) {
        let sum: f64 = histogram.iter().sum();

        if sum > 0.0 {
            let scale_factor = total_volume_cc / sum;
            for bin in histogram.iter_mut() {
                *bin *= scale_factor;
            }
        }
    }
}
