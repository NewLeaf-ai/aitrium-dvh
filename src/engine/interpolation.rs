use ndarray::{Array2, ArrayView2};

/// Bilinear interpolation for 2D dose grids
/// Matches scikit-image rescale with order=1, mode='symmetric'
pub fn rescale_2d(input: ArrayView2<f32>, scale_row: f64, scale_col: f64) -> Array2<f32> {
    let (in_rows, in_cols) = input.dim();
    let out_rows = (in_rows as f64 * scale_row).round() as usize;
    let out_cols = (in_cols as f64 * scale_col).round() as usize;

    let mut output = Array2::<f32>::zeros((out_rows, out_cols));

    for out_r in 0..out_rows {
        for out_c in 0..out_cols {
            // Map output coordinate to input coordinate
            let in_r = out_r as f64 / scale_row;
            let in_c = out_c as f64 / scale_col;

            // Get the four surrounding pixels
            let r0 = in_r.floor() as isize;
            let r1 = r0 + 1;
            let c0 = in_c.floor() as isize;
            let c1 = c0 + 1;

            // Calculate interpolation weights
            let wr1 = in_r - r0 as f64;
            let wr0 = 1.0 - wr1;
            let wc1 = in_c - c0 as f64;
            let wc0 = 1.0 - wc1;

            // Handle boundaries with symmetric padding (reflection)
            let get_pixel = |r: isize, c: isize| -> f32 {
                let r_idx = if r < 0 {
                    (-r - 1) as usize
                } else if r >= in_rows as isize {
                    (2 * in_rows as isize - r - 1) as usize
                } else {
                    r as usize
                };

                let c_idx = if c < 0 {
                    (-c - 1) as usize
                } else if c >= in_cols as isize {
                    (2 * in_cols as isize - c - 1) as usize
                } else {
                    c as usize
                };

                // Ensure indices are within bounds after reflection
                let r_idx = r_idx.min(in_rows - 1);
                let c_idx = c_idx.min(in_cols - 1);

                input[[r_idx, c_idx]]
            };

            // Bilinear interpolation
            let p00 = get_pixel(r0, c0);
            let p01 = get_pixel(r0, c1);
            let p10 = get_pixel(r1, c0);
            let p11 = get_pixel(r1, c1);

            let value = (p00 as f64 * wr0 * wc0
                + p01 as f64 * wr0 * wc1
                + p10 as f64 * wr1 * wc0
                + p11 as f64 * wr1 * wc1) as f32;

            output[[out_r, out_c]] = value;
        }
    }

    output
}

/// Check if a resolution is valid (must be a power-of-2 factor of original)
pub fn is_valid_interpolation_resolution(original_spacing: f64, new_spacing: f64) -> bool {
    if new_spacing > original_spacing {
        return false; // Can't upsample
    }

    let ratio = original_spacing / new_spacing;

    // Check if ratio is a power of 2
    if ratio < 1.0 {
        return false;
    }

    // Allow some tolerance for floating point
    let log2_ratio = ratio.log2();
    let rounded = log2_ratio.round();

    (log2_ratio - rounded).abs() < 0.01
}

/// Calculate the interpolation scale factors
pub fn calculate_interpolation_scale(
    original_pixel_spacing: (f64, f64),    // (row, col) in mm
    target_resolution: Option<(f64, f64)>, // (row, col) in mm
) -> Option<(f64, f64)> {
    let (orig_row_spacing, orig_col_spacing) = original_pixel_spacing;

    if let Some((target_row, target_col)) = target_resolution {
        // Validate that target resolution is valid
        if !is_valid_interpolation_resolution(orig_row_spacing, target_row) {
            eprintln!(
                "Warning: Invalid row interpolation resolution: {} -> {}",
                orig_row_spacing, target_row
            );
            return None;
        }
        if !is_valid_interpolation_resolution(orig_col_spacing, target_col) {
            eprintln!(
                "Warning: Invalid col interpolation resolution: {} -> {}",
                orig_col_spacing, target_col
            );
            return None;
        }

        // Calculate scale factors (> 1 means upsampling)
        let row_scale = orig_row_spacing / target_row;
        let col_scale = orig_col_spacing / target_col;

        Some((row_scale, col_scale))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_rescale_2x2_to_4x4() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let output = rescale_2d(input.view(), 2.0, 2.0);

        // Should interpolate to 4x4
        assert_eq!(output.dim(), (4, 4));

        // Corner values should be preserved
        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[0, 3]], 2.0);
        assert_eq!(output[[3, 0]], 3.0);
        assert_eq!(output[[3, 3]], 4.0);

        // Check some interpolated values
        assert!((output[[1, 1]] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_is_valid_interpolation_resolution() {
        // Valid: 3.0 -> 1.5 (3.0 / 2^1)
        assert!(is_valid_interpolation_resolution(3.0, 1.5));

        // Valid: 3.0 -> 0.75 (3.0 / 2^2)
        assert!(is_valid_interpolation_resolution(3.0, 0.75));

        // Invalid: 3.0 -> 1.0 (not power of 2)
        assert!(!is_valid_interpolation_resolution(3.0, 1.0));

        // Invalid: 3.0 -> 4.0 (upsampling)
        assert!(!is_valid_interpolation_resolution(3.0, 4.0));
    }
}
