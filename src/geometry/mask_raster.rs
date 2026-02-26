use ndarray::Array2;

/// Alternative scanline rasterizer for performance optimization
/// This will be used in production after parity is confirmed
pub struct MaskRasterizer;

impl MaskRasterizer {
    /// Scanline rasterization with direct histogram accumulation
    /// Avoids creating boolean masks for better performance
    pub fn scanline_accumulate<F>(
        contours: &[Vec<[f64; 2]>],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
        dose_plane: &Array2<f32>,
        mut accumulator: F,
    ) where
        F: FnMut(f32, usize, usize),
    {
        // For each row, find intersections and accumulate dose values
        for (row_idx, &y_coord) in row_lut.iter().enumerate() {
            let mut intervals = Vec::new();

            // Collect all intersections for this scanline
            for contour in contours {
                let contour_intervals =
                    Self::get_scanline_intersections(contour, y_coord, x_lut_index);
                intervals.extend(contour_intervals);
            }

            // Sort intervals and apply even-odd rule with XOR
            intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Process intervals with even-odd fill
            let mut inside = false;
            let mut start_x = 0.0;

            for x in intervals {
                if inside {
                    // End of fill interval
                    let end_x = x;

                    // Find column indices for this interval
                    let start_col = Self::find_column_index(start_x, col_lut, x_lut_index);
                    let end_col = Self::find_column_index(end_x, col_lut, x_lut_index);

                    // Accumulate dose values in this interval
                    for col_idx in start_col..=end_col {
                        if col_idx < dose_plane.ncols() {
                            let dose_value = dose_plane[[row_idx, col_idx]];
                            accumulator(dose_value, row_idx, col_idx);
                        }
                    }
                }
                inside = !inside;
                start_x = x;
            }
        }
    }

    /// Get scanline intersections for a contour at a given y coordinate
    fn get_scanline_intersections(contour: &[[f64; 2]], y: f64, x_lut_index: u8) -> Vec<f64> {
        let mut intersections = Vec::new();
        let n = contour.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let p1 = contour[i];
            let p2 = contour[j];

            let (y1, y2) = if x_lut_index == 0 {
                (p1[1], p2[1])
            } else {
                (p1[0], p2[0])
            };

            // Check if edge crosses scanline
            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                // Calculate intersection x coordinate
                let x1 = if x_lut_index == 0 { p1[0] } else { p1[1] };
                let x2 = if x_lut_index == 0 { p2[0] } else { p2[1] };

                let t = (y - y1) / (y2 - y1);
                let x = x1 + t * (x2 - x1);

                intersections.push(x);
            }
        }

        intersections
    }

    /// Find column index for a given x coordinate
    fn find_column_index(x: f64, col_lut: &[f64], _x_lut_index: u8) -> usize {
        // Binary search for closest column
        match col_lut.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx >= col_lut.len() {
                    col_lut.len() - 1
                } else {
                    // Return closest index
                    if (x - col_lut[idx - 1]).abs() < (col_lut[idx] - x).abs() {
                        idx - 1
                    } else {
                        idx
                    }
                }
            }
        }
    }
}
