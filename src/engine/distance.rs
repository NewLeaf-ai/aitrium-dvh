/// Euclidean Distance Transform using Felzenszwalb & Huttenlocher algorithm
///
/// This module implements fast, exact 2D Euclidean distance transforms using
/// the parabola envelope method. The algorithm runs in O(n) time where n is
/// the number of pixels.
///
/// Reference: Felzenszwalb & Huttenlocher (2012), "Distance Transforms of Sampled Functions"
use ndarray::Array2;

/// Compute 2D Euclidean Distance Transform
///
/// Returns the distance (in mm) from each pixel to the nearest feature pixel (True value).
/// Handles anisotropic pixel spacing (different dx and dy).
///
/// # Arguments
/// * `mask` - Binary mask where True pixels are features
/// * `dx_mm` - Pixel spacing in X direction (mm)
/// * `dy_mm` - Pixel spacing in Y direction (mm)
///
/// # Returns
/// Array of distances in mm for each pixel to nearest feature
pub fn euclidean_distance_transform(mask: &Array2<bool>, dx_mm: f64, dy_mm: f64) -> Array2<f64> {
    let (height, width) = mask.dim();

    // Initialize with squared distances: 0 for feature pixels, infinity for non-features
    let mut distance_sq = Array2::from_elem((height, width), f64::INFINITY);

    for ((i, j), &is_feature) in mask.indexed_iter() {
        if is_feature {
            distance_sq[[i, j]] = 0.0;
        }
    }

    // Pass 1: Process rows (horizontal direction)
    for i in 0..height {
        let mut row: Vec<f64> = distance_sq.row(i).iter().copied().collect();
        edt_1d_pass(&mut row, dx_mm);

        for (j, &val) in row.iter().enumerate() {
            distance_sq[[i, j]] = val;
        }
    }

    // Pass 2: Process columns (vertical direction)
    for j in 0..width {
        let mut col: Vec<f64> = distance_sq.column(j).iter().copied().collect();
        edt_1d_pass(&mut col, dy_mm);

        for (i, &val) in col.iter().enumerate() {
            distance_sq[[i, j]] = val;
        }
    }

    // Take square root to get actual distances
    distance_sq.mapv(|x| x.sqrt())
}

/// 1D Euclidean Distance Transform using parabola envelope method
///
/// This processes a single row or column of squared distances.
/// The algorithm maintains the lower envelope of parabolas centered at feature points.
///
/// # Arguments
/// * `distances` - Array of squared distances (modified in place)
/// * `spacing_mm` - Pixel spacing in mm for this dimension
fn edt_1d_pass(distances: &mut [f64], spacing_mm: f64) {
    let n = distances.len();
    if n == 0 {
        return;
    }

    // Convert to squared distances with spacing
    let f: Vec<f64> = distances
        .iter()
        .enumerate()
        .map(|(i, &d)| d + (i as f64 * spacing_mm).powi(2))
        .collect();

    // Parabola envelope algorithm
    let mut v = vec![0usize]; // Indices of parabolas in lower envelope
    let mut z = vec![f64::NEG_INFINITY, f64::INFINITY]; // Intersection x-coordinates

    // Build lower envelope
    for q in 1..n {
        let mut k = v.len();

        // Remove parabolas that are no longer in the lower envelope
        loop {
            let i = v[k - 1];

            // Calculate intersection of parabola at i and parabola at q
            let numerator =
                (f[q] + (q as f64 * spacing_mm).powi(2)) - (f[i] + (i as f64 * spacing_mm).powi(2));
            let denominator = 2.0 * spacing_mm * (q as f64 - i as f64);

            let s = if denominator.abs() > 1e-10 {
                numerator / denominator
            } else {
                f64::INFINITY
            };

            if s <= z[k - 1] {
                // Parabola i is dominated, remove it
                k -= 1;
                v.pop();
                z.pop();

                if k == 1 {
                    break;
                }
            } else {
                break;
            }
        }

        // Add new parabola to envelope
        v.push(q);

        // Calculate intersection with previous parabola
        let prev_idx = v[v.len() - 2];
        let numerator = (f[q] + (q as f64 * spacing_mm).powi(2))
            - (f[prev_idx] + (prev_idx as f64 * spacing_mm).powi(2));
        let denominator = 2.0 * spacing_mm * (q as f64 - prev_idx as f64);

        let intersection = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            f64::INFINITY
        };

        z.push(intersection);
        z.push(f64::INFINITY);
    }

    // Fill distances from lower envelope
    let mut k = 0;
    for q in 0..n {
        let q_pos = q as f64 * spacing_mm;

        // Find which parabola q belongs to
        while z[k + 1] < q_pos {
            k += 1;
        }

        let i = v[k];
        let dist_sq = (q as f64 - i as f64).powi(2) * spacing_mm.powi(2) + f[i];
        distances[q] = dist_sq;
    }
}

/// Compute signed distance field for a binary mask
///
/// For each pixel:
/// - If inside mask: negative distance to nearest boundary
/// - If outside mask: positive distance to nearest boundary
/// - On boundary: approximately 0
///
/// # Arguments
/// * `mask` - Binary mask (True = inside structure)
/// * `dx_mm` - Pixel spacing in X direction (mm)
/// * `dy_mm` - Pixel spacing in Y direction (mm)
///
/// # Returns
/// Signed distance field in mm
pub fn signed_distance_field(mask: &Array2<bool>, dx_mm: f64, dy_mm: f64) -> Array2<f64> {
    // Compute EDT for interior (distance from inside to boundary)
    let edt_interior = euclidean_distance_transform(mask, dx_mm, dy_mm);

    // Compute EDT for exterior (distance from outside to boundary)
    let mask_inverted = mask.mapv(|x| !x);
    let edt_exterior = euclidean_distance_transform(&mask_inverted, dx_mm, dy_mm);

    // Combine into signed distance field
    let mut sdf = Array2::zeros(mask.dim());

    for ((i, j), &is_inside) in mask.indexed_iter() {
        if is_inside {
            // Inside: negative distance
            sdf[[i, j]] = -edt_interior[[i, j]];
        } else {
            // Outside: positive distance
            sdf[[i, j]] = edt_exterior[[i, j]];
        }
    }

    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_edt_single_pixel() {
        // 5x5 grid with center pixel as feature
        let mut mask = Array2::from_elem((5, 5), false);
        mask[[2, 2]] = true;

        let edt = euclidean_distance_transform(&mask, 1.0, 1.0);

        // Center should be 0
        assert_relative_eq!(edt[[2, 2]], 0.0, epsilon = 0.01);

        // Adjacent pixels should be 1mm
        assert_relative_eq!(edt[[2, 1]], 1.0, epsilon = 0.01);
        assert_relative_eq!(edt[[2, 3]], 1.0, epsilon = 0.01);
        assert_relative_eq!(edt[[1, 2]], 1.0, epsilon = 0.01);
        assert_relative_eq!(edt[[3, 2]], 1.0, epsilon = 0.01);

        // Diagonal should be sqrt(2) ≈ 1.414mm
        assert_relative_eq!(edt[[1, 1]], 1.414, epsilon = 0.01);
        assert_relative_eq!(edt[[3, 3]], 1.414, epsilon = 0.01);
    }

    #[test]
    fn test_edt_horizontal_line() {
        // 5x10 grid with horizontal line at row 2
        let mut mask = Array2::from_elem((5, 10), false);
        for j in 0..10 {
            mask[[2, j]] = true;
        }

        let edt = euclidean_distance_transform(&mask, 1.0, 1.0);

        // Points on line should be 0
        for j in 0..10 {
            assert_relative_eq!(edt[[2, j]], 0.0, epsilon = 0.01);
        }

        // Point one row above should be 1mm
        assert_relative_eq!(edt[[1, 5]], 1.0, epsilon = 0.01);

        // Point two rows above should be 2mm
        assert_relative_eq!(edt[[0, 5]], 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_edt_anisotropic() {
        // Test with different pixel spacings
        let mut mask = Array2::from_elem((5, 10), false);
        mask[[2, 5]] = true;

        // dx = 1mm, dy = 2mm
        let edt = euclidean_distance_transform(&mask, 1.0, 2.0);

        // One row up (2mm in y direction)
        assert_relative_eq!(edt[[1, 5]], 2.0, epsilon = 0.01);

        // One column right (1mm in x direction)
        assert_relative_eq!(edt[[2, 6]], 1.0, epsilon = 0.01);

        // Diagonal: sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236mm
        assert_relative_eq!(edt[[1, 6]], 2.236, epsilon = 0.01);
    }

    #[test]
    fn test_signed_distance_field() {
        // 7x7 grid with 3x3 solid center
        let mut mask = Array2::from_elem((7, 7), false);
        for i in 2..5 {
            for j in 2..5 {
                mask[[i, j]] = true;
            }
        }

        let sdf = signed_distance_field(&mask, 1.0, 1.0);

        // Center point should be negative (inside)
        assert!(sdf[[3, 3]] < 0.0, "Center should be inside (negative)");

        // Far outside point should be positive
        assert!(sdf[[0, 0]] > 0.0, "Corner should be outside (positive)");

        // Edge points should be near zero
        // Point at (2, 1) is just outside the left edge
        assert_relative_eq!(sdf[[2, 1]].abs(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_sdf_symmetry() {
        // Create symmetric structure
        let mut mask = Array2::from_elem((9, 9), false);
        for i in 3..6 {
            for j in 3..6 {
                mask[[i, j]] = true;
            }
        }

        let sdf = signed_distance_field(&mask, 1.0, 1.0);

        // Check symmetry: points equidistant from center should have same magnitude
        let center_dist = sdf[[4, 4]].abs();
        let corner_dist = sdf[[3, 3]].abs();

        // All corners of the square should have similar distances
        assert_relative_eq!(sdf[[3, 3]].abs(), corner_dist, epsilon = 0.1);
        assert_relative_eq!(sdf[[3, 5]].abs(), corner_dist, epsilon = 0.1);
        assert_relative_eq!(sdf[[5, 3]].abs(), corner_dist, epsilon = 0.1);
        assert_relative_eq!(sdf[[5, 5]].abs(), corner_dist, epsilon = 0.1);
    }
}
