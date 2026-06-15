/// Euclidean Distance Transform using Felzenszwalb & Huttenlocher algorithm
///
/// This module implements fast, exact 2D Euclidean distance transforms using
/// the parabola envelope method. The algorithm runs in O(n) time where n is
/// the number of pixels.
///
/// Reference: Felzenszwalb & Huttenlocher (2012), "Distance Transforms of Sampled Functions"
use ndarray::{Array2, Array3};

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

/// Compute 3D Euclidean Distance Transform
///
/// Returns the distance (in mm) from each voxel to the nearest feature voxel (True value).
/// Handles anisotropic voxel spacing (different dx/dy/dz).
pub fn euclidean_distance_transform_3d(
    mask: &Array3<bool>,
    dx_mm: f64,
    dy_mm: f64,
    dz_mm: f64,
) -> Array3<f64> {
    let (depth, height, width) = mask.dim();

    let mut distance_sq = Array3::from_elem((depth, height, width), f64::INFINITY);
    for ((k, i, j), &is_feature) in mask.indexed_iter() {
        if is_feature {
            distance_sq[[k, i, j]] = 0.0;
        }
    }

    // Pass 1: X axis (columns)
    for k in 0..depth {
        for i in 0..height {
            let mut line: Vec<f64> = (0..width).map(|j| distance_sq[[k, i, j]]).collect();
            edt_1d_pass(&mut line, dx_mm);
            for (j, &val) in line.iter().enumerate() {
                distance_sq[[k, i, j]] = val;
            }
        }
    }

    // Pass 2: Y axis (rows)
    for k in 0..depth {
        for j in 0..width {
            let mut line: Vec<f64> = (0..height).map(|i| distance_sq[[k, i, j]]).collect();
            edt_1d_pass(&mut line, dy_mm);
            for (i, &val) in line.iter().enumerate() {
                distance_sq[[k, i, j]] = val;
            }
        }
    }

    // Pass 3: Z axis (frames)
    for i in 0..height {
        for j in 0..width {
            let mut line: Vec<f64> = (0..depth).map(|k| distance_sq[[k, i, j]]).collect();
            edt_1d_pass(&mut line, dz_mm);
            for (k, &val) in line.iter().enumerate() {
                distance_sq[[k, i, j]] = val;
            }
        }
    }

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

    // Squared distance input (0 for features, +inf for non-features in pass 1).
    let f = distances.to_vec();

    // No finite sites in this 1D line => remain +inf.
    let Some(first_finite_idx) = f.iter().position(|d| d.is_finite()) else {
        return;
    };

    // Felzenszwalb & Huttenlocher lower-envelope construction.
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;

    v[0] = first_finite_idx;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for q in (first_finite_idx + 1)..n {
        if !f[q].is_finite() {
            continue;
        }

        let q_pos = q as f64 * spacing_mm;
        let mut s;

        loop {
            let i = v[k];
            let i_pos = i as f64 * spacing_mm;
            let numerator = (f[q] + q_pos * q_pos) - (f[i] + i_pos * i_pos);
            let denominator = 2.0 * (q_pos - i_pos);

            s = if denominator.abs() > 1e-12 {
                numerator / denominator
            } else {
                f64::INFINITY
            };

            if s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                break;
            }
        }

        // q does not improve envelope for any x.
        if s <= z[k] {
            continue;
        }

        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }

    // Evaluate lower envelope at each sample position.
    k = 0;
    for (q, out) in distances.iter_mut().enumerate() {
        let q_pos = q as f64 * spacing_mm;
        while z[k + 1] < q_pos {
            k += 1;
        }

        let i = v[k];
        let i_pos = i as f64 * spacing_mm;
        let delta = q_pos - i_pos;
        *out = delta * delta + f[i];
    }

    // Numerical guard for tiny negative values from floating-point roundoff.
    for d in distances.iter_mut() {
        if *d < 0.0 && (*d).abs() < 1e-12 {
            *d = 0.0;
        }
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
    // Distance to nearest inside voxel (0 inside, positive outside)
    let edt_to_inside = euclidean_distance_transform(mask, dx_mm, dy_mm);

    // Distance to nearest outside voxel (0 outside, positive inside)
    let mask_inverted = mask.mapv(|x| !x);
    let edt_to_outside = euclidean_distance_transform(&mask_inverted, dx_mm, dy_mm);

    // Combine into signed distance field
    let mut sdf = Array2::zeros(mask.dim());

    for ((i, j), &is_inside) in mask.indexed_iter() {
        if is_inside {
            // Inside: negative distance to boundary
            sdf[[i, j]] = -edt_to_outside[[i, j]];
        } else {
            // Outside: positive distance to boundary
            sdf[[i, j]] = edt_to_inside[[i, j]];
        }
    }

    sdf
}

/// Compute signed distance field for a 3D binary mask.
///
/// Convention:
/// - inside voxels: negative distance to nearest outside voxel
/// - outside voxels: positive distance to nearest inside voxel
pub fn signed_distance_field_3d(
    mask: &Array3<bool>,
    dx_mm: f64,
    dy_mm: f64,
    dz_mm: f64,
) -> Array3<f64> {
    let edt_to_inside = euclidean_distance_transform_3d(mask, dx_mm, dy_mm, dz_mm);
    let mask_inverted = mask.mapv(|x| !x);
    let edt_to_outside = euclidean_distance_transform_3d(&mask_inverted, dx_mm, dy_mm, dz_mm);

    let mut sdf = Array3::zeros(mask.dim());
    for ((k, i, j), &is_inside) in mask.indexed_iter() {
        if is_inside {
            sdf[[k, i, j]] = -edt_to_outside[[k, i, j]];
        } else {
            sdf[[k, i, j]] = edt_to_inside[[k, i, j]];
        }
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

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

    #[test]
    fn test_edt_all_true_is_zero() {
        let mask = Array2::from_elem((5, 7), true);
        let edt = euclidean_distance_transform(&mask, 1.0, 1.0);
        assert!(edt.iter().all(|d| d.is_finite() && *d == 0.0));
    }

    #[test]
    fn test_edt_all_false_is_infinite() {
        let mask = Array2::from_elem((5, 7), false);
        let edt = euclidean_distance_transform(&mask, 1.0, 1.0);
        assert!(edt.iter().all(|d| d.is_infinite() && d.is_sign_positive()));
    }

    #[test]
    fn test_sdf_is_finite_for_mixed_mask() {
        let mut mask = Array2::from_elem((9, 9), false);
        for i in 2..7 {
            for j in 2..7 {
                mask[[i, j]] = true;
            }
        }

        let sdf = signed_distance_field(&mask, 1.0, 1.0);
        assert!(sdf.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn test_sdf_all_true_is_negative_infinite() {
        let mask = Array2::from_elem((5, 7), true);
        let sdf = signed_distance_field(&mask, 1.0, 1.0);
        assert!(sdf.iter().all(|d| d.is_infinite() && d.is_sign_negative()));
    }

    #[test]
    fn test_sdf_all_false_is_positive_infinite() {
        let mask = Array2::from_elem((5, 7), false);
        let sdf = signed_distance_field(&mask, 1.0, 1.0);
        assert!(sdf.iter().all(|d| d.is_infinite() && d.is_sign_positive()));
    }

    #[test]
    fn test_edt_3d_simple() {
        let mut mask = Array3::from_elem((5, 5, 5), false);
        mask[[2, 2, 2]] = true;
        let edt = euclidean_distance_transform_3d(&mask, 1.0, 1.0, 1.0);
        assert_relative_eq!(edt[[2, 2, 2]], 0.0, epsilon = 0.01);
        assert_relative_eq!(edt[[2, 2, 3]], 1.0, epsilon = 0.01);
        assert_relative_eq!(edt[[2, 3, 2]], 1.0, epsilon = 0.01);
        assert_relative_eq!(edt[[3, 2, 2]], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_sdf_3d_mixed_signs() {
        let mut mask = Array3::from_elem((7, 7, 7), false);
        for k in 2..5 {
            for i in 2..5 {
                for j in 2..5 {
                    mask[[k, i, j]] = true;
                }
            }
        }
        let sdf = signed_distance_field_3d(&mask, 1.0, 1.0, 1.0);
        assert!(sdf[[3, 3, 3]] < 0.0);
        assert!(sdf[[0, 0, 0]] > 0.0);
        assert!(sdf.iter().all(|d| d.is_finite()));
    }
}
