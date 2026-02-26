/// Structure margin metrics calculation module
///
/// This module computes signed distances from one ROI to another with statistical
/// summaries (min, percentiles, mean) and coverage metrics.
use crate::dicom_parser::{parse_rtdose, parse_rtstruct};
use crate::engine::distance::signed_distance_field;
use crate::engine::orientation::{
    calculate_center_of_mass_2d, direction_to_vector, is_point_in_direction, PatientPosition,
};
use crate::engine::z_interpolation::ZInterpolator;
use crate::geometry::matplotlib_poly::MatplotlibPolygon;
use crate::types::{Contour, DoseGrid, DvhError, MarginDirection, OrderedFloat, Roi};
use ndarray::Array2;
use std::path::Path;

/// Options for margin calculation
#[derive(Debug, Clone)]
pub struct MarginOptions {
    /// Number of segments to interpolate between structure planes (0 = off)
    pub interpolation_segments_between_planes: u32,
    /// Optional XY grid resolution in mm (row, col). None uses native dose grid.
    pub interpolation_resolution_mm: Option<(f64, f64)>,
    /// Distance thresholds (in mm) for coverage metrics
    pub coverage_thresholds_mm: Vec<f64>,
    /// Optional direction for margin calculation (None = uniform)
    pub direction: Option<MarginDirection>,
}

impl Default for MarginOptions {
    fn default() -> Self {
        Self {
            interpolation_segments_between_planes: 0,
            interpolation_resolution_mm: None,
            coverage_thresholds_mm: vec![3.0, 5.0, 7.0],
            direction: None,
        }
    }
}

/// Result of margin calculation from ROI A to ROI B
#[derive(Debug, Clone)]
pub struct MarginResult {
    /// Minimum distance in mm (most critical point, negative = overlap)
    pub min_mm: f64,
    /// 5th percentile distance in mm
    pub p05_mm: f64,
    /// Median (50th percentile) distance in mm
    pub p50_mm: f64,
    /// 95th percentile distance in mm
    pub p95_mm: f64,
    /// Volume-weighted mean distance in mm
    pub mean_mm: f64,
    /// Coverage metrics: Vec of (threshold_mm, percent_within)
    pub coverage_within_thresholds: Vec<(f64, f64)>,
}

/// Compute margin from ROI A to ROI B by name
///
/// This is a directed measurement: distances from A → B
/// (not symmetric, A→B ≠ B→A)
///
/// # Arguments
/// * `rtstruct_path` - Path to RTSTRUCT DICOM file
/// * `rtdose_path` - Path to RTDOSE DICOM file
/// * `roi_from` - Name of source ROI (A)
/// * `roi_to` - Name of target ROI (B)
/// * `options` - Calculation options
///
/// # Returns
/// * `Ok(Some(result))` - Margin calculated successfully
/// * `Ok(None)` - One or both ROIs not found
/// * `Err` - DICOM parsing or calculation error
pub fn compute_margin_directed(
    rtstruct_path: &Path,
    rtdose_path: &Path,
    roi_from: &str,
    roi_to: &str,
    options: &MarginOptions,
) -> Result<Option<MarginResult>, DvhError> {
    // Parse DICOM files
    let rois = parse_rtstruct(rtstruct_path)?;
    let dose_grid = parse_rtdose(rtdose_path)?;

    // Find ROIs by name (case-insensitive)
    let roi_a = rois.iter().find(|r| r.name.eq_ignore_ascii_case(roi_from));
    let roi_b = rois.iter().find(|r| r.name.eq_ignore_ascii_case(roi_to));

    match (roi_a, roi_b) {
        (Some(a), Some(b)) => {
            let result = compute_margin(a, b, &dose_grid, options)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Core margin calculation between two ROIs
fn compute_margin(
    roi_a: &Roi,
    roi_b: &Roi,
    dose_grid: &DoseGrid,
    options: &MarginOptions,
) -> Result<MarginResult, DvhError> {
    // Calculate LUTs from dose grid
    let (col_lut, row_lut) = calculate_luts(dose_grid);

    // Calculate voxel dimensions
    let dx_mm = mean_diff(&col_lut);
    let dy_mm = mean_diff(&row_lut);

    // Apply Z interpolation if requested
    let planes_a = if options.interpolation_segments_between_planes > 0 {
        ZInterpolator::interpolate_planes(
            &roi_a.planes,
            options.interpolation_segments_between_planes,
        )
    } else {
        roi_a.planes.clone()
    };

    let planes_b = if options.interpolation_segments_between_planes > 0 {
        ZInterpolator::interpolate_planes(
            &roi_b.planes,
            options.interpolation_segments_between_planes,
        )
    } else {
        roi_b.planes.clone()
    };

    // Calculate adjusted slice thickness
    let thickness_a = ZInterpolator::adjusted_thickness(
        roi_a.thickness_mm,
        options.interpolation_segments_between_planes,
    );
    let thickness_b = ZInterpolator::adjusted_thickness(
        roi_b.thickness_mm,
        options.interpolation_segments_between_planes,
    );

    // Find common Z positions
    let common_z_positions: Vec<OrderedFloat> = planes_a
        .keys()
        .filter(|z| planes_b.contains_key(z))
        .copied()
        .collect();

    if common_z_positions.is_empty() {
        // No common planes - structures don't overlap spatially
        // Return large positive distances
        return Ok(MarginResult {
            min_mm: f64::INFINITY,
            p05_mm: f64::INFINITY,
            p50_mm: f64::INFINITY,
            p95_mm: f64::INFINITY,
            mean_mm: f64::INFINITY,
            coverage_within_thresholds: options
                .coverage_thresholds_mm
                .iter()
                .map(|&t| (t, 0.0))
                .collect(),
        });
    }

    // Prepare for directional filtering if needed
    let (direction_vector, needs_filtering) = if let Some(direction) = options.direction {
        if direction != MarginDirection::Uniform {
            // Parse patient position
            let patient_pos = dose_grid
                .patient_position
                .as_ref()
                .and_then(|s| PatientPosition::from_dicom_string(s));

            // Log patient position for debugging
            eprintln!(
                "Patient position from dose grid: {:?}, parsed as: {:?}",
                dose_grid.patient_position, patient_pos
            );

            // Get direction vector
            let vec = direction_to_vector(direction, patient_pos);
            eprintln!("Direction {:?} mapped to vector: {:?}", direction, vec);
            (vec, true)
        } else {
            ([0.0, 0.0, 0.0], false)
        }
    } else {
        ([0.0, 0.0, 0.0], false)
    };

    // Calculate center of mass for structure B if directional filtering is needed
    let mut center_b = [0.0, 0.0, 0.0];
    if needs_filtering {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        let mut total_volume = 0.0;

        for z_pos in &common_z_positions {
            let contours_b = &planes_b[z_pos];
            let mask_b =
                build_combined_mask(contours_b, &col_lut, &row_lut, dose_grid.x_lut_index)?;
            let center_2d = calculate_center_of_mass_2d(&mask_b, &col_lut, &row_lut, z_pos.0);

            // Weight by slice volume
            let slice_volume: f64 = mask_b.iter().filter(|&&v| v).count() as f64;
            if slice_volume > 0.0 {
                sum_x += center_2d[0] * slice_volume;
                sum_y += center_2d[1] * slice_volume;
                sum_z += center_2d[2] * slice_volume;
                total_volume += slice_volume;
            }
        }

        if total_volume > 0.0 {
            center_b = [
                sum_x / total_volume,
                sum_y / total_volume,
                sum_z / total_volume,
            ];
        }
    }

    // Collect weighted distance samples: (distance_mm, voxel_volume_mm3)
    let mut distance_samples: Vec<(f64, f64)> = Vec::new();
    const DIRECTION_TOLERANCE_DEGREES: f64 = 45.0; // 45-degree cone

    let mut total_points_in_a = 0;
    let mut points_filtered_out = 0;

    for z_pos in &common_z_positions {
        let contours_a = &planes_a[z_pos];
        let contours_b = &planes_b[z_pos];

        // Build binary masks using XOR for multiple contours
        let mask_a = build_combined_mask(contours_a, &col_lut, &row_lut, dose_grid.x_lut_index)?;
        let mask_b = build_combined_mask(contours_b, &col_lut, &row_lut, dose_grid.x_lut_index)?;

        // Compute signed distance field for B
        let sdf_b = signed_distance_field(&mask_b, dx_mm, dy_mm);

        // Calculate voxel volume for weighting
        let dz_mm = thickness_a.min(thickness_b);
        let voxel_volume_mm3 = dx_mm * dy_mm * dz_mm;

        // Sample SDF at all A voxels
        for ((i, j), &is_in_a) in mask_a.indexed_iter() {
            if is_in_a {
                total_points_in_a += 1;
                // Check direction filter if needed
                if needs_filtering {
                    let point_a = [col_lut[j], row_lut[i], z_pos.0];
                    if !is_point_in_direction(
                        point_a,
                        center_b,
                        direction_vector,
                        DIRECTION_TOLERANCE_DEGREES,
                    ) {
                        points_filtered_out += 1;
                        continue; // Skip points not in the specified direction
                    }
                }

                let distance_mm = sdf_b[[i, j]];
                distance_samples.push((distance_mm, voxel_volume_mm3));
            }
        }
    }

    if needs_filtering {
        eprintln!(
            "Directional margin filtering: {} of {} points from {} kept (filtered out {} points)",
            distance_samples.len(),
            total_points_in_a,
            roi_a.name,
            points_filtered_out
        );
        eprintln!("Center of {}: {:?}", roi_b.name, center_b);
    }

    if distance_samples.is_empty() {
        // ROI A has no volume in common planes
        return Ok(MarginResult {
            min_mm: f64::INFINITY,
            p05_mm: f64::INFINITY,
            p50_mm: f64::INFINITY,
            p95_mm: f64::INFINITY,
            mean_mm: f64::INFINITY,
            coverage_within_thresholds: options
                .coverage_thresholds_mm
                .iter()
                .map(|&t| (t, 0.0))
                .collect(),
        });
    }

    // Calculate statistics
    let total_weight: f64 = distance_samples.iter().map(|(_, w)| w).sum();

    // Sort by distance for percentile calculation
    distance_samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_mm = distance_samples.first().unwrap().0;

    let mean_mm = distance_samples.iter().map(|(d, w)| d * w).sum::<f64>() / total_weight;

    // Weighted percentiles
    let p05_mm = weighted_percentile(&distance_samples, 5.0);
    let p50_mm = weighted_percentile(&distance_samples, 50.0);
    let p95_mm = weighted_percentile(&distance_samples, 95.0);

    // Coverage metrics: percent of A within each threshold
    let coverage: Vec<(f64, f64)> = options
        .coverage_thresholds_mm
        .iter()
        .map(|&threshold_mm| {
            let volume_within: f64 = distance_samples
                .iter()
                .filter(|(d, _)| *d <= threshold_mm)
                .map(|(_, w)| w)
                .sum();

            let percent_within = (volume_within / total_weight) * 100.0;
            (threshold_mm, percent_within)
        })
        .collect();

    Ok(MarginResult {
        min_mm,
        p05_mm,
        p50_mm,
        p95_mm,
        mean_mm,
        coverage_within_thresholds: coverage,
    })
}

/// Build a combined mask from multiple contours using XOR (even-odd rule)
fn build_combined_mask(
    contours: &[Contour],
    col_lut: &[f64],
    row_lut: &[f64],
    x_lut_index: u8,
) -> Result<Array2<bool>, DvhError> {
    // Convert contours to the format expected by create_plane_mask
    let contour_points: Vec<Vec<[f64; 2]>> = contours.iter().map(|c| c.points.clone()).collect();

    // Use the same plane mask creation as DVH for consistency
    Ok(MatplotlibPolygon::create_plane_mask(
        &contour_points,
        col_lut,
        row_lut,
        x_lut_index,
    ))
}

/// Calculate weighted percentile
///
/// Samples should be pre-sorted by value
fn weighted_percentile(samples: &[(f64, f64)], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let total_weight: f64 = samples.iter().map(|(_, w)| w).sum();
    let target_weight = (percentile / 100.0) * total_weight;

    let mut cumulative_weight = 0.0;
    for (value, weight) in samples {
        cumulative_weight += weight;
        if cumulative_weight >= target_weight {
            return *value;
        }
    }

    samples.last().unwrap().0
}

/// Get LUTs from dose grid - uses pre-calculated LUTs that respect orientation
fn calculate_luts(dose_grid: &DoseGrid) -> (Vec<f64>, Vec<f64>) {
    // Use the pre-calculated LUTs from DoseGrid that already account for
    // ImageOrientationPatient and x_lut_index
    (dose_grid.col_lut.clone(), dose_grid.row_lut.clone())
}

/// Calculate mean difference between consecutive values
fn mean_diff(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let sum: f64 = values.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    sum / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_options_default() {
        let opts = MarginOptions::default();
        assert_eq!(opts.interpolation_segments_between_planes, 0);
        assert!(opts.interpolation_resolution_mm.is_none());
        assert_eq!(opts.coverage_thresholds_mm, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_weighted_percentile() {
        // Simple test with uniform weights
        let samples = vec![
            (1.0, 10.0),
            (2.0, 10.0),
            (3.0, 10.0),
            (4.0, 10.0),
            (5.0, 10.0),
        ];

        assert_eq!(weighted_percentile(&samples, 0.0), 1.0);
        assert_eq!(weighted_percentile(&samples, 50.0), 3.0);
        assert_eq!(weighted_percentile(&samples, 100.0), 5.0);
    }

    #[test]
    fn test_weighted_percentile_with_varying_weights() {
        // Heavy weight on first value
        let samples = vec![
            (1.0, 90.0),  // 90% of weight
            (10.0, 10.0), // 10% of weight
        ];

        // Median should be close to 1.0 since most weight is there
        let median = weighted_percentile(&samples, 50.0);
        assert_eq!(median, 1.0);

        // 95th percentile should reach the second value
        let p95 = weighted_percentile(&samples, 95.0);
        assert_eq!(p95, 10.0);
    }

    #[test]
    fn test_mean_diff() {
        let values = vec![0.0, 2.0, 4.0, 6.0];
        assert_eq!(mean_diff(&values), 2.0);

        let empty: Vec<f64> = vec![];
        assert_eq!(mean_diff(&empty), 0.0);

        let single = vec![1.0];
        assert_eq!(mean_diff(&single), 0.0);
    }
}
