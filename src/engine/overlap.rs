/// Structure overlap calculation module
///
/// This module computes volumetric overlaps between ROI pairs using the same
/// coordinate system and masking logic as DVH calculations, ensuring geometric consistency.
use crate::dicom_parser::{parse_rtdose, parse_rtstruct};
use crate::engine::z_interpolation::ZInterpolator;
use crate::geometry::matplotlib_poly::MatplotlibPolygon;
use crate::types::{Contour, DoseGrid, DvhError, OrderedFloat, Roi};
use ndarray::Array2;
use std::collections::BTreeMap;
use std::path::Path;

/// Options for overlap calculation
#[derive(Debug, Clone)]
pub struct OverlapOptions {
    /// Number of segments to interpolate between structure planes (0 = off)
    pub interpolation_segments_between_planes: u32,
    /// Optional XY grid resolution in mm (row, col). None uses native dose grid.
    pub interpolation_resolution_mm: Option<(f64, f64)>,
}

impl Default for OverlapOptions {
    fn default() -> Self {
        Self {
            interpolation_segments_between_planes: 0,
            interpolation_resolution_mm: None,
        }
    }
}

/// Result of overlap calculation between two ROIs
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// Overlap volume in cc
    pub volume_cc: f64,
    /// Overlap as percentage of ROI A
    pub percent_a: Option<f64>,
    /// Overlap as percentage of ROI B
    pub percent_b: Option<f64>,
    /// Total volume of ROI A in cc
    pub volume_a_cc: f64,
    /// Total volume of ROI B in cc
    pub volume_b_cc: f64,
}

/// Compute overlap between two ROIs by name
///
/// # Arguments
/// * `rtstruct_path` - Path to RTSTRUCT DICOM file
/// * `rtdose_path` - Path to RTDOSE DICOM file
/// * `roi_name_a` - Name of first ROI
/// * `roi_name_b` - Name of second ROI
/// * `options` - Calculation options
///
/// # Returns
/// * `Ok(Some(result))` - Overlap calculated successfully
/// * `Ok(None)` - One or both ROIs not found
/// * `Err` - DICOM parsing or calculation error
pub fn compute_overlap_by_name(
    rtstruct_path: &Path,
    rtdose_path: &Path,
    roi_name_a: &str,
    roi_name_b: &str,
    options: &OverlapOptions,
) -> Result<Option<OverlapResult>, DvhError> {
    // Parse DICOM files
    let rois = parse_rtstruct(rtstruct_path)?;
    let dose_grid = parse_rtdose(rtdose_path)?;

    // Find ROIs by name (case-insensitive)
    let roi_a = rois
        .iter()
        .find(|r| r.name.eq_ignore_ascii_case(roi_name_a));
    let roi_b = rois
        .iter()
        .find(|r| r.name.eq_ignore_ascii_case(roi_name_b));

    match (roi_a, roi_b) {
        (Some(a), Some(b)) => {
            let result = compute_overlap(a, b, &dose_grid, options)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Core overlap calculation between two ROIs
fn compute_overlap(
    roi_a: &Roi,
    roi_b: &Roi,
    dose_grid: &DoseGrid,
    options: &OverlapOptions,
) -> Result<OverlapResult, DvhError> {
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
        return Ok(OverlapResult {
            volume_cc: 0.0,
            percent_a: Some(0.0),
            percent_b: Some(0.0),
            volume_a_cc: 0.0,
            volume_b_cc: 0.0,
        });
    }

    // Accumulate volumes across planes
    let mut volume_a_mm3 = 0.0;
    let mut volume_b_mm3 = 0.0;
    let mut overlap_mm3 = 0.0;

    for z_pos in &common_z_positions {
        let contours_a = &planes_a[z_pos];
        let contours_b = &planes_b[z_pos];

        // Build binary masks using XOR for multiple contours
        let mask_a = build_combined_mask(contours_a, &col_lut, &row_lut, dose_grid.x_lut_index)?;
        let mask_b = build_combined_mask(contours_b, &col_lut, &row_lut, dose_grid.x_lut_index)?;

        // Use minimum thickness to avoid double-counting
        let dz_mm = thickness_a.min(thickness_b);
        let voxel_volume_mm3 = dx_mm * dy_mm * dz_mm;

        // Count voxels in each mask
        let n_a = mask_a.iter().filter(|&&x| x).count() as f64;
        let n_b = mask_b.iter().filter(|&&x| x).count() as f64;

        // Compute intersection mask
        let n_intersection = mask_a
            .iter()
            .zip(mask_b.iter())
            .filter(|(&a, &b)| a && b)
            .count() as f64;

        // Accumulate volumes
        volume_a_mm3 += n_a * voxel_volume_mm3;
        volume_b_mm3 += n_b * voxel_volume_mm3;
        overlap_mm3 += n_intersection * voxel_volume_mm3;
    }

    // Convert to cc (1 cc = 1000 mm³)
    let volume_a_cc = volume_a_mm3 / 1000.0;
    let volume_b_cc = volume_b_mm3 / 1000.0;
    let overlap_cc = overlap_mm3 / 1000.0;

    // Calculate percentages
    let percent_a = if volume_a_cc > 0.0 {
        Some((overlap_cc / volume_a_cc) * 100.0)
    } else {
        None
    };

    let percent_b = if volume_b_cc > 0.0 {
        Some((overlap_cc / volume_b_cc) * 100.0)
    } else {
        None
    };

    Ok(OverlapResult {
        volume_cc: overlap_cc,
        percent_a,
        percent_b,
        volume_a_cc,
        volume_b_cc,
    })
}

/// Build a combined mask from multiple contours using XOR (even-odd rule)
///
/// Multiple contours at the same Z position are combined via XOR to handle
/// cavities and holes, matching DICOM and DVH behavior.
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
    fn test_overlap_options_default() {
        let opts = OverlapOptions::default();
        assert_eq!(opts.interpolation_segments_between_planes, 0);
        assert!(opts.interpolation_resolution_mm.is_none());
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

    #[test]
    fn test_build_combined_mask_with_hole() {
        // Outer square and inner hole (XOR should create ring)
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];

        let contours = vec![
            Contour {
                points: outer,
                contour_type: crate::types::ContourType::External,
            },
            Contour {
                points: hole,
                contour_type: crate::types::ContourType::Cavity,
            },
        ];

        let col_lut: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let row_lut: Vec<f64> = (0..15).map(|i| i as f64).collect();

        // Use x_lut_index = 0 for test (col_lut = X, row_lut = Y)
        let mask = build_combined_mask(&contours, &col_lut, &row_lut, 0).unwrap();

        // Point in ring (between outer and hole) should be true
        assert!(mask[[1, 1]]); // In outer, not in hole

        // Point in hole should be false (XOR'd twice)
        assert!(!mask[[5, 5]]); // In both outer and hole

        // Point outside outer should be false
        assert!(!mask[[12, 12]]); // Outside both
    }
}
