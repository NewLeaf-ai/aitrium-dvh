use crate::dicom_parser::{parse_rtdose, parse_rtstruct};
use crate::engine::interpolation;
use crate::engine::{HistogramCalculator, ZInterpolator};
use crate::geometry::{MatplotlibPolygon, PolygonMask};
use crate::types::{DoseGrid, DvhError, DvhOptions, DvhResult, DvhStats, Roi};
use ndarray::Array2;
use rayon::prelude::*;
use std::path::Path;

/// Helper function to get dose plane at Z position with optional interpolation
fn get_dose_plane_at_z(
    dose_grid: &DoseGrid,
    z_mm: f64,
    _thickness_mm: f64, // Unused now, but kept for API compatibility
    interpolation_resolution: Option<(f64, f64)>, // (row_mm, col_mm)
    calculate_full_volume: bool, // Whether to include planes outside dose grid
) -> Option<Array2<f32>> {
    // Reproduce dicompylercore z_sign logic for dose plane selection
    fn z_sign(ori: &[f64; 6]) -> f64 {
        // Head-first orientations
        const HF: [[f64; 6]; 4] = [
            [1., 0., 0., 0., 1., 0.],   // HFS
            [-1., 0., 0., 0., -1., 0.], // HFP
            [0., -1., 0., 1., 0., 0.],  // HFDL
            [0., 1., 0., -1., 0., 0.],  // HFDR
        ];
        // Feet-first orientations
        const FF: [[f64; 6]; 4] = [
            [0., 1., 0., 1., 0., 0.],   // FFDL
            [0., -1., 0., -1., 0., 0.], // FFDR
            [1., 0., 0., 0., -1., 0.],  // FFP
            [-1., 0., 0., 0., 1., 0.],  // FFS
        ];
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6;
        let eq = |a: &[f64; 6], b: &[f64; 6]| (0..6).all(|i| close(a[i], b[i]));

        if HF.iter().any(|p| eq(ori, p)) {
            1.0
        } else if FF.iter().any(|p| eq(ori, p)) {
            -1.0
        } else {
            1.0 // Default to head-first
        }
    }

    let sign = z_sign(&dose_grid.image_orientation_patient);

    // Find the closest plane and check if interpolation is needed
    let z_positions: Vec<f64> = dose_grid
        .grid_frame_offset_vector_mm
        .iter()
        .map(|&offset| dose_grid.image_position_patient[2] + sign * offset)
        .collect();

    // Determine dose grid Z bounds
    let z_min = z_positions
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);
    let z_max = z_positions
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);

    // Check if Z is outside dose grid bounds
    if z_mm < z_min || z_mm > z_max {
        if calculate_full_volume {
            // Return zero-dose plane for planes outside grid when including full volume
            let dose_shape = dose_grid.dose_3d.get_plane(0)?.dim();
            log::debug!(
                "Z={:.2} outside dose grid [{:.2}, {:.2}], returning zero-dose plane",
                z_mm,
                z_min,
                z_max
            );
            return Some(Array2::zeros(dose_shape));
        } else {
            // Skip planes outside dose grid when not calculating full volume
            log::debug!(
                "Z={:.2} outside dose grid [{:.2}, {:.2}], skipping",
                z_mm,
                z_min,
                z_max
            );
            return None;
        }
    }

    // Find the closest plane
    let mut closest_idx = 0;
    let mut min_distance = f64::MAX;
    for (idx, &z_pos) in z_positions.iter().enumerate() {
        let dist = (z_pos - z_mm).abs();
        if dist < min_distance {
            min_distance = dist;
            closest_idx = idx;
        }
    }

    // Check if we need to interpolate (threshold matches Python's default 0.5 mm)
    const THRESHOLD: f64 = 0.5;
    let dose_plane = if min_distance < THRESHOLD {
        // Direct match, no interpolation needed
        log::debug!(
            "Z-match: z={:.2} found close match at index {} (dist={:.3}mm < {:.1}mm threshold)",
            z_mm,
            closest_idx,
            min_distance,
            THRESHOLD
        );
        dose_grid.dose_3d.get_plane(closest_idx)?
    } else {
        // Need to interpolate between planes
        // Find lower and upper bound planes
        let mut lower_idx = None;
        let mut upper_idx = None;
        let mut lower_z = f64::NEG_INFINITY;
        let mut upper_z = f64::INFINITY;

        for (idx, &z_pos) in z_positions.iter().enumerate() {
            if z_pos <= z_mm && z_pos > lower_z {
                lower_z = z_pos;
                lower_idx = Some(idx);
            }
            if z_pos >= z_mm && z_pos < upper_z {
                upper_z = z_pos;
                upper_idx = Some(idx);
            }
        }

        // If we have both bounds, interpolate
        if let (Some(lb), Some(ub)) = (lower_idx, upper_idx) {
            if lb != ub {
                // Calculate fractional distance
                let fz = (z_mm - lower_z) / (upper_z - lower_z);

                // Get both dose planes
                let lower_plane = dose_grid.dose_3d.get_plane(lb)?;
                let upper_plane = dose_grid.dose_3d.get_plane(ub)?;

                // Linear interpolation between planes
                let mut interpolated_plane = Array2::zeros(lower_plane.dim());
                for ((r, c), lower_val) in lower_plane.indexed_iter() {
                    let upper_val = upper_plane[[r, c]];
                    // Interpolated = (1-f) * lower + f * upper
                    interpolated_plane[[r, c]] =
                        (1.0 - fz as f32) * lower_val + (fz as f32) * upper_val;
                }

                log::debug!(
                    "Z-interpolation: z={:.2} between planes at {:.2} and {:.2} (f={:.3})",
                    z_mm,
                    lower_z,
                    upper_z,
                    fz
                );
                interpolated_plane
            } else {
                // Same index for lower and upper, just use it
                dose_grid.dose_3d.get_plane(lb)?
            }
        } else {
            // Can't interpolate - this should not happen as we check bounds earlier
            // But if it does, return zero plane for safety
            log::warn!(
                "Z-interpolation: Cannot interpolate at z={:.2}, returning zero plane",
                z_mm
            );
            let dose_shape = dose_grid.dose_3d.get_plane(0)?.dim();
            Array2::zeros(dose_shape)
        }
    };

    // Apply XY interpolation if requested
    if let Some((target_row_mm, target_col_mm)) = interpolation_resolution {
        let orig_spacing = (
            dose_grid.pixel_spacing_row_mm,
            dose_grid.pixel_spacing_col_mm,
        );

        if let Some((row_scale, col_scale)) = interpolation::calculate_interpolation_scale(
            orig_spacing,
            Some((target_row_mm, target_col_mm)),
        ) {
            // Apply bilinear interpolation
            let interpolated = interpolation::rescale_2d(dose_plane.view(), row_scale, col_scale);
            return Some(interpolated);
        }
    }

    Some(dose_plane)
}

pub struct DvhEngine;

impl DvhEngine {
    /// Calculate DVH for a single ROI
    pub fn calculate_dvh(
        roi: &Roi,
        dose_grid: &DoseGrid,
        options: &DvhOptions,
    ) -> Result<DvhResult, DvhError> {
        log::debug!("Calculating DVH for ROI {} ({})", roi.id, roi.name);

        // Calculate max dose in cGy
        let max_dose_cgy = Self::calculate_max_dose(dose_grid, options.limit_cgy)?;

        // Initialize accumulators for late rescale pattern
        let mut plane_histograms: Vec<Vec<f64>> = Vec::new();
        let mut plane_volumes_mm3: Vec<f64> = Vec::new();
        let mut notes = None;

        // Get dose grid LUTs (potentially modified for extents/interpolation)
        let (col_lut, row_lut) = if let Some((target_row_mm, target_col_mm)) =
            options.interpolation_resolution_mm
        {
            // Recalculate LUTs for interpolated resolution
            // Calculate scale factors based on target resolution
            let orig_row_spacing = dose_grid.pixel_spacing_row_mm;
            let orig_col_spacing = dose_grid.pixel_spacing_col_mm;
            let row_scale = orig_row_spacing / target_row_mm;
            let col_scale = orig_col_spacing / target_col_mm;

            // Calculate new dimensions based on scale
            let orig_cols = dose_grid.col_lut.len();
            let orig_rows = dose_grid.row_lut.len();
            let new_cols = ((orig_cols as f64 - 1.0) * col_scale + 1.0).round() as usize;
            let new_rows = ((orig_rows as f64 - 1.0) * row_scale + 1.0).round() as usize;

            // Generate new LUTs with linear interpolation
            let col_start = dose_grid.col_lut[0];
            let col_end = dose_grid.col_lut[orig_cols - 1];
            let new_col_lut: Vec<f64> = (0..new_cols)
                .map(|i| col_start + (col_end - col_start) * (i as f64) / ((new_cols - 1) as f64))
                .collect();

            let row_start = dose_grid.row_lut[0];
            let row_end = dose_grid.row_lut[orig_rows - 1];
            let new_row_lut: Vec<f64> = (0..new_rows)
                .map(|i| row_start + (row_end - row_start) * (i as f64) / ((new_rows - 1) as f64))
                .collect();

            (new_col_lut, new_row_lut)
        } else if options.use_structure_extents {
            // TODO: Implement structure extents calculation
            (dose_grid.col_lut.clone(), dose_grid.row_lut.clone())
        } else {
            (dose_grid.col_lut.clone(), dose_grid.row_lut.clone())
        };

        // Apply thickness override if specified
        let base_thickness = options.thickness_override_mm.unwrap_or(roi.thickness_mm);

        // Apply Z-plane interpolation if requested
        let (interpolated_planes, adjusted_thickness) =
            if options.interpolation_segments_between_planes > 0 {
                let interpolated = ZInterpolator::interpolate_planes(
                    &roi.planes,
                    options.interpolation_segments_between_planes,
                );
                let new_thickness = ZInterpolator::adjusted_thickness(
                    base_thickness,
                    options.interpolation_segments_between_planes,
                );
                log::debug!(
                    "Z-interpolation: {} -> {} planes, thickness: {:.2} -> {:.2} mm",
                    roi.planes.len(),
                    interpolated.len(),
                    base_thickness,
                    new_thickness
                );
                (interpolated, new_thickness)
            } else {
                (roi.planes.clone(), base_thickness)
            };

        // Process each plane (original or interpolated)
        let plane_count = interpolated_planes.len();

        for (plane_idx, (z_pos, contours)) in interpolated_planes.iter().enumerate() {
            log::debug!(
                "Processing plane {}/{} at z={} with {} contours",
                plane_idx + 1,
                plane_count,
                z_pos.0,
                contours.len()
            );

            // Get dose plane at this Z position with interpolation if enabled
            let interpolation_res = options.interpolation_resolution_mm;

            let dose_plane = get_dose_plane_at_z(
                dose_grid,
                z_pos.0,
                adjusted_thickness,
                interpolation_res,
                options.calculate_full_volume,
            );

            // Debug for skin
            if (roi.name.to_lowercase() == "skin" || roi.name.to_lowercase() == "peau")
                && plane_idx < 5
            {
                eprintln!(
                    "  Plane {} Z={:.1}: dose_plane={}",
                    plane_idx,
                    z_pos.0,
                    if dose_plane.is_some() {
                        "FOUND"
                    } else {
                        "MISSING"
                    }
                );
            }

            if let Some(dose_plane) = dose_plane {
                // Get appropriate LUTs based on whether interpolation was applied
                let (plane_col_lut, plane_row_lut) = if interpolation_res.is_some() {
                    // Already calculated interpolated LUTs above
                    (&col_lut, &row_lut)
                } else {
                    // Use original LUTs
                    (&col_lut, &row_lut)
                };

                // Calculate plane histogram (returns counts, not volume)
                let (plane_hist, voxel_count) = Self::calculate_plane_histogram(
                    contours,
                    &dose_plane,
                    plane_col_lut,
                    plane_row_lut,
                    dose_grid.x_lut_index,
                    dose_grid.scale_to_gy,
                    max_dose_cgy,
                )?;

                // Store histogram and volume for late rescale
                plane_histograms.push(plane_hist);

                // Calculate volume using mean LUT spacing (not PixelSpacing directly)
                let dx_mm = Self::mean_lut_spacing(&col_lut);
                let dy_mm = Self::mean_lut_spacing(&row_lut);
                let plane_volume_mm3 = voxel_count * dx_mm * dy_mm * adjusted_thickness;

                plane_volumes_mm3.push(plane_volume_mm3);
            } else {
                // Handle missing dose plane
                if options.calculate_full_volume {
                    log::warn!(
                        "Dose plane not found for z={}. Including in volume calculation.",
                        z_pos.0
                    );

                    // Add zero histogram but count volume
                    let zero_hist = vec![0.0; max_dose_cgy as usize];
                    plane_histograms.push(zero_hist);

                    // Calculate volume using mean LUT spacing
                    let dx_mm = Self::mean_lut_spacing(&col_lut);
                    let dy_mm = Self::mean_lut_spacing(&row_lut);
                    // Use approximate voxel count based on contour bounds
                    let approx_voxels = Self::estimate_contour_voxels(
                        contours,
                        &col_lut,
                        &row_lut,
                        dose_grid.x_lut_index,
                    );
                    let plane_volume_mm3 =
                        approx_voxels as f64 * dx_mm * dy_mm * adjusted_thickness;
                    plane_volumes_mm3.push(plane_volume_mm3);
                    if notes.is_none() {
                        notes = Some("Dose grid does not encompass every contour. Volume calculated for all contours.".to_string());
                    }
                } else {
                    log::info!("Skipping contour at z={} - outside dose grid", z_pos.0);
                    if notes.is_none() {
                        notes = Some("Dose grid does not encompass every contour. Volume calculated within dose grid.".to_string());
                    }
                }
            }
        }

        // Apply late rescale pattern (Python parity)
        let (differential_hist_cgy, total_volume_cc) =
            Self::finalize_histogram(plane_histograms, plane_volumes_mm3);

        log::debug!("Total volume: {} cc", total_volume_cc);
        // Trim trailing zeros (matches np.trim_zeros(hist, trim='b'))
        let trimmed_histogram = HistogramCalculator::trim_zeros(differential_hist_cgy);

        // Generate bins (matching numpy.arange behavior)
        let bins: Vec<f64> = if trimmed_histogram.len() == 1 {
            vec![0.0, 1.0]
        } else {
            (0..=trimmed_histogram.len())
                .map(|i| i as f64 / 100.0) // Convert from cGy to Gy
                .collect()
        };

        // Calculate cumulative DVH
        let cumulative = HistogramCalculator::to_cumulative(&trimmed_histogram);

        // Calculate statistics
        let stats = Self::calculate_stats(&trimmed_histogram, &cumulative, &bins, total_volume_cc);

        Ok(DvhResult {
            notes,
            differential_hist_cgy: trimmed_histogram,
            bins: bins.clone(),
            cumulative,
            name: roi.name.clone(),
            total_volume_cc,
            stats,
        })
    }

    /// Calculate histogram for a single plane
    fn calculate_plane_histogram(
        contours: &[crate::types::Contour],
        dose_plane: &Array2<f32>,
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
        dose_scaling: f64,
        max_dose_cgy: u32,
    ) -> Result<(Vec<f64>, f64), DvhError> {
        // Create mask for all contours with XOR using matplotlib-compatible algorithm
        let mask = MatplotlibPolygon::create_plane_mask(
            &contours
                .iter()
                .map(|c| c.points.clone())
                .collect::<Vec<_>>(),
            col_lut,
            row_lut,
            x_lut_index,
        );

        // Calculate histogram
        let (histogram, voxel_count) =
            HistogramCalculator::calculate_histogram(dose_plane, &mask, dose_scaling, max_dose_cgy);

        Ok((histogram, voxel_count))
    }

    /// Calculate maximum dose for histogram binning
    /// Matches Python: maxdose = int(dosemax * scaling * 100) + 1
    fn calculate_max_dose(dose_grid: &DoseGrid, limit_cgy: Option<u32>) -> Result<u32, DvhError> {
        // Find max dose value in the grid
        let max_dose_value = match &dose_grid.dose_3d {
            crate::types::DoseBacking::Owned(arr) => arr
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(0.0) as f64,
            #[cfg(feature = "memmap")]
            crate::types::DoseBacking::MemMapped { .. } => {
                // For memory-mapped, we'd need to iterate through planes
                // For now, use a reasonable default
                1000.0
            }
        };

        // Convert to cGy and add 1 (matching Python exactly)
        // Python: maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100) + 1
        let max_dose_cgy = (max_dose_value * dose_grid.scale_to_gy * 100.0) as u32 + 1;

        // Apply limit if specified (Python: if limit < maxdose: maxdose = limit)
        Ok(limit_cgy
            .map(|limit| limit.min(max_dose_cgy))
            .unwrap_or(max_dose_cgy))
    }

    /// Calculate mean spacing from LUT differences
    /// Matches Python: abs(mean(diff(lut)))
    fn mean_lut_spacing(lut: &[f64]) -> f64 {
        if lut.len() < 2 {
            return 1.0; // Default spacing
        }

        let mut sum = 0.0;
        for i in 1..lut.len() {
            sum += (lut[i] - lut[i - 1]).abs();
        }
        sum / (lut.len() - 1) as f64
    }

    /// Estimate voxel count for missing dose plane
    fn estimate_contour_voxels(
        contours: &[crate::types::Contour],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
    ) -> usize {
        // Create mask to count voxels using matplotlib-compatible algorithm
        let mask = MatplotlibPolygon::create_plane_mask(
            &contours
                .iter()
                .map(|c| c.points.clone())
                .collect::<Vec<_>>(),
            col_lut,
            row_lut,
            x_lut_index,
        );
        mask.iter().filter(|&&v| v).count()
    }

    /// Finalize histogram with late rescale (Python parity)
    /// Matches: hist = hist * volume_cc / sum(hist)
    fn finalize_histogram(
        plane_histograms: Vec<Vec<f64>>,
        plane_volumes_mm3: Vec<f64>,
    ) -> (Vec<f64>, f64) {
        // Calculate total volume in cc
        let total_volume_cc = plane_volumes_mm3.iter().sum::<f64>() / 1000.0;

        if plane_histograms.is_empty() {
            return (vec![0.0], 0.0);
        }

        let max_bins = plane_histograms.iter().map(|h| h.len()).max().unwrap_or(1);
        let mut combined_counts = vec![0.0; max_bins];

        // Sum all histogram counts (not scaled by volume yet)
        for hist in &plane_histograms {
            for (i, &count) in hist.iter().enumerate() {
                combined_counts[i] += count;
            }
        }

        // Check if we have any counts
        let total_counts: f64 = combined_counts.iter().sum();
        if total_counts == 0.0 {
            return (vec![0.0], total_volume_cc);
        }

        // Late rescale: scale counts to volume
        let scale_factor = total_volume_cc / total_counts;
        let mut scaled_histogram: Vec<f64> = combined_counts
            .iter()
            .map(|&count| count * scale_factor)
            .collect();

        (scaled_histogram, total_volume_cc)
    }

    /// Calculate DVH statistics
    fn calculate_stats(
        differential: &[f64],
        cumulative: &[f64],
        bins: &[f64],
        total_volume_cc: f64,
    ) -> DvhStats {
        // Calculate D-values (dose at volume percentages)
        let d100 = Self::calculate_dx(cumulative, bins, 100.0);
        let d98 = Self::calculate_dx(cumulative, bins, 98.0);
        let d95 = Self::calculate_dx(cumulative, bins, 95.0);
        let d90 = Self::calculate_dx(cumulative, bins, 90.0);
        let d80 = Self::calculate_dx(cumulative, bins, 80.0);
        let d70 = Self::calculate_dx(cumulative, bins, 70.0);
        let d60 = Self::calculate_dx(cumulative, bins, 60.0);
        let d50 = Self::calculate_dx(cumulative, bins, 50.0);
        let d40 = Self::calculate_dx(cumulative, bins, 40.0);
        let d30 = Self::calculate_dx(cumulative, bins, 30.0);
        let d20 = Self::calculate_dx(cumulative, bins, 20.0);
        let d10 = Self::calculate_dx(cumulative, bins, 10.0);
        let d5 = Self::calculate_dx(cumulative, bins, 5.0);
        let d2 = Self::calculate_dx(cumulative, bins, 2.0);
        let d1 = Self::calculate_dx(cumulative, bins, 1.0);
        let d0 = Self::calculate_dx(cumulative, bins, 0.0);

        // Calculate mean dose from differential histogram
        let mean_gy = Self::calculate_mean_dose(differential, bins);

        // Homogeneity index
        let homogeneity_index = if d50 > 0.0 { (d2 - d98) / d50 } else { 0.0 };

        DvhStats {
            n_bins: cumulative.len(),
            total_cc: total_volume_cc,
            min_gy: d100,
            max_gy: d0,
            mean_gy,
            d100_gy: d100,
            d98_gy: d98,
            d95_gy: d95,
            d90_gy: d90,
            d80_gy: d80,
            d70_gy: d70,
            d60_gy: d60,
            d50_gy: d50,
            d40_gy: d40,
            d30_gy: d30,
            d20_gy: d20,
            d10_gy: d10,
            d5_gy: d5,
            d2_gy: d2,
            d1_gy: d1,
            d0_gy: d0,
            homogeneity_index,
        }
    }

    /// Calculate Dx (dose at x% volume)
    fn calculate_dx(cumulative: &[f64], bins: &[f64], percent: f64) -> f64 {
        if cumulative.is_empty() || cumulative[0] == 0.0 {
            return 0.0;
        }

        let target_volume = cumulative[0] * percent / 100.0;

        // Find where cumulative volume drops below target
        for (i, &vol) in cumulative.iter().enumerate() {
            if vol <= target_volume {
                if i < bins.len() - 1 {
                    return bins[i];
                }
            }
        }

        // If not found, return min dose for D100 or max dose for D0
        if percent == 100.0 && bins.len() > 0 {
            bins[0]
        } else if percent == 0.0 && bins.len() > 1 {
            bins[bins.len() - 2]
        } else {
            0.0
        }
    }

    /// Calculate mean dose from differential histogram
    fn calculate_mean_dose(differential: &[f64], bins: &[f64]) -> f64 {
        if differential.is_empty() {
            return 0.0;
        }

        let total_volume: f64 = differential.iter().sum();
        if total_volume == 0.0 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;

        // Use bin centers for mean calculation
        for i in 0..differential.len().min(bins.len() - 1) {
            let bin_center = (bins[i] + bins[i + 1]) / 2.0;
            weighted_sum += bin_center * differential[i];
        }

        weighted_sum / total_volume
    }
}

/// Public API for computing DVH from files
pub fn compute_dvh(
    rtstruct_path: impl AsRef<Path>,
    rtdose_path: impl AsRef<Path>,
    roi_number: i32,
    options: &DvhOptions,
) -> Result<DvhResult, DvhError> {
    // Parse DICOM files using proper parser
    let rois = parse_rtstruct(rtstruct_path.as_ref())?;
    let dose_grid = parse_rtdose(rtdose_path.as_ref())?;

    // Find ROI
    let roi = rois
        .iter()
        .find(|r| r.id == roi_number)
        .ok_or(DvhError::InvalidRoi(roi_number))?;

    // Calculate DVH
    DvhEngine::calculate_dvh(roi, &dose_grid, options)
}

/// Compute DVH for all ROIs (parallel)
pub fn compute_all_dvhs(
    rtstruct_path: impl AsRef<Path>,
    rtdose_path: impl AsRef<Path>,
    options: &DvhOptions,
) -> Result<Vec<DvhResult>, DvhError> {
    // Parse DICOM files using proper parser
    let rois = parse_rtstruct(rtstruct_path.as_ref())?;
    let dose_grid = parse_rtdose(rtdose_path.as_ref())?;

    // Calculate DVH for each ROI in parallel
    let results: Result<Vec<_>, _> = rois
        .par_iter()
        .map(|roi| DvhEngine::calculate_dvh(roi, &dose_grid, options))
        .collect();

    results
}
