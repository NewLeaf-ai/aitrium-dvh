/// Structure margin metrics calculation module
///
/// This module computes signed distances from one ROI to another with statistical
/// summaries (min, percentiles, mean) and coverage metrics.
use crate::dicom_parser::{parse_rtdose, parse_rtstruct};
use crate::engine::distance::{signed_distance_field, signed_distance_field_3d};
use crate::engine::orientation::{
    calculate_center_of_mass_2d, direction_to_vector, is_point_in_direction, PatientPosition,
};
use crate::engine::z_interpolation::ZInterpolator;
use crate::geometry::matplotlib_poly::MatplotlibPolygon;
use crate::types::{Contour, DoseGrid, DvhError, MarginDirection, OrderedFloat, Roi};
use ndarray::{Array2, Array3};
use std::collections::BTreeMap;
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
    /// Percentile used as primary summary value (e.g., 5 => P05).
    pub summary_percentile: f64,
    /// Angular cone used for directional filtering based on local outward normal.
    pub direction_cone_degrees: f64,
    /// Synthetic XY resolution in mm for RTSTRUCT-only margin evaluation.
    pub xy_resolution_mm: f64,
    /// Synthetic Z resolution in mm for RTSTRUCT-only margin evaluation.
    /// <= 0.0 means derive from contour thickness and cap to a sane range.
    pub z_resolution_mm: f64,
    /// Maximum synthetic voxels for RTSTRUCT-only margin evaluation.
    /// Resolution is auto-coarsened when this limit is exceeded.
    pub max_voxels: usize,
}

impl Default for MarginOptions {
    fn default() -> Self {
        Self {
            interpolation_segments_between_planes: 0,
            interpolation_resolution_mm: None,
            coverage_thresholds_mm: vec![3.0, 5.0, 7.0],
            direction: None,
            summary_percentile: 5.0,
            direction_cone_degrees: 45.0,
            xy_resolution_mm: 1.0,
            z_resolution_mm: 0.0,
            max_voxels: 5_000_000,
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
    /// Primary summary used for policy evaluation.
    pub summary_mm: f64,
    /// Percentile backing `summary_mm`.
    pub summary_percentile: f64,
    /// Number of boundary samples used for summary statistics.
    pub sample_count: usize,
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

    // Find and merge ROIs by name (case-insensitive). Some RTSTRUCTs contain
    // duplicate ROI names with complementary contour sets.
    let roi_a = merge_matching_rois(&rois, roi_from);
    let roi_b = merge_matching_rois(&rois, roi_to);

    match (roi_a, roi_b) {
        (Some(a), Some(b)) => {
            let result = compute_margin(&a, &b, &dose_grid, options)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Compute directional margin (clearance semantics) from RTSTRUCT only.
///
/// This V2 path is dataset-agnostic for delineation QA:
/// - It does not depend on RTDOSE field-of-view.
/// - It measures boundary clearance from ROI A to ROI B in patient-space mm.
/// - Positive clearance means A is inside B with margin; negative means protrusion.
pub fn compute_margin_directed_rtstruct(
    rtstruct_path: &Path,
    roi_from: &str,
    roi_to: &str,
    options: &MarginOptions,
) -> Result<Option<MarginResult>, DvhError> {
    let rois = parse_rtstruct(rtstruct_path)?;
    compute_margin_directed_rtstruct_on_rois(&rois, roi_from, roi_to, options)
}

/// Compute directional margin (clearance semantics) from an already-parsed RTSTRUCT ROI list.
pub fn compute_margin_directed_rtstruct_on_rois(
    rois: &[Roi],
    roi_from: &str,
    roi_to: &str,
    options: &MarginOptions,
) -> Result<Option<MarginResult>, DvhError> {
    let roi_a = merge_matching_rois(rois, roi_from);
    let roi_b = merge_matching_rois(rois, roi_to);

    match (roi_a, roi_b) {
        (Some(a), Some(b)) => Ok(Some(compute_margin_rtstruct_v2(&a, &b, options)?)),
        _ => Ok(None),
    }
}

fn merge_matching_rois(rois: &[Roi], name: &str) -> Option<Roi> {
    let matches: Vec<&Roi> = rois
        .iter()
        .filter(|r| r.name.eq_ignore_ascii_case(name))
        .collect();

    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        return Some(matches[0].clone());
    }

    let mut planes: BTreeMap<OrderedFloat, Vec<Contour>> = BTreeMap::new();
    let mut thickness_candidates: Vec<f64> = Vec::new();
    for roi in &matches {
        if roi.thickness_mm.is_finite() && roi.thickness_mm > 0.0 {
            thickness_candidates.push(roi.thickness_mm);
        }
        for (z, contours) in &roi.planes {
            planes.entry(*z).or_default().extend(contours.clone());
        }
    }

    let thickness_mm = thickness_candidates
        .into_iter()
        .reduce(f64::min)
        .unwrap_or(matches[0].thickness_mm);

    Some(Roi {
        id: matches[0].id,
        name: matches[0].name.clone(),
        planes,
        thickness_mm,
    })
}

#[derive(Debug, Clone)]
struct SyntheticGrid {
    col_lut: Vec<f64>, // X coordinates
    row_lut: Vec<f64>, // Y coordinates
    z_lut: Vec<f64>,   // Z coordinates
    dx_mm: f64,
    dy_mm: f64,
    dz_mm: f64,
}

#[derive(Debug, Clone, Copy)]
enum ContourCombineMode {
    XorEvenOdd,
    Union,
}

fn compute_margin_rtstruct_v2(
    roi_a: &Roi,
    roi_b: &Roi,
    options: &MarginOptions,
) -> Result<MarginResult, DvhError> {
    let grid = build_synthetic_grid(roi_a, roi_b, options)?;
    let mask_a = voxelize_roi_on_grid(roi_a, &grid, options)?;
    let mask_b = voxelize_roi_on_grid(roi_b, &grid, options)?;

    if !mask_a.iter().any(|v| *v) || !mask_b.iter().any(|v| *v) {
        return Err(DvhError::CalculationError(format!(
            "Unable to voxelize ROI geometry for {} -> {} on RTSTRUCT synthetic grid",
            roi_a.name, roi_b.name
        )));
    }

    let boundary_a = extract_boundary_voxels(&mask_a);
    if boundary_a.is_empty() {
        return Err(DvhError::CalculationError(format!(
            "No boundary samples found for source ROI {}",
            roi_a.name
        )));
    }

    let sdf_outer = signed_distance_field_3d(&mask_b, grid.dx_mm, grid.dy_mm, grid.dz_mm);
    let sdf_inner = signed_distance_field_3d(&mask_a, grid.dx_mm, grid.dy_mm, grid.dz_mm);

    match options.direction.unwrap_or(MarginDirection::Uniform) {
        MarginDirection::Uniform => {
            let clearances = collect_directional_clearances(
                &boundary_a,
                &sdf_outer,
                Some(&sdf_inner),
                None,
                options.direction_cone_degrees,
                grid.dx_mm,
                grid.dy_mm,
                grid.dz_mm,
            );
            if clearances.is_empty() {
                return Err(DvhError::CalculationError(format!(
                    "No boundary samples available for Uniform margin extraction from {} to {}",
                    roi_a.name, roi_b.name
                )));
            }
            Ok(build_margin_result_from_clearances(clearances, options))
        }
        MarginDirection::Lateral => {
            let left = collect_directional_clearances(
                &boundary_a,
                &sdf_outer,
                Some(&sdf_inner),
                direction_to_lps_vector(MarginDirection::Left),
                options.direction_cone_degrees,
                grid.dx_mm,
                grid.dy_mm,
                grid.dz_mm,
            );
            if left.is_empty() {
                return Err(DvhError::CalculationError(format!(
                    "No boundary samples available for Left directional subset from {} to {}",
                    roi_a.name, roi_b.name
                )));
            }
            let right = collect_directional_clearances(
                &boundary_a,
                &sdf_outer,
                Some(&sdf_inner),
                direction_to_lps_vector(MarginDirection::Right),
                options.direction_cone_degrees,
                grid.dx_mm,
                grid.dy_mm,
                grid.dz_mm,
            );
            if right.is_empty() {
                return Err(DvhError::CalculationError(format!(
                    "No boundary samples available for Right directional subset from {} to {}",
                    roi_a.name, roi_b.name
                )));
            }

            let left_result = build_margin_result_from_clearances(left, options);
            let right_result = build_margin_result_from_clearances(right, options);

            if !left_result.summary_mm.is_finite() || !right_result.summary_mm.is_finite() {
                return Ok(infinite_margin_result(options));
            }

            Ok(MarginResult {
                min_mm: left_result.min_mm.min(right_result.min_mm),
                p05_mm: left_result.p05_mm.min(right_result.p05_mm),
                p50_mm: left_result.p50_mm.min(right_result.p50_mm),
                p95_mm: left_result.p95_mm.min(right_result.p95_mm),
                mean_mm: (left_result.mean_mm + right_result.mean_mm) / 2.0,
                summary_mm: left_result.summary_mm.min(right_result.summary_mm),
                summary_percentile: options.summary_percentile.clamp(0.0, 100.0),
                sample_count: left_result.sample_count + right_result.sample_count,
                coverage_within_thresholds: left_result
                    .coverage_within_thresholds
                    .iter()
                    .zip(right_result.coverage_within_thresholds.iter())
                    .map(|((t, a), (_, b))| (*t, a.min(*b)))
                    .collect(),
            })
        }
        direction => {
            let clearances = collect_directional_clearances(
                &boundary_a,
                &sdf_outer,
                Some(&sdf_inner),
                direction_to_lps_vector(direction),
                options.direction_cone_degrees,
                grid.dx_mm,
                grid.dy_mm,
                grid.dz_mm,
            );
            if clearances.is_empty() {
                return Err(DvhError::CalculationError(format!(
                    "No boundary samples available for {:?} directional subset from {} to {}",
                    direction, roi_a.name, roi_b.name
                )));
            }
            Ok(build_margin_result_from_clearances(clearances, options))
        }
    }
}

fn collect_directional_clearances(
    boundary_voxels: &[(usize, usize, usize)],
    sdf_outer: &Array3<f64>,
    sdf_inner: Option<&Array3<f64>>,
    direction_vector: Option<[f64; 3]>,
    cone_degrees: f64,
    dx_mm: f64,
    dy_mm: f64,
    dz_mm: f64,
) -> Vec<f64> {
    let mut clearances = Vec::with_capacity(boundary_voxels.len());
    let cos_threshold = cone_degrees.to_radians().cos();

    for &(k, i, j) in boundary_voxels {
        if let Some(dir) = direction_vector {
            let Some(sdf_inner) = sdf_inner else {
                continue;
            };
            let normal = local_outward_normal(sdf_inner, k, i, j, dx_mm, dy_mm, dz_mm);
            let Some(normal) = normal else {
                continue;
            };
            let dot = normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2];
            if !dot.is_finite() || dot < cos_threshold {
                continue;
            }
        }

        let signed_distance = sdf_outer[[k, i, j]];
        if !signed_distance.is_finite() {
            continue;
        }
        clearances.push(-signed_distance);
    }

    clearances
}

fn build_margin_result_from_clearances(
    mut clearances: Vec<f64>,
    options: &MarginOptions,
) -> MarginResult {
    if clearances.is_empty() {
        return infinite_margin_result(options);
    }

    clearances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_mm = clearances[0];
    let p05_mm = percentile_sorted(&clearances, 5.0);
    let p50_mm = percentile_sorted(&clearances, 50.0);
    let p95_mm = percentile_sorted(&clearances, 95.0);
    let mean_mm = clearances.iter().sum::<f64>() / clearances.len() as f64;
    let summary_percentile = options.summary_percentile.clamp(0.0, 100.0);
    let summary_mm = percentile_sorted(&clearances, summary_percentile);

    let coverage_within_thresholds = options
        .coverage_thresholds_mm
        .iter()
        .map(|&threshold| {
            let covered = clearances.iter().filter(|&&v| v >= threshold).count() as f64;
            let pct = (covered / clearances.len() as f64) * 100.0;
            (threshold, pct)
        })
        .collect();

    MarginResult {
        min_mm,
        p05_mm,
        p50_mm,
        p95_mm,
        mean_mm,
        summary_mm,
        summary_percentile,
        sample_count: clearances.len(),
        coverage_within_thresholds,
    }
}

fn infinite_margin_result(options: &MarginOptions) -> MarginResult {
    MarginResult {
        min_mm: f64::INFINITY,
        p05_mm: f64::INFINITY,
        p50_mm: f64::INFINITY,
        p95_mm: f64::INFINITY,
        mean_mm: f64::INFINITY,
        summary_mm: f64::INFINITY,
        summary_percentile: options.summary_percentile.clamp(0.0, 100.0),
        sample_count: 0,
        coverage_within_thresholds: options
            .coverage_thresholds_mm
            .iter()
            .map(|&t| (t, 0.0))
            .collect(),
    }
}

fn build_synthetic_grid(
    roi_a: &Roi,
    roi_b: &Roi,
    options: &MarginOptions,
) -> Result<SyntheticGrid, DvhError> {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;

    for roi in [roi_a, roi_b] {
        for (z, contours) in &roi.planes {
            z_min = z_min.min(z.0);
            z_max = z_max.max(z.0);
            for contour in contours {
                for p in &contour.points {
                    x_min = x_min.min(p[0]);
                    x_max = x_max.max(p[0]);
                    y_min = y_min.min(p[1]);
                    y_max = y_max.max(p[1]);
                }
            }
        }
    }

    if !x_min.is_finite() || !y_min.is_finite() || !z_min.is_finite() {
        return Err(DvhError::DoseGridError(
            "Unable to derive ROI bounds for RTSTRUCT-only margin".to_string(),
        ));
    }

    let mut dx_mm = options.xy_resolution_mm.max(0.2);
    let mut dy_mm = options.xy_resolution_mm.max(0.2);
    let mut dz_mm = if options.z_resolution_mm > 0.0 {
        options.z_resolution_mm
    } else {
        derive_default_z_resolution(roi_a, roi_b)
    }
    .max(0.2);

    let mut col_lut;
    let mut row_lut;
    let mut z_lut;
    let max_voxels = options.max_voxels.max(1);

    let mut iter = 0;
    loop {
        let pad_x = dx_mm * 2.0;
        let pad_y = dy_mm * 2.0;
        let pad_z = dz_mm * 2.0;

        col_lut = build_axis_lut(x_min - pad_x, x_max + pad_x, dx_mm);
        row_lut = build_axis_lut(y_min - pad_y, y_max + pad_y, dy_mm);
        z_lut = build_axis_lut(z_min - pad_z, z_max + pad_z, dz_mm);

        let voxels = col_lut
            .len()
            .saturating_mul(row_lut.len())
            .saturating_mul(z_lut.len());
        if voxels <= max_voxels || iter >= 8 {
            break;
        }

        let scale = (voxels as f64 / max_voxels as f64).cbrt() * 1.05;
        dx_mm *= scale;
        dy_mm *= scale;
        dz_mm *= scale;
        iter += 1;
    }

    Ok(SyntheticGrid {
        col_lut,
        row_lut,
        z_lut,
        dx_mm,
        dy_mm,
        dz_mm,
    })
}

fn derive_default_z_resolution(roi_a: &Roi, roi_b: &Roi) -> f64 {
    let t = roi_a.thickness_mm.min(roi_b.thickness_mm);
    if t.is_finite() && t > 0.0 {
        t.clamp(0.5, 5.0)
    } else {
        2.5
    }
}

fn build_axis_lut(min_v: f64, max_v: f64, step: f64) -> Vec<f64> {
    let step = step.max(0.1);
    let span = (max_v - min_v).max(0.0);
    let n = (span / step).ceil() as usize + 1;
    (0..n).map(|idx| min_v + idx as f64 * step).collect()
}

fn voxelize_roi_on_grid(
    roi: &Roi,
    grid: &SyntheticGrid,
    options: &MarginOptions,
) -> Result<Array3<bool>, DvhError> {
    let planes = if options.interpolation_segments_between_planes > 0 {
        ZInterpolator::interpolate_planes(
            &roi.planes,
            options.interpolation_segments_between_planes,
        )
    } else {
        roi.planes.clone()
    };

    let mut volume = Array3::from_elem(
        (grid.z_lut.len(), grid.row_lut.len(), grid.col_lut.len()),
        false,
    );

    if planes.is_empty() {
        return Ok(volume);
    }

    let plane_zs: Vec<f64> = planes.keys().map(|k| k.0).collect();
    let z_tolerance = (grid.dz_mm * 0.75).max(roi.thickness_mm * 0.5).max(0.5);

    for (k, &z) in grid.z_lut.iter().enumerate() {
        let Some(nearest_z) = find_nearest_z(&plane_zs, z) else {
            continue;
        };
        if (nearest_z - z).abs() > z_tolerance {
            continue;
        }

        let Some(contours) = planes.get(&OrderedFloat(nearest_z)) else {
            continue;
        };
        let mask_2d = build_combined_mask(
            contours,
            &grid.col_lut,
            &grid.row_lut,
            0,
            ContourCombineMode::Union,
        )?;
        for ((i, j), &is_inside) in mask_2d.indexed_iter() {
            if is_inside {
                volume[[k, i, j]] = true;
            }
        }
    }

    Ok(volume)
}

fn find_nearest_z(values: &[f64], target: f64) -> Option<f64> {
    let mut best = None;
    let mut best_dist = f64::INFINITY;
    for &v in values {
        let d = (v - target).abs();
        if d < best_dist {
            best_dist = d;
            best = Some(v);
        }
    }
    best
}

fn extract_boundary_voxels(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
    let (nz, ny, nx) = mask.dim();
    let mut out = Vec::new();
    for k in 0..nz {
        for i in 0..ny {
            for j in 0..nx {
                if !mask[[k, i, j]] {
                    continue;
                }
                let neighbors = [
                    (k.wrapping_sub(1), i, j, k > 0),
                    (k + 1, i, j, k + 1 < nz),
                    (k, i.wrapping_sub(1), j, i > 0),
                    (k, i + 1, j, i + 1 < ny),
                    (k, i, j.wrapping_sub(1), j > 0),
                    (k, i, j + 1, j + 1 < nx),
                ];
                let mut is_boundary = false;
                for (nk, ni, nj, in_bounds) in neighbors {
                    if !in_bounds || !mask[[nk, ni, nj]] {
                        is_boundary = true;
                        break;
                    }
                }
                if is_boundary {
                    out.push((k, i, j));
                }
            }
        }
    }
    out
}

fn local_outward_normal(
    sdf: &Array3<f64>,
    k: usize,
    i: usize,
    j: usize,
    dx_mm: f64,
    dy_mm: f64,
    dz_mm: f64,
) -> Option<[f64; 3]> {
    let sample = |kk: isize, ii: isize, jj: isize| -> f64 {
        let (nz, ny, nx) = sdf.dim();
        let kk = kk.clamp(0, (nz - 1) as isize) as usize;
        let ii = ii.clamp(0, (ny - 1) as isize) as usize;
        let jj = jj.clamp(0, (nx - 1) as isize) as usize;
        sdf[[kk, ii, jj]]
    };

    let k_i = k as isize;
    let i_i = i as isize;
    let j_i = j as isize;

    let gx = (sample(k_i, i_i, j_i + 1) - sample(k_i, i_i, j_i - 1)) / (2.0 * dx_mm.max(1e-6));
    let gy = (sample(k_i, i_i + 1, j_i) - sample(k_i, i_i - 1, j_i)) / (2.0 * dy_mm.max(1e-6));
    let gz = (sample(k_i + 1, i_i, j_i) - sample(k_i - 1, i_i, j_i)) / (2.0 * dz_mm.max(1e-6));

    let mag = (gx * gx + gy * gy + gz * gz).sqrt();
    if !mag.is_finite() || mag < 1e-6 {
        return None;
    }

    Some([gx / mag, gy / mag, gz / mag])
}

fn direction_to_lps_vector(direction: MarginDirection) -> Option<[f64; 3]> {
    match direction {
        MarginDirection::Uniform | MarginDirection::Lateral => None,
        MarginDirection::Left => Some([1.0, 0.0, 0.0]),
        MarginDirection::Right => Some([-1.0, 0.0, 0.0]),
        MarginDirection::Posterior => Some([0.0, 1.0, 0.0]),
        MarginDirection::Anterior => Some([0.0, -1.0, 0.0]),
        MarginDirection::Superior => Some([0.0, 0.0, 1.0]),
        MarginDirection::Inferior => Some([0.0, 0.0, -1.0]),
    }
}

fn percentile_sorted(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
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
            summary_mm: f64::INFINITY,
            summary_percentile: options.summary_percentile,
            sample_count: 0,
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
            let mask_b = build_combined_mask(
                contours_b,
                &col_lut,
                &row_lut,
                dose_grid.x_lut_index,
                ContourCombineMode::XorEvenOdd,
            )?;
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
        let mask_a = build_combined_mask(
            contours_a,
            &col_lut,
            &row_lut,
            dose_grid.x_lut_index,
            ContourCombineMode::XorEvenOdd,
        )?;
        let mask_b = build_combined_mask(
            contours_b,
            &col_lut,
            &row_lut,
            dose_grid.x_lut_index,
            ContourCombineMode::XorEvenOdd,
        )?;

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
            summary_mm: f64::INFINITY,
            summary_percentile: options.summary_percentile,
            sample_count: 0,
            coverage_within_thresholds: options
                .coverage_thresholds_mm
                .iter()
                .map(|&t| (t, 0.0))
                .collect(),
        });
    }

    // Drop non-finite samples before computing summary statistics.
    let mut non_finite_samples = 0usize;
    distance_samples.retain(|(d, _)| {
        if d.is_finite() {
            true
        } else {
            non_finite_samples += 1;
            false
        }
    });

    if non_finite_samples > 0 {
        eprintln!(
            "Dropped {} non-finite margin samples for {} -> {}",
            non_finite_samples, roi_a.name, roi_b.name
        );
    }

    if distance_samples.is_empty() {
        return Ok(MarginResult {
            min_mm: f64::INFINITY,
            p05_mm: f64::INFINITY,
            p50_mm: f64::INFINITY,
            p95_mm: f64::INFINITY,
            mean_mm: f64::INFINITY,
            summary_mm: f64::INFINITY,
            summary_percentile: options.summary_percentile,
            sample_count: 0,
            coverage_within_thresholds: options
                .coverage_thresholds_mm
                .iter()
                .map(|&t| (t, 0.0))
                .collect(),
        });
    }

    // Calculate statistics
    let total_weight: f64 = distance_samples.iter().map(|(_, w)| w).sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Ok(MarginResult {
            min_mm: f64::INFINITY,
            p05_mm: f64::INFINITY,
            p50_mm: f64::INFINITY,
            p95_mm: f64::INFINITY,
            mean_mm: f64::INFINITY,
            summary_mm: f64::INFINITY,
            summary_percentile: options.summary_percentile,
            sample_count: 0,
            coverage_within_thresholds: options
                .coverage_thresholds_mm
                .iter()
                .map(|&t| (t, 0.0))
                .collect(),
        });
    }

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
        summary_mm: weighted_percentile(&distance_samples, options.summary_percentile),
        summary_percentile: options.summary_percentile,
        sample_count: distance_samples.len(),
        coverage_within_thresholds: coverage,
    })
}

/// Build a combined mask from multiple contours using XOR (even-odd rule)
fn build_combined_mask(
    contours: &[Contour],
    col_lut: &[f64],
    row_lut: &[f64],
    x_lut_index: u8,
    combine_mode: ContourCombineMode,
) -> Result<Array2<bool>, DvhError> {
    if contours.is_empty() {
        return Ok(Array2::from_elem((row_lut.len(), col_lut.len()), false));
    }

    match combine_mode {
        ContourCombineMode::XorEvenOdd => {
            // Convert contours to the format expected by create_plane_mask
            let contour_points: Vec<Vec<[f64; 2]>> =
                contours.iter().map(|c| c.points.clone()).collect();

            // Preserve legacy XOR semantics used by v1.
            Ok(MatplotlibPolygon::create_plane_mask(
                &contour_points,
                col_lut,
                row_lut,
                x_lut_index,
            ))
        }
        ContourCombineMode::Union => {
            // Robust union composition for v2 avoids cancellation when near-duplicate
            // overlapping contours are present in the same ROI plane.
            let mut combined = Array2::from_elem((row_lut.len(), col_lut.len()), false);
            for contour in contours {
                let contour_mask =
                    MatplotlibPolygon::create_mask(&contour.points, col_lut, row_lut, x_lut_index);
                for ((r, c), &inside) in contour_mask.indexed_iter() {
                    if inside {
                        combined[[r, c]] = true;
                    }
                }
            }
            Ok(combined)
        }
    }
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
    use crate::types::ContourType;
    use std::collections::BTreeMap;

    #[test]
    fn test_margin_options_default() {
        let opts = MarginOptions::default();
        assert_eq!(opts.interpolation_segments_between_planes, 0);
        assert!(opts.interpolation_resolution_mm.is_none());
        assert_eq!(opts.coverage_thresholds_mm, vec![3.0, 5.0, 7.0]);
        assert_eq!(opts.summary_percentile, 5.0);
        assert_eq!(opts.direction_cone_degrees, 45.0);
        assert_eq!(opts.xy_resolution_mm, 1.0);
        assert_eq!(opts.z_resolution_mm, 0.0);
        assert_eq!(opts.max_voxels, 5_000_000);
    }

    #[test]
    fn test_union_mode_prevents_overlap_cancellation() {
        let contour_a = Contour {
            points: vec![[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
            contour_type: ContourType::External,
        };
        let contour_b = Contour {
            points: vec![[2.0, 0.0], [8.0, 0.0], [8.0, 6.0], [2.0, 6.0]],
            contour_type: ContourType::External,
        };
        let contours = vec![contour_a, contour_b];
        let col_lut: Vec<f64> = (0..=8).map(|v| v as f64).collect();
        let row_lut: Vec<f64> = (0..=6).map(|v| v as f64).collect();

        let xor_mask = build_combined_mask(
            &contours,
            &col_lut,
            &row_lut,
            0,
            ContourCombineMode::XorEvenOdd,
        )
        .unwrap();
        let union_mask =
            build_combined_mask(&contours, &col_lut, &row_lut, 0, ContourCombineMode::Union)
                .unwrap();

        let xor_count = xor_mask.iter().filter(|&&v| v).count();
        let union_count = union_mask.iter().filter(|&&v| v).count();
        assert!(union_count > xor_count);
        // Point (3,3) is in overlap region and should survive union composition.
        assert!(union_mask[[3, 3]]);
        assert!(!xor_mask[[3, 3]]);
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

    fn make_rect_roi(
        name: &str,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_planes: &[f64],
        thickness_mm: f64,
    ) -> Roi {
        let mut planes = BTreeMap::new();
        for &z in z_planes {
            planes.insert(
                OrderedFloat(z),
                vec![Contour {
                    points: vec![
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    contour_type: ContourType::External,
                }],
            );
        }
        Roi {
            id: 1,
            name: name.to_string(),
            planes,
            thickness_mm,
        }
    }

    #[test]
    fn test_merge_matching_rois_combines_duplicate_name_contours() {
        let inner = make_rect_roi("CTV", -10.0, 10.0, -10.0, 10.0, &[0.0], 2.0);
        // First duplicate (same name) does NOT cover the inner ROI.
        let outer_part_a = make_rect_roi("PTV", 20.0, 35.0, -10.0, 10.0, &[0.0], 2.0);
        // Second duplicate (same name) does cover the inner ROI.
        let outer_part_b = make_rect_roi("PTV", -15.0, 15.0, -15.0, 15.0, &[0.0], 2.0);

        let mut opts = MarginOptions::default();
        opts.xy_resolution_mm = 1.0;
        opts.z_resolution_mm = 1.0;

        let bad_single = compute_margin_rtstruct_v2(&inner, &outer_part_a, &opts).unwrap();
        assert!(bad_single.summary_mm < 0.0);

        let merged =
            merge_matching_rois(&[outer_part_a.clone(), outer_part_b.clone()], "PTV").unwrap();
        let merged_result = compute_margin_rtstruct_v2(&inner, &merged, &opts).unwrap();
        assert!(merged_result.summary_mm > 0.0);
    }

    #[test]
    fn test_margin_rtstruct_v2_uniform_nested_is_near_expected() {
        let inner = make_rect_roi("CTV", -10.0, 10.0, -10.0, 10.0, &[-2.0, 0.0, 2.0], 2.0);
        // Include superior/inferior expansion so z-boundary samples do not dominate P05.
        let outer = make_rect_roi(
            "PTV",
            -15.0,
            15.0,
            -15.0,
            15.0,
            &[-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            2.0,
        );

        let mut opts = MarginOptions::default();
        opts.xy_resolution_mm = 1.0;
        opts.z_resolution_mm = 1.0;
        opts.summary_percentile = 5.0;
        let result = compute_margin_rtstruct_v2(&inner, &outer, &opts).unwrap();

        assert!(result.summary_mm.is_finite());
        assert!(result.summary_mm > 0.0);
        assert!(
            (result.summary_mm - 5.0).abs() < 1.5,
            "expected near 5mm isotropic margin, got {}",
            result.summary_mm
        );
    }

    #[test]
    fn test_margin_rtstruct_v2_directional_posterior_reduced() {
        let inner = make_rect_roi("CTV", -10.0, 10.0, -10.0, 10.0, &[-2.0, 0.0, 2.0], 2.0);
        // Posterior (+Y) margin is 3mm, anterior (-Y) margin is 7mm
        let outer = make_rect_roi("PTV", -15.0, 15.0, -17.0, 13.0, &[-2.0, 0.0, 2.0], 2.0);

        let mut opts = MarginOptions::default();
        opts.xy_resolution_mm = 1.0;
        opts.z_resolution_mm = 1.0;
        opts.direction = Some(MarginDirection::Posterior);

        let posterior = compute_margin_rtstruct_v2(&inner, &outer, &opts).unwrap();
        assert!(posterior.summary_mm.is_finite());
        assert!(posterior.summary_mm > 1.0 && posterior.summary_mm < 5.0);
    }

    #[test]
    fn test_margin_rtstruct_v2_lateral_uses_min_side() {
        let inner = make_rect_roi("CTV", -10.0, 10.0, -10.0, 10.0, &[-2.0, 0.0, 2.0], 2.0);
        // Right margin = 4mm, Left margin = 6mm
        let outer = make_rect_roi("PTV", -14.0, 16.0, -15.0, 15.0, &[-2.0, 0.0, 2.0], 2.0);

        let mut left_opts = MarginOptions::default();
        left_opts.xy_resolution_mm = 1.0;
        left_opts.z_resolution_mm = 1.0;
        left_opts.direction = Some(MarginDirection::Left);
        let left = compute_margin_rtstruct_v2(&inner, &outer, &left_opts).unwrap();

        let mut right_opts = left_opts.clone();
        right_opts.direction = Some(MarginDirection::Right);
        let right = compute_margin_rtstruct_v2(&inner, &outer, &right_opts).unwrap();

        let mut lateral_opts = left_opts;
        lateral_opts.direction = Some(MarginDirection::Lateral);
        let lateral = compute_margin_rtstruct_v2(&inner, &outer, &lateral_opts).unwrap();

        assert!(left.summary_mm.is_finite() && right.summary_mm.is_finite());
        let expected = left.summary_mm.min(right.summary_mm);
        assert!((lateral.summary_mm - expected).abs() < 0.5);
    }

    #[test]
    fn test_margin_rtstruct_v2_protrusion_negative() {
        let inner = make_rect_roi("CTV", -10.0, 10.0, -10.0, 10.0, &[-2.0, 0.0, 2.0], 2.0);
        // Outer protrudes less than inner on +X side.
        let outer = make_rect_roi("PTV", -15.0, 8.0, -15.0, 15.0, &[-2.0, 0.0, 2.0], 2.0);

        let mut opts = MarginOptions::default();
        opts.xy_resolution_mm = 1.0;
        opts.z_resolution_mm = 1.0;
        let result = compute_margin_rtstruct_v2(&inner, &outer, &opts).unwrap();
        assert!(result.summary_mm < 0.0);
    }
}
