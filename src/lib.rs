pub mod dicom_parser;
pub mod engine;
pub mod geometry;
pub mod types;

pub use dicom_parser::{parse_rtdose, parse_rtstruct};
pub use engine::dvh::{compute_all_dvhs, compute_dvh};
pub use engine::{
    compute_margin_directed, compute_margin_directed_rtstruct,
    compute_margin_directed_rtstruct_on_rois, compute_overlap_by_name,
    euclidean_distance_transform, euclidean_distance_transform_3d, signed_distance_field,
    signed_distance_field_3d, DvhEngine, MarginOptions, MarginResult, OverlapOptions,
    OverlapResult,
};
pub use types::{
    BatchOutput, DvhError, DvhOptions, DvhResult, DvhStats, MarginDirection, RoiDvhJson,
};

use std::path::Path;

/// Convert DvhResult to JSON format matching Python output
pub fn to_json_format(result: &DvhResult) -> RoiDvhJson {
    // Convert cumulative DVH to volumes in cc
    let volumes_cc: Vec<f64> = result.cumulative.clone();

    // Convert bins from Gy to doses
    let doses_gy: Vec<f64> = result.bins[..result.bins.len() - 1].to_vec();

    RoiDvhJson {
        roi_name: result.name.clone(),
        stats: result.stats.clone(),
        doses_gy,
        volumes_cc,
    }
}

/// Compute DVH for all ROIs and return in JSON format
pub fn compute_all_rois_json(
    dicom_dir: impl AsRef<Path>,
    options: &DvhOptions,
) -> Result<BatchOutput, DvhError> {
    // Find RTSTRUCT and RTDOSE files
    let (rtstruct_path, rtdose_path) = find_dicom_files(dicom_dir.as_ref())?;

    // Compute DVH for all ROIs
    let results = compute_all_dvhs(rtstruct_path, rtdose_path, options)?;

    // Convert to JSON format
    let dvhs: Vec<RoiDvhJson> = results.iter().map(to_json_format).collect();

    Ok(BatchOutput { dvhs })
}

/// Helper to find DICOM files in a directory
fn find_dicom_files(dir: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf), DvhError> {
    use std::fs;

    let mut rtstruct_path = None;
    let mut rtdose_path = None;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(name) = path.file_name() {
                let name_str = name.to_string_lossy();
                // Skip macOS hidden files
                if name_str.starts_with("._") {
                    continue;
                }
                if (name_str.contains("RTSTRUCT") || name_str.contains("RS"))
                    && rtstruct_path.is_none()
                {
                    rtstruct_path = Some(path.clone());
                } else if (name_str.contains("RTDOSE") || name_str.contains("RD"))
                    && rtdose_path.is_none()
                {
                    rtdose_path = Some(path.clone());
                }
            }
        }
    }

    let rtstruct =
        rtstruct_path.ok_or_else(|| DvhError::DicomError("RTSTRUCT file not found".to_string()))?;
    let rtdose =
        rtdose_path.ok_or_else(|| DvhError::DicomError("RTDOSE file not found".to_string()))?;

    Ok((rtstruct, rtdose))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dvh_options_default() {
        let opts = DvhOptions::default();
        assert!(opts.calculate_full_volume);
        assert!(!opts.use_structure_extents);
        assert_eq!(opts.interpolation_segments_between_planes, 0);
    }
}
