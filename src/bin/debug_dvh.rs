use aitrium_dvh::dicom_parser::{parse_rtdose, parse_rtstruct};
use aitrium_dvh::engine::DvhEngine;
/// Debug DVH calculation to find dose discrepancy
use aitrium_dvh::DvhOptions;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <dicom_directory> <roi_name>", args[0]);
        std::process::exit(1);
    }

    let dicom_dir = PathBuf::from(&args[1]);
    let target_roi = &args[2];

    // Find DICOM files
    let (rtstruct_path, rtdose_path) = find_dicom_files(&dicom_dir)?;
    eprintln!("Found RTSTRUCT: {}", rtstruct_path.display());
    eprintln!("Found RTDOSE: {}", rtdose_path.display());

    // Parse DICOM files
    eprintln!("Parsing RTDOSE...");
    let dose_grid = parse_rtdose(&rtdose_path)?;

    eprintln!("Parsing RTSTRUCT...");
    let rois = parse_rtstruct(&rtstruct_path)?;

    // Find target ROI
    let roi = rois
        .iter()
        .find(|r| r.name == *target_roi)
        .ok_or_else(|| format!("ROI '{}' not found", target_roi))?;

    eprintln!("\n=== Debugging ROI: {} ===", roi.name);
    eprintln!("ROI ID: {}", roi.id);
    eprintln!("Number of planes: {}", roi.planes.len());
    eprintln!("Thickness: {:.2} mm", roi.thickness_mm);

    // First calculate without interpolation
    eprintln!("\n--- Without Interpolation ---");
    let options_no_interp = DvhOptions {
        limit_cgy: None,
        calculate_full_volume: true,
        use_structure_extents: false,
        interpolation_resolution_mm: None,
        interpolation_segments_between_planes: 0,
        thickness_override_mm: None,
        memmap_rtdose: false,
    };

    match DvhEngine::calculate_dvh(roi, &dose_grid, &options_no_interp) {
        Ok(result) => {
            eprintln!("Volume: {:.2} cc", result.total_volume_cc);
            eprintln!("Mean dose: {:.2} Gy", result.stats.mean_gy);
            eprintln!("Min dose: {:.2} Gy", result.stats.min_gy);
            eprintln!("Max dose: {:.2} Gy", result.stats.max_gy);
            eprintln!("D50: {:.2} Gy", result.stats.d50_gy);
            eprintln!("D95: {:.2} Gy", result.stats.d95_gy);
            eprintln!("Number of bins: {}", result.differential_hist_cgy.len());

            // Check histogram
            let non_zero_bins: Vec<_> = result
                .differential_hist_cgy
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .collect();
            eprintln!("Non-zero bins: {}", non_zero_bins.len());
            if non_zero_bins.len() < 10 {
                for (i, &val) in &non_zero_bins {
                    eprintln!("  Bin {}: {:.4} cc", i, val);
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // Now with interpolation
    eprintln!("\n--- With Interpolation (2x) ---");
    let target_row = dose_grid.pixel_spacing_row_mm / 2.0;
    let target_col = dose_grid.pixel_spacing_col_mm / 2.0;

    let options_interp = DvhOptions {
        limit_cgy: None,
        calculate_full_volume: true,
        use_structure_extents: false,
        interpolation_resolution_mm: Some((target_row, target_col)),
        interpolation_segments_between_planes: 0,
        thickness_override_mm: None,
        memmap_rtdose: false,
    };

    match DvhEngine::calculate_dvh(roi, &dose_grid, &options_interp) {
        Ok(result) => {
            eprintln!("Volume: {:.2} cc", result.total_volume_cc);
            eprintln!("Mean dose: {:.2} Gy", result.stats.mean_gy);
            eprintln!("Min dose: {:.2} Gy", result.stats.min_gy);
            eprintln!("Max dose: {:.2} Gy", result.stats.max_gy);
            eprintln!("D50: {:.2} Gy", result.stats.d50_gy);
            eprintln!("D95: {:.2} Gy", result.stats.d95_gy);
            eprintln!("Number of bins: {}", result.differential_hist_cgy.len());

            // Check histogram
            let non_zero_bins: Vec<_> = result
                .differential_hist_cgy
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .collect();
            eprintln!("Non-zero bins: {}", non_zero_bins.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

/// Helper to find DICOM files in a directory
fn find_dicom_files(dir: &PathBuf) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
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

    let rtstruct = rtstruct_path.ok_or("RTSTRUCT file not found")?;
    let rtdose = rtdose_path.ok_or("RTDOSE file not found")?;

    Ok((rtstruct, rtdose))
}
