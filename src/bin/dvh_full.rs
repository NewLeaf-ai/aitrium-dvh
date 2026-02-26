use aitrium_dvh::dicom_parser::{parse_rtdose, parse_rtstruct};
use aitrium_dvh::engine::DvhEngine;
/// Full DVH calculator with configurable options
use aitrium_dvh::{to_json_format, BatchOutput, DvhOptions};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing DICOM files
    dicom_dir: PathBuf,

    /// Enable interpolation (2x oversampling)
    #[arg(short, long)]
    interpolate: bool,

    /// Number of Z-plane segments to interpolate between slices
    #[arg(short = 'z', long, default_value = "0")]
    z_segments: u32,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    /// Include contours outside dose grid (default skips them, matching Python)
    #[arg(long)]
    include_outside_dose: bool,

    /// Use structure extents to limit calculation
    #[arg(long)]
    use_extents: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.debug {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    }

    eprintln!("Processing DICOM directory: {}", args.dicom_dir.display());

    // Find DICOM files
    let (rtstruct_path, rtdose_path) = find_dicom_files(&args.dicom_dir)?;
    eprintln!("Found RTSTRUCT: {}", rtstruct_path.display());
    eprintln!("Found RTDOSE: {}", rtdose_path.display());

    // Parse DICOM files
    eprintln!("Parsing RTDOSE...");
    let dose_grid = parse_rtdose(&rtdose_path)?;
    eprintln!(
        "Dose grid: {} x {} x {} voxels",
        dose_grid.col_lut.len(),
        dose_grid.row_lut.len(),
        dose_grid.grid_frame_offset_vector_mm.len()
    );
    eprintln!(
        "Pixel spacing: row={:.3}mm, col={:.3}mm",
        dose_grid.pixel_spacing_row_mm, dose_grid.pixel_spacing_col_mm
    );

    eprintln!("Parsing RTSTRUCT...");
    let rois = parse_rtstruct(&rtstruct_path)?;
    eprintln!("Parsed {} ROIs", rois.len());

    // Set up options based on CLI args
    let interpolation_resolution = if args.interpolate {
        // Enable interpolation with 2x oversampling
        let target_row = dose_grid.pixel_spacing_row_mm / 2.0;
        let target_col = dose_grid.pixel_spacing_col_mm / 2.0;
        eprintln!(
            "XY Interpolation enabled: {:.3}mm x {:.3}mm",
            target_row, target_col
        );
        Some((target_row, target_col))
    } else {
        None
    };

    if args.z_segments > 0 {
        eprintln!(
            "Z-plane interpolation: {} segments between planes",
            args.z_segments
        );
    }

    let options = DvhOptions {
        limit_cgy: None,
        calculate_full_volume: args.include_outside_dose, // Default false, matching Python
        use_structure_extents: args.use_extents,
        interpolation_resolution_mm: interpolation_resolution,
        interpolation_segments_between_planes: args.z_segments,
        thickness_override_mm: None,
        memmap_rtdose: false,
    };

    // Calculate DVH for each ROI
    let mut dvhs = Vec::new();
    for (i, roi) in rois.iter().enumerate() {
        eprintln!(
            "Calculating DVH for ROI {}/{}: {} ({})",
            i + 1,
            rois.len(),
            roi.id,
            roi.name
        );

        match DvhEngine::calculate_dvh(roi, &dose_grid, &options) {
            Ok(result) => {
                // Convert to JSON format
                let json_dvh = to_json_format(&result);
                dvhs.push(json_dvh);
                eprintln!("  Volume: {:.2} cc", result.total_volume_cc);
                eprintln!("  Mean dose: {:.2} Gy", result.stats.mean_gy);
                eprintln!("  D50: {:.2} Gy", result.stats.d50_gy);
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    // Output results
    let output = BatchOutput { dvhs };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{}", json);

    eprintln!("Successfully computed DVH for {} ROIs", output.dvhs.len());
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
