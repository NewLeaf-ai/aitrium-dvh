use aitrium_dvh::dicom_parser::{parse_rtdose, parse_rtstruct};
use aitrium_dvh::dicom_simple::find_dicom_files;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dicom_directory>", args[0]);
        std::process::exit(1);
    }

    let dicom_dir = PathBuf::from(&args[1]);

    // Find DICOM files
    let (rtstruct_path, rtdose_path) = find_dicom_files(&dicom_dir)?;

    // Parse RTDOSE
    let start = Instant::now();
    let dose_grid = parse_rtdose(&rtdose_path)?;
    println!("RTDOSE parsing took: {:?}", start.elapsed());
    println!(
        "Dose grid: {}x{}x{} frames",
        dose_grid.rows,
        dose_grid.cols,
        dose_grid.grid_frame_offset_vector_mm.len()
    );

    // Parse RTSTRUCT
    let start = Instant::now();
    let rois = parse_rtstruct(&rtstruct_path)?;
    println!("RTSTRUCT parsing took: {:?}", start.elapsed());
    println!("Found {} ROIs", rois.len());

    // Test mask creation for first ROI with contours
    for roi in &rois {
        if !roi.planes.is_empty() {
            println!("\nTesting mask creation for ROI: {}", roi.name);

            // Get first plane with contours
            if let Some((z, contours)) = roi.planes.iter().next() {
                println!("  Plane at z={:.2} has {} contours", z.0, contours.len());

                let start = Instant::now();

                // Test mask creation
                use aitrium_dvh::geometry::PolygonMask;
                let contour_vecs: Vec<Vec<[f64; 2]>> =
                    contours.iter().map(|c| c.points.clone()).collect();

                let mask = PolygonMask::create_plane_mask(
                    &contour_vecs,
                    &dose_grid.col_lut,
                    &dose_grid.row_lut,
                    dose_grid.x_lut_index,
                );

                let elapsed = start.elapsed();
                let points_in_mask = mask.iter().filter(|&&v| v).count();

                println!("  Mask creation took: {:?}", elapsed);
                println!("  Points in mask: {}/{}", points_in_mask, mask.len());

                if elapsed.as_secs() > 1 {
                    println!("  WARNING: Mask creation is slow!");
                }

                break; // Just test first ROI with contours
            }
        }
    }

    Ok(())
}
