use aitrium_dvh::dicom_parser::{parse_rtdose, parse_rtstruct};
use aitrium_dvh::dicom_simple::find_dicom_files;
use std::path::PathBuf;

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
    let dose_grid = parse_rtdose(&rtdose_path)?;
    println!("Dose grid info:");
    println!(
        "  Dimensions: {}x{}x{} frames",
        dose_grid.rows,
        dose_grid.cols,
        dose_grid.grid_frame_offset_vector_mm.len()
    );
    println!("  Image position: {:?}", dose_grid.image_position_patient);
    println!(
        "  Pixel spacing: {:.2}mm x {:.2}mm",
        dose_grid.pixel_spacing_row_mm, dose_grid.pixel_spacing_col_mm
    );
    println!("  X LUT index: {}", dose_grid.x_lut_index);

    // Parse RTSTRUCT
    let rois = parse_rtstruct(&rtstruct_path)?;

    // Find an ROI with contours
    for roi in &rois {
        if !roi.planes.is_empty() {
            println!("\nROI: {} (ID {})", roi.name, roi.id);

            // Get first plane
            if let Some((z, contours)) = roi.planes.iter().next() {
                println!("  First plane at z={:.2}", z.0);

                // Get first contour
                if let Some(contour) = contours.first() {
                    println!("  First contour has {} points", contour.points.len());

                    // Show first few points
                    for (i, point) in contour.points.iter().take(3).enumerate() {
                        println!("    Point {}: x={:.2}, y={:.2}", i, point[0], point[1]);
                    }

                    // Check if points are within dose grid bounds
                    let x_min = contour.points.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
                    let x_max = contour.points.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
                    let y_min = contour.points.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
                    let y_max = contour.points.iter().map(|p| p[1]).fold(f64::MIN, f64::max);

                    println!("\n  Contour bounds:");
                    println!("    X: {:.2} to {:.2}", x_min, x_max);
                    println!("    Y: {:.2} to {:.2}", y_min, y_max);

                    println!("\n  Dose grid bounds:");
                    if dose_grid.x_lut_index == 0 {
                        println!(
                            "    X (cols): {:.2} to {:.2}",
                            dose_grid.col_lut.first().unwrap_or(&0.0),
                            dose_grid.col_lut.last().unwrap_or(&0.0)
                        );
                        println!(
                            "    Y (rows): {:.2} to {:.2}",
                            dose_grid.row_lut.first().unwrap_or(&0.0),
                            dose_grid.row_lut.last().unwrap_or(&0.0)
                        );
                    } else {
                        println!(
                            "    X (rows): {:.2} to {:.2}",
                            dose_grid.row_lut.first().unwrap_or(&0.0),
                            dose_grid.row_lut.last().unwrap_or(&0.0)
                        );
                        println!(
                            "    Y (cols): {:.2} to {:.2}",
                            dose_grid.col_lut.first().unwrap_or(&0.0),
                            dose_grid.col_lut.last().unwrap_or(&0.0)
                        );
                    }

                    // Check overlap
                    let x_dose_min = if dose_grid.x_lut_index == 0 {
                        *dose_grid.col_lut.first().unwrap_or(&0.0)
                    } else {
                        *dose_grid.row_lut.first().unwrap_or(&0.0)
                    };
                    let x_dose_max = if dose_grid.x_lut_index == 0 {
                        *dose_grid.col_lut.last().unwrap_or(&0.0)
                    } else {
                        *dose_grid.row_lut.last().unwrap_or(&0.0)
                    };

                    let y_dose_min = if dose_grid.x_lut_index == 0 {
                        *dose_grid.row_lut.first().unwrap_or(&0.0)
                    } else {
                        *dose_grid.col_lut.first().unwrap_or(&0.0)
                    };
                    let y_dose_max = if dose_grid.x_lut_index == 0 {
                        *dose_grid.row_lut.last().unwrap_or(&0.0)
                    } else {
                        *dose_grid.col_lut.last().unwrap_or(&0.0)
                    };

                    let x_overlap = x_min < x_dose_max && x_max > x_dose_min;
                    let y_overlap = y_min < y_dose_max && y_max > y_dose_min;

                    println!("\n  Overlap check:");
                    println!("    X overlap: {}", if x_overlap { "YES" } else { "NO" });
                    println!("    Y overlap: {}", if y_overlap { "YES" } else { "NO" });

                    if !x_overlap || !y_overlap {
                        println!("  WARNING: Contour does not overlap with dose grid!");
                    }
                }

                break;
            }
        }
    }

    Ok(())
}
