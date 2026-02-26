use aitrium_dvh::dicom_parser::{parse_rtdose, parse_rtstruct};
use aitrium_dvh::dicom_simple::find_dicom_files;
use aitrium_dvh::geometry::PolygonMask;
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

    // Parse files
    let dose_grid = parse_rtdose(&rtdose_path)?;
    let rois = parse_rtstruct(&rtstruct_path)?;

    // Find "peau" ROI
    if let Some(roi) = rois.iter().find(|r| r.name.to_lowercase().contains("peau")) {
        println!("Testing ROI: {}", roi.name);

        // Get first plane
        if let Some((z, contours)) = roi.planes.iter().next() {
            println!("Plane at z={:.2}", z.0);

            // Create mask
            let contour_vecs: Vec<Vec<[f64; 2]>> =
                contours.iter().map(|c| c.points.clone()).collect();

            let mask = PolygonMask::create_plane_mask(
                &contour_vecs,
                &dose_grid.col_lut,
                &dose_grid.row_lut,
                dose_grid.x_lut_index,
            );

            // Count points in mask
            let points_in_mask = mask.iter().filter(|&&v| v).count();
            println!("Points in mask: {}/{}", points_in_mask, mask.len());

            // Find bounding box of mask
            let mut min_r = usize::MAX;
            let mut max_r = 0;
            let mut min_c = usize::MAX;
            let mut max_c = 0;

            for ((r, c), &val) in mask.indexed_iter() {
                if val {
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                    min_c = min_c.min(c);
                    max_c = max_c.max(c);
                }
            }

            if points_in_mask > 0 {
                println!(
                    "Mask bounding box: rows {}-{}, cols {}-{}",
                    min_r, max_r, min_c, max_c
                );

                // Get dose plane at this Z
                let z_index = dose_grid
                    .grid_frame_offset_vector_mm
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &z_offset)| {
                        let z_pos = dose_grid.image_position_patient[2] + z_offset;
                        ((z_pos - z.0) * 1000.0) as i64
                    })
                    .map(|(idx, _)| idx);

                if let Some(z_idx) = z_index {
                    println!("Using dose plane at index {}", z_idx);

                    // Get dose values at masked points
                    if let aitrium_dvh::types::DoseBacking::Owned(ref dose_3d) = dose_grid.dose_3d {
                        let mut dose_values = Vec::new();

                        for ((r, c), &in_mask) in mask.indexed_iter() {
                            if in_mask {
                                let dose_val = dose_3d[[z_idx, r, c]];
                                if dose_val > 0.0 {
                                    dose_values.push(dose_val);
                                }
                            }
                        }

                        if !dose_values.is_empty() {
                            dose_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let min_dose = dose_values[0];
                            let max_dose = dose_values[dose_values.len() - 1];
                            let median_dose = dose_values[dose_values.len() / 2];

                            println!(
                                "Dose statistics for {} points with dose > 0:",
                                dose_values.len()
                            );
                            println!(
                                "  Min: {:.2} (scaled: {:.4} Gy)",
                                min_dose,
                                min_dose as f64 * dose_grid.scale_to_gy
                            );
                            println!(
                                "  Max: {:.2} (scaled: {:.4} Gy)",
                                max_dose,
                                max_dose as f64 * dose_grid.scale_to_gy
                            );
                            println!(
                                "  Median: {:.2} (scaled: {:.4} Gy)",
                                median_dose,
                                median_dose as f64 * dose_grid.scale_to_gy
                            );
                        } else {
                            println!("No non-zero dose values found in masked region!");

                            // Check dose values in the whole plane
                            let mut plane_doses = Vec::new();
                            for r in 0..dose_grid.rows {
                                for c in 0..dose_grid.cols {
                                    let val = dose_3d[[z_idx, r, c]];
                                    if val > 0.0 {
                                        plane_doses.push(val);
                                    }
                                }
                            }
                            println!("Non-zero doses in entire plane: {}", plane_doses.len());
                        }
                    }
                }
            } else {
                println!("WARNING: Mask is empty!");
            }
        }
    }

    Ok(())
}
