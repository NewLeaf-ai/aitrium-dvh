use aitrium_dvh::dicom_parser::parse_rtdose;
use aitrium_dvh::dicom_simple::find_dicom_files;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dicom_directory>", args[0]);
        std::process::exit(1);
    }

    let dicom_dir = PathBuf::from(&args[1]);
    let (_, rtdose_path) = find_dicom_files(&dicom_dir)?;

    let dose_grid = parse_rtdose(&rtdose_path)?;

    println!("Dose grid Z range:");
    println!(
        "  Image position Z: {:.2}",
        dose_grid.image_position_patient[2]
    );
    println!(
        "  Number of frames: {}",
        dose_grid.grid_frame_offset_vector_mm.len()
    );

    if !dose_grid.grid_frame_offset_vector_mm.is_empty() {
        let first_offset = dose_grid.grid_frame_offset_vector_mm[0];
        let last_offset =
            dose_grid.grid_frame_offset_vector_mm[dose_grid.grid_frame_offset_vector_mm.len() - 1];

        let z_min = dose_grid.image_position_patient[2] + first_offset;
        let z_max = dose_grid.image_position_patient[2] + last_offset;

        println!("  First offset: {:.2} mm", first_offset);
        println!("  Last offset: {:.2} mm", last_offset);
        println!("  Z min: {:.2} mm", z_min);
        println!("  Z max: {:.2} mm", z_max);

        // Show first few Z positions
        println!("\n  First 5 Z positions:");
        for i in 0..5.min(dose_grid.grid_frame_offset_vector_mm.len()) {
            let z = dose_grid.image_position_patient[2] + dose_grid.grid_frame_offset_vector_mm[i];
            println!("    Frame {}: z = {:.2} mm", i, z);
        }

        // Show last few Z positions
        println!("\n  Last 5 Z positions:");
        let start = dose_grid
            .grid_frame_offset_vector_mm
            .len()
            .saturating_sub(5);
        for i in start..dose_grid.grid_frame_offset_vector_mm.len() {
            let z = dose_grid.image_position_patient[2] + dose_grid.grid_frame_offset_vector_mm[i];
            println!("    Frame {}: z = {:.2} mm", i, z);
        }
    }

    Ok(())
}
