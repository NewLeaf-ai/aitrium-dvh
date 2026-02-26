use clap::Parser;
use dicom::core::DataElement;
use dicom::dictionary_std::tags;
use dicom::object::open_file;
/// Debug pixel data reading from RTDOSE
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to RTDOSE file
    rtdose_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Analyzing RTDOSE file: {}", args.rtdose_path.display());

    let obj = open_file(&args.rtdose_path)?;

    // Get pixel data metadata
    let bits_allocated = obj
        .element(tags::BITS_ALLOCATED)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(0);

    let bits_stored = obj
        .element(tags::BITS_STORED)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(bits_allocated);

    let high_bit = obj
        .element(tags::HIGH_BIT)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(bits_stored - 1);

    let pixel_representation = obj
        .element(tags::PIXEL_REPRESENTATION)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(0);

    let rows = obj
        .element(tags::ROWS)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(0);

    let cols = obj
        .element(tags::COLUMNS)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(0);

    let frames = obj
        .element(tags::NUMBER_OF_FRAMES)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(1);

    let dose_grid_scaling = obj
        .element(tags::DOSE_GRID_SCALING)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);

    println!("\n=== Pixel Data Metadata ===");
    println!("Bits Allocated: {}", bits_allocated);
    println!("Bits Stored: {}", bits_stored);
    println!("High Bit: {}", high_bit);
    println!(
        "Pixel Representation: {} ({})",
        pixel_representation,
        if pixel_representation == 0 {
            "unsigned"
        } else {
            "signed"
        }
    );
    println!("Dimensions: {} x {} x {} frames", rows, cols, frames);
    println!("Dose Grid Scaling: {}", dose_grid_scaling);

    // Check for scaling/intercept tags
    if let Ok(rescale_intercept) = obj.element(tags::RESCALE_INTERCEPT) {
        println!("Rescale Intercept: {:?}", rescale_intercept.to_str());
    }

    if let Ok(rescale_slope) = obj.element(tags::RESCALE_SLOPE) {
        println!("Rescale Slope: {:?}", rescale_slope.to_str());
    }

    // Get pixel data
    if let Ok(pixel_data_elem) = obj.element(tags::PIXEL_DATA) {
        let pixel_bytes = pixel_data_elem.to_bytes()?;

        println!("\n=== Pixel Data Analysis ===");
        println!("Total bytes: {}", pixel_bytes.len());
        println!(
            "Expected bytes (frames * rows * cols * bytes_per_pixel): {}",
            frames * rows * cols * (bits_allocated / 8)
        );

        // Sample first few pixels
        println!("\n=== First 5 Pixels (Raw Bytes) ===");
        for i in 0..5.min((pixel_bytes.len() / (bits_allocated as usize / 8))) {
            let idx = i * (bits_allocated as usize / 8);

            match bits_allocated {
                16 => {
                    let raw_value = u16::from_le_bytes([pixel_bytes[idx], pixel_bytes[idx + 1]]);

                    // Apply bits stored mask if needed
                    let mask = (1u16 << bits_stored) - 1;
                    let masked_value = raw_value & mask;

                    // Check if sign extension is needed
                    let final_value = if pixel_representation == 1
                        && (masked_value & (1u16 << (bits_stored - 1))) != 0
                    {
                        // Sign extend
                        let sign_bit = 1u16 << (bits_stored - 1);
                        let sign_extension = !mask;
                        (masked_value | sign_extension) as i16 as f32
                    } else {
                        masked_value as f32
                    };

                    let dose_gy = final_value as f64 * dose_grid_scaling;
                    let dose_cgy = dose_gy * 100.0;

                    println!("Pixel {}: raw=0x{:04X} ({}), masked=0x{:04X} ({}), final={:.2}, dose={:.4} Gy ({:.2} cGy)",
                        i, raw_value, raw_value, masked_value, masked_value, final_value, dose_gy, dose_cgy);
                }
                32 => {
                    let raw_value = u32::from_le_bytes([
                        pixel_bytes[idx],
                        pixel_bytes[idx + 1],
                        pixel_bytes[idx + 2],
                        pixel_bytes[idx + 3],
                    ]);

                    // Apply bits stored mask if needed
                    let mask = if bits_stored < 32 {
                        (1u32 << bits_stored) - 1
                    } else {
                        0xFFFFFFFF
                    };
                    let masked_value = raw_value & mask;

                    // Check if sign extension is needed
                    let final_value = if pixel_representation == 1
                        && bits_stored < 32
                        && (masked_value & (1u32 << (bits_stored - 1))) != 0
                    {
                        // Sign extend
                        let sign_bit = 1u32 << (bits_stored - 1);
                        let sign_extension = !mask;
                        (masked_value | sign_extension) as i32 as f32
                    } else {
                        masked_value as f32
                    };

                    let dose_gy = final_value as f64 * dose_grid_scaling;
                    let dose_cgy = dose_gy * 100.0;

                    println!("Pixel {}: raw=0x{:08X} ({}), masked=0x{:08X} ({}), final={:.2}, dose={:.4} Gy ({:.2} cGy)",
                        i, raw_value, raw_value, masked_value, masked_value, final_value, dose_gy, dose_cgy);
                }
                _ => {
                    println!("Unsupported bits allocated: {}", bits_allocated);
                }
            }
        }

        // Find max value
        println!("\n=== Finding Max Dose ===");
        let mut max_value = 0.0f32;
        let bytes_per_pixel = bits_allocated as usize / 8;
        let total_pixels = pixel_bytes.len() / bytes_per_pixel;

        for i in 0..total_pixels {
            let idx = i * bytes_per_pixel;

            let value = match bits_allocated {
                16 => {
                    let raw = u16::from_le_bytes([pixel_bytes[idx], pixel_bytes[idx + 1]]);
                    let mask = (1u16 << bits_stored) - 1;
                    let masked = raw & mask;

                    if pixel_representation == 1 && (masked & (1u16 << (bits_stored - 1))) != 0 {
                        let sign_extension = !mask;
                        (masked | sign_extension) as i16 as f32
                    } else {
                        masked as f32
                    }
                }
                32 => {
                    let raw = u32::from_le_bytes([
                        pixel_bytes[idx],
                        pixel_bytes[idx + 1],
                        pixel_bytes[idx + 2],
                        pixel_bytes[idx + 3],
                    ]);

                    let mask = if bits_stored < 32 {
                        (1u32 << bits_stored) - 1
                    } else {
                        0xFFFFFFFF
                    };
                    let masked = raw & mask;

                    if pixel_representation == 1
                        && bits_stored < 32
                        && (masked & (1u32 << (bits_stored - 1))) != 0
                    {
                        let sign_extension = !mask;
                        (masked | sign_extension) as i32 as f32
                    } else {
                        masked as f32
                    }
                }
                _ => 0.0,
            };

            if value > max_value {
                max_value = value;
            }
        }

        println!("Max pixel value: {}", max_value);
        println!(
            "Max dose: {:.4} Gy ({:.2} cGy)",
            max_value as f64 * dose_grid_scaling,
            max_value as f64 * dose_grid_scaling * 100.0
        );
        println!(
            "Max dose bins (Python formula): {}",
            (max_value as f64 * dose_grid_scaling * 100.0) as u32 + 1
        );
    }

    Ok(())
}
