// Check RTDOSE pixel data format
use dicom::dictionary_std::tags;
use dicom::object::open_file;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <rtdose.dcm>", args[0]);
        std::process::exit(1);
    }

    let obj = open_file(&args[1]).expect("Failed to open RTDOSE");

    // Check pixel data attributes
    if let Ok(bits_alloc) = obj.element(tags::BITS_ALLOCATED) {
        println!("Bits Allocated: {:?}", bits_alloc.to_int::<i32>());
    }

    if let Ok(bits_stored) = obj.element(tags::BITS_STORED) {
        println!("Bits Stored: {:?}", bits_stored.to_int::<i32>());
    }

    if let Ok(high_bit) = obj.element(tags::HIGH_BIT) {
        println!("High Bit: {:?}", high_bit.to_int::<i32>());
    }

    if let Ok(pixel_repr) = obj.element(tags::PIXEL_REPRESENTATION) {
        println!("Pixel Representation: {:?}", pixel_repr.to_int::<i32>());
    }

    if let Ok(photometric) = obj.element(tags::PHOTOMETRIC_INTERPRETATION) {
        println!("Photometric Interpretation: {:?}", photometric.to_str());
    }

    // Get pixel data size
    if let Ok(pixel_data) = obj.element(tags::PIXEL_DATA) {
        if let Ok(bytes) = pixel_data.to_bytes() {
            println!("Pixel data size: {} bytes", bytes.len());

            // Sample first few values
            if bytes.len() >= 32 {
                println!("\nFirst 16 bytes (hex):");
                for i in 0..16 {
                    print!("{:02x} ", bytes[i]);
                }
                println!();

                // Find first non-zero value
                let mut first_nonzero = None;
                for i in 0..bytes.len() / 4 {
                    let idx = i * 4;
                    if idx + 3 < bytes.len() {
                        let val = u32::from_le_bytes([
                            bytes[idx],
                            bytes[idx + 1],
                            bytes[idx + 2],
                            bytes[idx + 3],
                        ]);
                        if val != 0 {
                            first_nonzero = Some((i, val));
                            break;
                        }
                    }
                }

                if let Some((idx, val)) = first_nonzero {
                    println!(
                        "\nFirst non-zero u32 at index {}: {} (0x{:08x})",
                        idx, val, val
                    );

                    // Show surrounding values
                    println!("Values around first non-zero:");
                    for i in idx.saturating_sub(2)..=(idx + 2).min(bytes.len() / 4 - 1) {
                        let byte_idx = i * 4;
                        let v = u32::from_le_bytes([
                            bytes[byte_idx],
                            bytes[byte_idx + 1],
                            bytes[byte_idx + 2],
                            bytes[byte_idx + 3],
                        ]);
                        println!("  [{}]: {} (0x{:08x})", i, v, v);
                    }
                } else {
                    println!("\nNo non-zero values found in pixel data!");
                }
            }
        }
    }
}
