// Simple test to verify DICOM reading works
use dicom::object::open_file;

fn main() {
    // Test opening a DICOM file
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dicom_file>", args[0]);
        std::process::exit(1);
    }

    match open_file(&args[1]) {
        Ok(obj) => {
            println!("Successfully opened DICOM file");

            // Try to get modality
            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::MODALITY) {
                if let Ok(modality) = elem.to_str() {
                    println!("Modality: {}", modality);
                }
            }

            // For RTDOSE, try to get basic info
            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::ROWS) {
                if let Ok(rows) = elem.to_int::<i32>() {
                    println!("Rows: {}", rows);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::COLUMNS) {
                if let Ok(cols) = elem.to_int::<i32>() {
                    println!("Columns: {}", cols);
                }
            }
        }
        Err(e) => {
            eprintln!("Error opening DICOM file: {}", e);
            std::process::exit(1);
        }
    }
}
