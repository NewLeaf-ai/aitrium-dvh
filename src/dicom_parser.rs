/// Full DICOM parser implementation for RTSTRUCT and RTDOSE
use crate::types::{Contour, ContourType, DoseBacking, DoseGrid, DvhError, OrderedFloat, Roi};
use dicom::dictionary_std::tags;
use dicom::object::{open_file, InMemDicomObject};
use ndarray::Array3;
use std::collections::BTreeMap;
use std::path::Path;

/// Parse RTSTRUCT file with full contour data
pub fn parse_rtstruct(path: &Path) -> Result<Vec<Roi>, DvhError> {
    let obj = open_file(path)
        .map_err(|e| DvhError::DicomError(format!("Failed to open RTSTRUCT: {}", e)))?;

    let mut rois = Vec::new();
    let mut roi_map = std::collections::HashMap::new();

    // First, get ROI information from StructureSetROISequence
    if let Ok(roi_seq_elem) = obj.element(tags::STRUCTURE_SET_ROI_SEQUENCE) {
        if let Some(roi_items) = roi_seq_elem.items() {
            for roi_item in roi_items {
                let roi_number = roi_item
                    .element(tags::ROI_NUMBER)
                    .ok()
                    .and_then(|e| e.to_int::<i32>().ok())
                    .unwrap_or(0);

                let roi_name = roi_item
                    .element(tags::ROI_NAME)
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("ROI_{}", roi_number));

                roi_map.insert(roi_number, roi_name);
            }
        }
    }

    // Now get contour data from ROIContourSequence
    if let Ok(contour_seq_elem) = obj.element(tags::ROI_CONTOUR_SEQUENCE) {
        if let Some(contour_items) = contour_seq_elem.items() {
            for contour_item in contour_items {
                // Get the referenced ROI number
                let roi_number = contour_item
                    .element(tags::REFERENCED_ROI_NUMBER)
                    .ok()
                    .and_then(|e| e.to_int::<i32>().ok())
                    .unwrap_or(0);

                let roi_name = roi_map
                    .get(&roi_number)
                    .cloned()
                    .unwrap_or_else(|| format!("ROI_{}", roi_number));

                // Parse contour planes
                let planes = parse_contour_planes(contour_item)?;

                // Calculate thickness from plane spacing
                let thickness_mm = calculate_plane_thickness(&planes);

                rois.push(Roi {
                    id: roi_number,
                    name: roi_name,
                    planes,
                    thickness_mm,
                });
            }
        }
    }

    eprintln!("Parsed {} ROIs from RTSTRUCT", rois.len());
    Ok(rois)
}

/// Parse contour planes from ROIContourSequence item
fn parse_contour_planes(
    contour_item: &InMemDicomObject,
) -> Result<BTreeMap<OrderedFloat, Vec<Contour>>, DvhError> {
    let mut planes = BTreeMap::new();

    if let Ok(contour_seq_elem) = contour_item.element(tags::CONTOUR_SEQUENCE) {
        if let Some(contour_items) = contour_seq_elem.items() {
            for contour in contour_items {
                // Check contour type (should be CLOSED_PLANAR)
                let geometric_type = contour
                    .element(tags::CONTOUR_GEOMETRIC_TYPE)
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "CLOSED_PLANAR".to_string());

                if geometric_type != "CLOSED_PLANAR" {
                    continue; // Skip non-planar contours
                }

                // Get contour data (x,y,z coordinates)
                if let Ok(contour_data_elem) = contour.element(tags::CONTOUR_DATA) {
                    // Try to get as string first (common format)
                    let coords = if let Ok(data_str) = contour_data_elem.to_str() {
                        data_str
                            .split('\\')
                            .filter_map(|s| s.parse::<f64>().ok())
                            .collect::<Vec<_>>()
                    } else {
                        // Try to get as multi_float64
                        contour_data_elem
                            .to_multi_float64()
                            .ok()
                            .unwrap_or_default()
                    };

                    if coords.len() % 3 != 0 || coords.is_empty() {
                        continue; // Invalid contour data
                    }

                    // Extract points
                    let mut points = Vec::new();
                    let mut z_position = None;

                    for chunk in coords.chunks_exact(3) {
                        let x = chunk[0];
                        let y = chunk[1];
                        let z = chunk[2];

                        if z_position.is_none() {
                            z_position = Some(z);
                        }

                        points.push([x, y]);
                    }

                    if let Some(z) = z_position {
                        // Determine contour type based on whether it's a hole
                        let contour_type = if contour.element(tags::CONTOUR_IMAGE_SEQUENCE).is_ok()
                        {
                            ContourType::External
                        } else {
                            ContourType::Cavity
                        };

                        planes
                            .entry(OrderedFloat(z))
                            .or_insert_with(Vec::new)
                            .push(Contour {
                                points,
                                contour_type,
                            });
                    }
                }
            }
        }
    }

    Ok(planes)
}

/// Calculate structure thickness from plane spacing
fn calculate_plane_thickness(planes: &BTreeMap<OrderedFloat, Vec<Contour>>) -> f64 {
    let z_positions: Vec<f64> = planes.keys().map(|k| k.0).collect();

    if z_positions.len() < 2 {
        // Default thickness if only one plane
        return 2.5;
    }

    // Calculate median spacing
    let mut spacings = Vec::new();
    for i in 1..z_positions.len() {
        spacings.push((z_positions[i] - z_positions[i - 1]).abs());
    }

    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if spacings.len() % 2 == 0 {
        (spacings[spacings.len() / 2 - 1] + spacings[spacings.len() / 2]) / 2.0
    } else {
        spacings[spacings.len() / 2]
    }
}

/// Parse RTDOSE file with actual pixel data
pub fn parse_rtdose(path: &Path) -> Result<DoseGrid, DvhError> {
    let obj = open_file(path)
        .map_err(|e| DvhError::DicomError(format!("Failed to open RTDOSE: {}", e)))?;

    // Get dimensions
    let rows =
        obj.element(tags::ROWS)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok())
            .ok_or_else(|| DvhError::DoseGridError("Missing Rows".to_string()))? as usize;

    let cols = obj
        .element(tags::COLUMNS)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .ok_or_else(|| DvhError::DoseGridError("Missing Columns".to_string()))?
        as usize;

    let frames = obj
        .element(tags::NUMBER_OF_FRAMES)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(1) as usize;

    eprintln!("RTDOSE dimensions: {}x{}x{} frames", rows, cols, frames);

    // Get pixel spacing
    let (pixel_spacing_row_mm, pixel_spacing_col_mm) = parse_pixel_spacing(&obj)?;

    // Get position and orientation
    let image_position_patient = parse_image_position(&obj)?;
    let image_orientation_patient = parse_image_orientation(&obj)?;

    // Get patient position (e.g., "HFS", "HFP", "FFS", "FFP")
    let patient_position = obj
        .element(tags::PATIENT_POSITION)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string());

    // Get dose scaling factor
    let scale_to_gy = obj
        .element(tags::DOSE_GRID_SCALING)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| DvhError::DoseGridError("Missing DoseGridScaling".to_string()))?;

    // Get grid frame offsets
    let grid_frame_offset_vector_mm = parse_grid_frame_offsets(&obj, frames)?;

    // Parse pixel data
    let dose_3d = parse_dose_pixel_data(&obj, rows, cols, frames)?;

    // Calculate LUTs
    let (col_lut, row_lut, x_lut_index) = calculate_luts(
        rows,
        cols,
        &image_position_patient,
        &image_orientation_patient,
        pixel_spacing_row_mm,
        pixel_spacing_col_mm,
    );

    Ok(DoseGrid {
        scale_to_gy,
        rows,
        cols,
        pixel_spacing_row_mm,
        pixel_spacing_col_mm,
        image_position_patient,
        image_orientation_patient,
        grid_frame_offset_vector_mm,
        dose_3d: DoseBacking::Owned(dose_3d),
        x_lut_index,
        col_lut,
        row_lut,
        patient_position,
    })
}

/// Parse pixel spacing from DICOM object
fn parse_pixel_spacing(obj: &InMemDicomObject) -> Result<(f64, f64), DvhError> {
    if let Ok(ps_elem) = obj.element(tags::PIXEL_SPACING) {
        // Try as string first (most common)
        if let Ok(ps_str) = ps_elem.to_str() {
            let parts: Vec<f64> = ps_str.split('\\').filter_map(|s| s.parse().ok()).collect();
            if parts.len() >= 2 {
                return Ok((parts[0], parts[1]));
            }
        }
        // Try as multi_float64
        if let Ok(ps_vec) = ps_elem.to_multi_float64() {
            if ps_vec.len() >= 2 {
                return Ok((ps_vec[0], ps_vec[1]));
            }
        }
    }
    Err(DvhError::DoseGridError("Invalid PixelSpacing".to_string()))
}

/// Parse image position from DICOM object
fn parse_image_position(obj: &InMemDicomObject) -> Result<[f64; 3], DvhError> {
    if let Ok(pos_elem) = obj.element(tags::IMAGE_POSITION_PATIENT) {
        // Try as string first
        if let Ok(pos_str) = pos_elem.to_str() {
            let parts: Vec<f64> = pos_str.split('\\').filter_map(|s| s.parse().ok()).collect();
            if parts.len() >= 3 {
                return Ok([parts[0], parts[1], parts[2]]);
            }
        }
        // Try as multi_float64
        if let Ok(pos_vec) = pos_elem.to_multi_float64() {
            if pos_vec.len() >= 3 {
                return Ok([pos_vec[0], pos_vec[1], pos_vec[2]]);
            }
        }
    }
    Err(DvhError::DoseGridError(
        "Invalid ImagePositionPatient".to_string(),
    ))
}

/// Parse image orientation from DICOM object
fn parse_image_orientation(obj: &InMemDicomObject) -> Result<[f64; 6], DvhError> {
    if let Ok(orient_elem) = obj.element(tags::IMAGE_ORIENTATION_PATIENT) {
        // Try as string first
        if let Ok(orient_str) = orient_elem.to_str() {
            let parts: Vec<f64> = orient_str
                .split('\\')
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() >= 6 {
                return Ok([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
            }
        }
        // Try as multi_float64
        if let Ok(orient_vec) = orient_elem.to_multi_float64() {
            if orient_vec.len() >= 6 {
                return Ok([
                    orient_vec[0],
                    orient_vec[1],
                    orient_vec[2],
                    orient_vec[3],
                    orient_vec[4],
                    orient_vec[5],
                ]);
            }
        }
    }
    Err(DvhError::DoseGridError(
        "Invalid ImageOrientationPatient".to_string(),
    ))
}

/// Parse grid frame offsets
fn parse_grid_frame_offsets(obj: &InMemDicomObject, frames: usize) -> Result<Vec<f64>, DvhError> {
    if let Ok(offset_elem) = obj.element(tags::GRID_FRAME_OFFSET_VECTOR) {
        // Try as string first
        if let Ok(offset_str) = offset_elem.to_str() {
            let offsets: Vec<f64> = offset_str
                .split('\\')
                .filter_map(|s| s.parse().ok())
                .collect();
            if !offsets.is_empty() {
                return Ok(offsets);
            }
        }
        // Try as multi_float64
        if let Ok(offset_vec) = offset_elem.to_multi_float64() {
            if !offset_vec.is_empty() {
                return Ok(offset_vec);
            }
        }
    }

    // Default: evenly spaced frames
    Ok((0..frames).map(|i| i as f64 * 3.0).collect())
}

/// Decode a 16-bit sample with proper BitsStored masking and sign extension
fn decode_sample_u16(raw: u16, bits_stored: usize, pixel_representation: i32) -> f32 {
    let mask: u16 = if bits_stored >= 16 {
        u16::MAX
    } else {
        (1u16 << bits_stored) - 1
    };
    let masked = raw & mask;

    if pixel_representation == 0 {
        masked as f32
    } else {
        // signed - apply sign extension if needed
        let sign_bit = 1u16 << (bits_stored.saturating_sub(1));
        let signed = if bits_stored < 16 && (masked & sign_bit) != 0 {
            (masked | !mask) as i16
        } else {
            masked as i16
        };
        signed as f32
    }
}

/// Decode a 32-bit sample with proper BitsStored masking and sign extension
fn decode_sample_u32(raw: u32, bits_stored: usize, pixel_representation: i32) -> f32 {
    let mask: u32 = if bits_stored >= 32 {
        u32::MAX
    } else {
        (1u32 << bits_stored) - 1
    };
    let masked = raw & mask;

    if pixel_representation == 0 {
        masked as f32
    } else {
        // signed - apply sign extension if needed
        let sign_bit = 1u32 << (bits_stored.saturating_sub(1));
        let signed = if bits_stored < 32 && (masked & sign_bit) != 0 {
            (masked | !mask) as i32
        } else {
            masked as i32
        };
        signed as f32
    }
}

/// Parse dose pixel data
fn parse_dose_pixel_data(
    obj: &InMemDicomObject,
    rows: usize,
    cols: usize,
    frames: usize,
) -> Result<Array3<f32>, DvhError> {
    // Get bits allocated to determine data type
    let bits_allocated = obj
        .element(tags::BITS_ALLOCATED)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(32) as usize;

    // Get bits stored (actual bits used)
    let bits_stored = obj
        .element(tags::BITS_STORED)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .map(|v| v as usize)
        .unwrap_or(bits_allocated);

    // Get pixel representation (0 = unsigned, 1 = signed)
    let pixel_representation = obj
        .element(tags::PIXEL_REPRESENTATION)
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .unwrap_or(0);

    eprintln!(
        "Pixel data: {} bits, representation: {} ({})",
        bits_allocated,
        pixel_representation,
        if pixel_representation == 0 {
            "unsigned"
        } else {
            "signed"
        }
    );

    // Get pixel data
    if let Ok(pixel_data_elem) = obj.element(tags::PIXEL_DATA) {
        // Get raw bytes
        let pixel_bytes = pixel_data_elem
            .to_bytes()
            .map_err(|_| DvhError::DoseGridError("Failed to read pixel data".to_string()))?;

        eprintln!(
            "Pixel data size: {} bytes, expected: {} bytes",
            pixel_bytes.len(),
            frames * rows * cols * (bits_allocated / 8)
        );

        let mut array = Array3::zeros((frames, rows, cols));

        match bits_allocated {
            16 => {
                // 16-bit integers
                if pixel_bytes.len() < frames * rows * cols * 2 {
                    return Err(DvhError::DoseGridError(
                        "Pixel data size mismatch".to_string(),
                    ));
                }

                for z in 0..frames {
                    for y in 0..rows {
                        for x in 0..cols {
                            let idx = (z * rows * cols + y * cols + x) * 2;
                            let raw = u16::from_le_bytes([pixel_bytes[idx], pixel_bytes[idx + 1]]);
                            let value = decode_sample_u16(raw, bits_stored, pixel_representation);
                            array[[z, y, x]] = value;
                        }
                    }
                }
            }
            32 => {
                // 32-bit integers (unsigned for dose data)
                if pixel_bytes.len() < frames * rows * cols * 4 {
                    return Err(DvhError::DoseGridError(
                        "Pixel data size mismatch".to_string(),
                    ));
                }

                for z in 0..frames {
                    for y in 0..rows {
                        for x in 0..cols {
                            let idx = (z * rows * cols + y * cols + x) * 4;
                            let bytes = [
                                pixel_bytes[idx],
                                pixel_bytes[idx + 1],
                                pixel_bytes[idx + 2],
                                pixel_bytes[idx + 3],
                            ];

                            let raw = u32::from_le_bytes(bytes);
                            let value = decode_sample_u32(raw, bits_stored, pixel_representation);
                            array[[z, y, x]] = value;
                        }
                    }
                }
            }
            _ => {
                return Err(DvhError::DoseGridError(format!(
                    "Unsupported bits allocated: {}",
                    bits_allocated
                )));
            }
        }

        Ok(array)
    } else {
        Err(DvhError::DoseGridError("Missing pixel data".to_string()))
    }
}

/// Calculate patient coordinate LUTs matching dicompyler-core exactly
fn calculate_luts(
    rows: usize,
    cols: usize,
    image_position: &[f64; 3],
    image_orientation: &[f64; 6],
    pixel_spacing_row: f64,
    pixel_spacing_col: f64,
) -> (Vec<f64>, Vec<f64>, u8) {
    // Helper function for floating point comparison
    fn isclose(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6
    }

    // Reproduce dicompylercore.dicomparser.x_lut_index()
    fn x_lut_index(ori: &[f64; 6]) -> u8 {
        // non-decubitus -> 0 (X across columns)
        const NON_DECUB: [[f64; 6]; 4] = [
            [1., 0., 0., 0., 1., 0.],   // HFS
            [-1., 0., 0., 0., -1., 0.], // HFP
            [-1., 0., 0., 0., 1., 0.],  // FFS
            [1., 0., 0., 0., -1., 0.],  // FFP
        ];
        for o in NON_DECUB.iter() {
            if (0..6).all(|i| isclose(ori[i], o[i])) {
                return 0;
            }
        }
        // decubitus -> 1 (X along rows)
        const DECUB: [[f64; 6]; 4] = [
            [0., -1., 0., 1., 0., 0.],  // HFDL
            [0., 1., 0., -1., 0., 0.],  // HFDR
            [0., 1., 0., 1., 0., 0.],   // FFDL
            [0., -1., 0., -1., 0., 0.], // FFDR
        ];
        for o in DECUB.iter() {
            if (0..6).all(|i| isclose(ori[i], o[i])) {
                return 1;
            }
        }
        // Fallback to 0 (matches python raising NotImplemented then defaulting in LUT build)
        0
    }

    // Build LUTs exactly like GetPatientToPixelLUT (matrix form)
    let ori = image_orientation;
    let first_x = image_position[0];
    let first_y = image_position[1];
    let drow = pixel_spacing_row;
    let dcol = pixel_spacing_col;

    // Build transformation matrix
    let m = [
        [ori[0] * dcol, ori[3] * drow, 0.0, first_x],
        [ori[1] * dcol, ori[4] * drow, 0.0, first_y],
        [ori[2] * dcol, ori[5] * drow, 0.0, image_position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Calculate last position using matrix
    let last = |cols: usize, rows: usize| -> (f64, f64) {
        let c = (cols as f64 - 1.0, rows as f64 - 1.0);
        let last_x = m[0][0] * c.0 + m[0][1] * c.1 + m[0][3];
        let last_y = m[1][0] * c.0 + m[1][1] * c.1 + m[1][3];
        (last_x, last_y)
    };
    let (last_x, last_y) = last(cols, rows);
    let index = x_lut_index(image_orientation);

    // Linear interpolation function
    let linspace = |start: f64, end: f64, n: usize| -> Vec<f64> {
        if n <= 1 {
            return vec![start];
        }
        (0..n)
            .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
            .collect()
    };

    // index==0 => col=X(first_x→last_x), row=Y(first_y→last_y)
    // index==1 => col=Y(first_y→last_y), row=X(first_x→last_x)
    let (col_lut, row_lut) = if index == 0 {
        (
            linspace(first_x, last_x, cols),
            linspace(first_y, last_y, rows),
        )
    } else {
        (
            linspace(first_y, last_y, cols),
            linspace(first_x, last_x, rows),
        )
    };

    eprintln!("X LUT index: {} (dicompyler convention)", index);
    eprintln!(
        "Col LUT range: {:.2} to {:.2}",
        col_lut.first().unwrap_or(&0.0),
        col_lut.last().unwrap_or(&0.0)
    );
    eprintln!(
        "Row LUT range: {:.2} to {:.2}",
        row_lut.first().unwrap_or(&0.0),
        row_lut.last().unwrap_or(&0.0)
    );

    (col_lut, row_lut, index)
}
