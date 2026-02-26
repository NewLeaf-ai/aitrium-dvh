use crate::types::{DoseBacking, DoseGrid, DvhError};
use dicom_object::{open_file, DefaultDicomObject, FileDicomObject, InMemDicomObject};
use dicom_object::Tag;
use ndarray::{Array2, Array3};
use std::path::Path;

// Define commonly used DICOM tags
const ROWS: Tag = Tag(0x0028, 0x0010);
const COLUMNS: Tag = Tag(0x0028, 0x0011);
const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
const PIXEL_SPACING: Tag = Tag(0x0028, 0x0030);
const IMAGE_POSITION_PATIENT: Tag = Tag(0x0020, 0x0032);
const IMAGE_ORIENTATION_PATIENT: Tag = Tag(0x0020, 0x0037);
const DOSE_GRID_SCALING: Tag = Tag(0x3004, 0x000E);
const GRID_FRAME_OFFSET_VECTOR: Tag = Tag(0x3004, 0x000C);
const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);

pub struct DoseParser;

impl DoseParser {
    /// Parse RTDOSE file
    pub fn parse_file(path: impl AsRef<Path>, use_memmap: bool) -> Result<DoseGrid, DvhError> {
        let obj = open_file(path)
            .map_err(|e| DvhError::DicomError(format!("Failed to open RTDOSE: {}", e)))?;
        
        Self::parse_object(&obj, use_memmap)
    }
    
    /// Parse RTDOSE from DICOM object
    pub fn parse_object(obj: &FileDicomObject<InMemDicomObject>, _use_memmap: bool) -> Result<DoseGrid, DvhError> {
        // Get grid dimensions
        let rows = obj
            .element(ROWS)
            .map_err(|_| DvhError::DoseGridError("Missing Rows".to_string()))?
            .to_int()
            .map_err(|_| DvhError::DoseGridError("Invalid Rows".to_string()))? as usize;
        
        let cols = obj
            .element(COLUMNS)
            .map_err(|_| DvhError::DoseGridError("Missing Columns".to_string()))?
            .to_int()
            .map_err(|_| DvhError::DoseGridError("Invalid Columns".to_string()))? as usize;
        
        let number_of_frames = obj
            .element(NUMBER_OF_FRAMES)
            .map_err(|_| DvhError::DoseGridError("Missing NumberOfFrames".to_string()))?
            .to_int()
            .map_err(|_| DvhError::DoseGridError("Invalid NumberOfFrames".to_string()))? as usize;
        
        // Get pixel spacing
        let pixel_spacing = obj
            .element(PIXEL_SPACING)
            .map_err(|_| DvhError::DoseGridError("Missing PixelSpacing".to_string()))?
            .to_multi_float64()
            .map_err(|_| DvhError::DoseGridError("Invalid PixelSpacing".to_string()))?;
        
        if pixel_spacing.len() != 2 {
            return Err(DvhError::DoseGridError("PixelSpacing must have 2 values".to_string()));
        }
        
        let pixel_spacing_row_mm = pixel_spacing[0];
        let pixel_spacing_col_mm = pixel_spacing[1];
        
        // Get position and orientation
        let image_position_patient = obj
            .element(IMAGE_POSITION_PATIENT)
            .map_err(|_| DvhError::DoseGridError("Missing ImagePositionPatient".to_string()))?
            .to_multi_float64()
            .map_err(|_| DvhError::DoseGridError("Invalid ImagePositionPatient".to_string()))?;
        
        if image_position_patient.len() != 3 {
            return Err(DvhError::DoseGridError("ImagePositionPatient must have 3 values".to_string()));
        }
        
        let image_orientation_patient = obj
            .element(IMAGE_ORIENTATION_PATIENT)
            .map_err(|_| DvhError::DoseGridError("Missing ImageOrientationPatient".to_string()))?
            .to_multi_float64()
            .map_err(|_| DvhError::DoseGridError("Invalid ImageOrientationPatient".to_string()))?;
        
        if image_orientation_patient.len() != 6 {
            return Err(DvhError::DoseGridError("ImageOrientationPatient must have 6 values".to_string()));
        }
        
        // Get dose scaling factor
        let scale_to_gy = obj
            .element(DOSE_GRID_SCALING)
            .map_err(|_| DvhError::DoseGridError("Missing DoseGridScaling".to_string()))?
            .to_float64()
            .map_err(|_| DvhError::DoseGridError("Invalid DoseGridScaling".to_string()))?;
        
        // Get GridFrameOffsetVector for Z positions
        let grid_frame_offset_vector_mm = obj
            .element(GRID_FRAME_OFFSET_VECTOR)
            .map_err(|_| DvhError::DoseGridError("Missing GridFrameOffsetVector".to_string()))?
            .to_multi_float64()
            .map_err(|_| DvhError::DoseGridError("Invalid GridFrameOffsetVector".to_string()))?;
        
        // Parse pixel data
        let pixel_data = obj
            .element(PIXEL_DATA)
            .map_err(|_| DvhError::DoseGridError("Missing PixelData".to_string()))?
            .to_bytes()
            .map_err(|_| DvhError::DoseGridError("Invalid PixelData".to_string()))?;
        
        // Convert pixel data to 3D array
        let dose_3d = Self::parse_pixel_data(pixel_data, rows, cols, number_of_frames)?;
        
        // Calculate LUTs (matching Python's GetDoseData logic)
        let (col_lut, row_lut, x_lut_index) = Self::calculate_luts(
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
            image_position_patient: [
                image_position_patient[0],
                image_position_patient[1],
                image_position_patient[2],
            ],
            image_orientation_patient: [
                image_orientation_patient[0],
                image_orientation_patient[1],
                image_orientation_patient[2],
                image_orientation_patient[3],
                image_orientation_patient[4],
                image_orientation_patient[5],
            ],
            grid_frame_offset_vector_mm,
            dose_3d: DoseBacking::Owned(dose_3d),
            x_lut_index,
            col_lut,
            row_lut,
        })
    }
    
    /// Parse pixel data into 3D array
    fn parse_pixel_data(
        pixel_data: &[u8],
        rows: usize,
        cols: usize,
        frames: usize,
    ) -> Result<Array3<f32>, DvhError> {
        let expected_size = rows * cols * frames * 4; // 4 bytes per f32
        
        if pixel_data.len() < expected_size {
            return Err(DvhError::DoseGridError(format!(
                "Pixel data size mismatch: expected {} bytes, got {}",
                expected_size,
                pixel_data.len()
            )));
        }
        
        let mut array = Array3::zeros((frames, rows, cols));
        
        for z in 0..frames {
            for y in 0..rows {
                for x in 0..cols {
                    let idx = (z * rows * cols + y * cols + x) * 4;
                    let bytes = [
                        pixel_data[idx],
                        pixel_data[idx + 1],
                        pixel_data[idx + 2],
                        pixel_data[idx + 3],
                    ];
                    array[[z, y, x]] = f32::from_le_bytes(bytes);
                }
            }
        }
        
        Ok(array)
    }
    
    /// Calculate patient coordinate LUTs (matching Python's logic)
    fn calculate_luts(
        rows: usize,
        cols: usize,
        image_position: &[f64],
        image_orientation: &[f64],
        pixel_spacing_row: f64,
        pixel_spacing_col: f64,
    ) -> (Vec<f64>, Vec<f64>, u8) {
        // Extract orientation cosines
        let row_cosines = [image_orientation[0], image_orientation[1], image_orientation[2]];
        let col_cosines = [image_orientation[3], image_orientation[4], image_orientation[5]];
        
        // Determine which axis is X (matching Python's x_lut_index logic)
        // If row cosines are primarily in X direction, x_lut_index = 0 (columns map to X)
        // Otherwise x_lut_index = 1 (rows map to X)
        let x_lut_index = if row_cosines[0].abs() > col_cosines[0].abs() {
            0 // X values across columns
        } else {
            1 // X values across rows (decubitus)
        };
        
        // Generate column LUT (patient coordinates)
        let mut col_lut = Vec::with_capacity(cols);
        for i in 0..cols {
            let x = image_position[0] + i as f64 * pixel_spacing_col * col_cosines[0];
            let y = image_position[1] + i as f64 * pixel_spacing_col * col_cosines[1];
            let _z = image_position[2] + i as f64 * pixel_spacing_col * col_cosines[2];
            
            // Select appropriate coordinate based on orientation
            let coord = if x_lut_index == 0 { x } else { y };
            col_lut.push(coord);
        }
        
        // Generate row LUT (patient coordinates)
        let mut row_lut = Vec::with_capacity(rows);
        for i in 0..rows {
            let x = image_position[0] + i as f64 * pixel_spacing_row * row_cosines[0];
            let y = image_position[1] + i as f64 * pixel_spacing_row * row_cosines[1];
            let _z = image_position[2] + i as f64 * pixel_spacing_row * row_cosines[2];
            
            // Select appropriate coordinate based on orientation
            let coord = if x_lut_index == 0 { y } else { x };
            row_lut.push(coord);
        }
        
        (col_lut, row_lut, x_lut_index)
    }
    
    /// Get dose plane at a specific Z position
    pub fn get_dose_plane(dose_grid: &DoseGrid, z_mm: f64) -> Option<Array2<f32>> {
        // Find closest Z plane (temporarily without orientation fix for debugging)
        let z_index = dose_grid.grid_frame_offset_vector_mm
            .iter()
            .enumerate()
            .min_by_key(|(_, &z_offset)| {
                let z_pos = dose_grid.image_position_patient[2] + z_offset;
                ((z_pos - z_mm) * 1000.0) as i64 // Convert to integer for comparison
            })
            .map(|(idx, _)| idx)?;
        
        dose_grid.dose_3d.get_plane(z_index)
    }
}