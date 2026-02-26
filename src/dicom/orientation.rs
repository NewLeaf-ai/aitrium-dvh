/// Patient orientation detection for DICOM files
use crate::DvhError;

/// Check if the patient orientation is head-first
pub fn is_head_first_orientation(image_orientation: &[f64; 6]) -> bool {
    // Head-first orientations
    const HEAD_FIRST_SUPINE: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    const HEAD_FIRST_PRONE: [f64; 6] = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
    const HEAD_FIRST_DECUBITUS_LEFT: [f64; 6] = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0];
    const HEAD_FIRST_DECUBITUS_RIGHT: [f64; 6] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
    
    // Feet-first orientations
    const FEET_FIRST_DECUBITUS_LEFT: [f64; 6] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    const FEET_FIRST_DECUBITUS_RIGHT: [f64; 6] = [0.0, -1.0, 0.0, -1.0, 0.0, 0.0];
    const FEET_FIRST_PRONE: [f64; 6] = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
    const FEET_FIRST_SUPINE: [f64; 6] = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    
    // Check if orientation matches any head-first orientation (with tolerance)
    let tolerance = 0.01;
    
    let is_close = |a: &[f64; 6], b: &[f64; 6]| -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tolerance)
    };
    
    if is_close(image_orientation, &HEAD_FIRST_SUPINE) ||
       is_close(image_orientation, &HEAD_FIRST_PRONE) ||
       is_close(image_orientation, &HEAD_FIRST_DECUBITUS_LEFT) ||
       is_close(image_orientation, &HEAD_FIRST_DECUBITUS_RIGHT) {
        return true;
    }
    
    if is_close(image_orientation, &FEET_FIRST_DECUBITUS_LEFT) ||
       is_close(image_orientation, &FEET_FIRST_DECUBITUS_RIGHT) ||
       is_close(image_orientation, &FEET_FIRST_PRONE) ||
       is_close(image_orientation, &FEET_FIRST_SUPINE) {
        return false;
    }
    
    // Default to head-first if orientation is non-standard
    eprintln!("Warning: Non-standard orientation detected: {:?}", image_orientation);
    true
}

/// Get the Z sign multiplier based on patient orientation
pub fn get_z_sign(image_orientation: &[f64; 6]) -> f64 {
    if is_head_first_orientation(image_orientation) {
        1.0
    } else {
        -1.0
    }
}

/// Calculate the actual Z positions of dose planes
pub fn calculate_dose_plane_positions(
    image_position_z: f64,
    grid_frame_offsets: &[f64],
    image_orientation: &[f64; 6],
) -> Vec<f64> {
    let z_sign = get_z_sign(image_orientation);
    
    grid_frame_offsets
        .iter()
        .map(|&offset| image_position_z + z_sign * offset)
        .collect()
}