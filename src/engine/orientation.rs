/// Patient orientation and directional margin calculation module
///
/// This module handles conversion between anatomical directions and patient coordinate systems
/// based on DICOM patient position tags.
use crate::types::MarginDirection;

/// Patient position (orientation) in DICOM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatientPosition {
    /// Head First Supine
    HFS,
    /// Head First Prone
    HFP,
    /// Feet First Supine
    FFS,
    /// Feet First Prone
    FFP,
    /// Head First Decubitus Left
    HFDL,
    /// Head First Decubitus Right
    HFDR,
    /// Feet First Decubitus Left
    FFDL,
    /// Feet First Decubitus Right
    FFDR,
}

impl PatientPosition {
    /// Parse from DICOM string
    pub fn from_dicom_string(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "HFS" => Some(PatientPosition::HFS),
            "HFP" => Some(PatientPosition::HFP),
            "FFS" => Some(PatientPosition::FFS),
            "FFP" => Some(PatientPosition::FFP),
            "HFDL" => Some(PatientPosition::HFDL),
            "HFDR" => Some(PatientPosition::HFDR),
            "FFDL" => Some(PatientPosition::FFDL),
            "FFDR" => Some(PatientPosition::FFDR),
            _ => None,
        }
    }
}

/// Convert anatomical direction to a 3D vector in patient coordinate system
///
/// Returns a normalized direction vector [x, y, z] where:
/// - X axis: left (+) to right (-) of patient
/// - Y axis: posterior (-) to anterior (+) for supine, opposite for prone
/// - Z axis: inferior (-) to superior (+) for head first, opposite for feet first
pub fn direction_to_vector(
    direction: MarginDirection,
    patient_position: Option<PatientPosition>,
) -> [f64; 3] {
    // Default to HFS if patient position not specified
    let pos = patient_position.unwrap_or(PatientPosition::HFS);

    match direction {
        MarginDirection::Uniform => {
            // No specific direction - this shouldn't be called for uniform margins
            [0.0, 0.0, 0.0]
        }
        MarginDirection::Lateral => {
            // V1 path does not support bilateral aggregation at this level.
            [0.0, 0.0, 0.0]
        }
        MarginDirection::Posterior => match pos {
            PatientPosition::HFS | PatientPosition::FFS => [0.0, -1.0, 0.0],
            PatientPosition::HFP | PatientPosition::FFP => [0.0, 1.0, 0.0],
            PatientPosition::HFDL | PatientPosition::FFDL => [-1.0, 0.0, 0.0],
            PatientPosition::HFDR | PatientPosition::FFDR => [1.0, 0.0, 0.0],
        },
        MarginDirection::Anterior => match pos {
            PatientPosition::HFS | PatientPosition::FFS => [0.0, 1.0, 0.0],
            PatientPosition::HFP | PatientPosition::FFP => [0.0, -1.0, 0.0],
            PatientPosition::HFDL | PatientPosition::FFDL => [1.0, 0.0, 0.0],
            PatientPosition::HFDR | PatientPosition::FFDR => [-1.0, 0.0, 0.0],
        },
        MarginDirection::Left => match pos {
            PatientPosition::HFS | PatientPosition::HFP => [1.0, 0.0, 0.0],
            PatientPosition::FFS | PatientPosition::FFP => [-1.0, 0.0, 0.0],
            PatientPosition::HFDL | PatientPosition::HFDR => [0.0, 1.0, 0.0],
            PatientPosition::FFDL | PatientPosition::FFDR => [0.0, -1.0, 0.0],
        },
        MarginDirection::Right => match pos {
            PatientPosition::HFS | PatientPosition::HFP => [-1.0, 0.0, 0.0],
            PatientPosition::FFS | PatientPosition::FFP => [1.0, 0.0, 0.0],
            PatientPosition::HFDL | PatientPosition::HFDR => [0.0, -1.0, 0.0],
            PatientPosition::FFDL | PatientPosition::FFDR => [0.0, 1.0, 0.0],
        },
        MarginDirection::Superior => match pos {
            PatientPosition::HFS
            | PatientPosition::HFP
            | PatientPosition::HFDL
            | PatientPosition::HFDR => [0.0, 0.0, 1.0],
            PatientPosition::FFS
            | PatientPosition::FFP
            | PatientPosition::FFDL
            | PatientPosition::FFDR => [0.0, 0.0, -1.0],
        },
        MarginDirection::Inferior => match pos {
            PatientPosition::HFS
            | PatientPosition::HFP
            | PatientPosition::HFDL
            | PatientPosition::HFDR => [0.0, 0.0, -1.0],
            PatientPosition::FFS
            | PatientPosition::FFP
            | PatientPosition::FFDL
            | PatientPosition::FFDR => [0.0, 0.0, 1.0],
        },
    }
}

/// Check if a point is in the specified direction from a center point
///
/// # Arguments
/// * `point` - The point to check [x, y, z] in patient coordinates
/// * `center` - The center point [x, y, z] in patient coordinates
/// * `direction_vector` - The normalized direction vector
/// * `tolerance_degrees` - Angular tolerance in degrees (e.g., 45 degrees)
///
/// # Returns
/// True if the point is within the direction cone from center
pub fn is_point_in_direction(
    point: [f64; 3],
    center: [f64; 3],
    direction_vector: [f64; 3],
    tolerance_degrees: f64,
) -> bool {
    // Skip uniform direction (zero vector)
    if direction_vector == [0.0, 0.0, 0.0] {
        return true;
    }

    // Vector from center to point
    let vec_to_point = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];

    // Normalize the vector
    let magnitude =
        (vec_to_point[0].powi(2) + vec_to_point[1].powi(2) + vec_to_point[2].powi(2)).sqrt();

    if magnitude < 1e-6 {
        // Point is at center, consider it in all directions
        return true;
    }

    let vec_normalized = [
        vec_to_point[0] / magnitude,
        vec_to_point[1] / magnitude,
        vec_to_point[2] / magnitude,
    ];

    // Calculate dot product
    let dot_product = vec_normalized[0] * direction_vector[0]
        + vec_normalized[1] * direction_vector[1]
        + vec_normalized[2] * direction_vector[2];

    // Calculate angle in radians
    let angle_radians = dot_product.clamp(-1.0, 1.0).acos();
    let angle_degrees = angle_radians.to_degrees();

    // Check if within tolerance cone
    angle_degrees <= tolerance_degrees
}

/// Calculate the center of mass for a structure
///
/// # Arguments
/// * `mask` - 2D binary mask for the structure in a slice
/// * `col_lut` - Column coordinate lookup table
/// * `row_lut` - Row coordinate lookup table
/// * `z` - Z coordinate of the slice
///
/// # Returns
/// Center of mass [x, y, z] in patient coordinates
pub fn calculate_center_of_mass_2d(
    mask: &ndarray::Array2<bool>,
    col_lut: &[f64],
    row_lut: &[f64],
    z: f64,
) -> [f64; 3] {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;

    for ((i, j), &is_inside) in mask.indexed_iter() {
        if is_inside {
            sum_x += col_lut[j];
            sum_y += row_lut[i];
            count += 1.0;
        }
    }

    if count > 0.0 {
        [sum_x / count, sum_y / count, z]
    } else {
        [0.0, 0.0, z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patient_position_parsing() {
        assert_eq!(
            PatientPosition::from_dicom_string("HFS"),
            Some(PatientPosition::HFS)
        );
        assert_eq!(
            PatientPosition::from_dicom_string("hfs"),
            Some(PatientPosition::HFS)
        );
        assert_eq!(PatientPosition::from_dicom_string("invalid"), None);
    }

    #[test]
    fn test_direction_vectors_hfs() {
        let pos = Some(PatientPosition::HFS);

        // In HFS: posterior is -Y, anterior is +Y, left is +X, right is -X
        assert_eq!(
            direction_to_vector(MarginDirection::Posterior, pos),
            [0.0, -1.0, 0.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Anterior, pos),
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Left, pos),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Right, pos),
            [-1.0, 0.0, 0.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Superior, pos),
            [0.0, 0.0, 1.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Inferior, pos),
            [0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn test_direction_vectors_hfp() {
        let pos = Some(PatientPosition::HFP);

        // In HFP (prone): posterior is +Y, anterior is -Y
        assert_eq!(
            direction_to_vector(MarginDirection::Posterior, pos),
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            direction_to_vector(MarginDirection::Anterior, pos),
            [0.0, -1.0, 0.0]
        );
    }

    #[test]
    fn test_point_in_direction() {
        let center = [0.0, 0.0, 0.0];
        let point_anterior = [0.0, 1.0, 0.0];
        let point_posterior = [0.0, -1.0, 0.0];
        let direction_anterior = [0.0, 1.0, 0.0];

        // Point in anterior direction should be in anterior cone
        assert!(is_point_in_direction(
            point_anterior,
            center,
            direction_anterior,
            45.0
        ));

        // Point in posterior direction should NOT be in anterior cone
        assert!(!is_point_in_direction(
            point_posterior,
            center,
            direction_anterior,
            45.0
        ));

        // Uniform direction (zero vector) accepts all points
        assert!(is_point_in_direction(
            point_anterior,
            center,
            [0.0, 0.0, 0.0],
            45.0
        ));
    }
}
