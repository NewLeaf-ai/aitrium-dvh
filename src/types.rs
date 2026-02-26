use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Options for DVH calculation, matching Python's parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DvhOptions {
    /// Dose limit in cGy as a maximum bin for the histogram
    pub limit_cgy: Option<u32>,

    /// Calculate full structure volume including contours outside dose grid
    pub calculate_full_volume: bool,

    /// Limit DVH calculation to in-plane structure boundaries
    pub use_structure_extents: bool,

    /// Resolution in mm (row, col) to interpolate structure and dose data
    pub interpolation_resolution_mm: Option<(f64, f64)>,

    /// Number of segments to interpolate between structure slices
    pub interpolation_segments_between_planes: u32,

    /// Structure thickness override in mm
    pub thickness_override_mm: Option<f64>,

    /// Use memory mapping for dose pixel array
    pub memmap_rtdose: bool,
}

impl Default for DvhOptions {
    fn default() -> Self {
        Self {
            limit_cgy: None,
            calculate_full_volume: true,
            use_structure_extents: false,
            interpolation_resolution_mm: None,
            interpolation_segments_between_planes: 0,
            thickness_override_mm: None,
            memmap_rtdose: false,
        }
    }
}

/// Result of DVH calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DvhResult {
    /// Optional notes about calculation (e.g., warnings)
    pub notes: Option<String>,

    /// Differential histogram in cm³ per cGy bin
    pub differential_hist_cgy: Vec<f64>,

    /// Bin edges (0..N+1)
    pub bins: Vec<f64>,

    /// Cumulative DVH (convenience field)
    pub cumulative: Vec<f64>,

    /// Structure name
    pub name: String,

    /// Total volume in cc
    pub total_volume_cc: f64,

    /// Statistics (for compatibility with Python output)
    pub stats: DvhStats,
}

/// DVH statistics matching Python output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DvhStats {
    pub n_bins: usize,
    pub total_cc: f64,
    pub min_gy: f64,
    pub max_gy: f64,
    pub mean_gy: f64,
    #[serde(rename = "D100_gy")]
    pub d100_gy: f64,
    #[serde(rename = "D98_gy")]
    pub d98_gy: f64,
    #[serde(rename = "D95_gy")]
    pub d95_gy: f64,
    #[serde(rename = "D90_gy")]
    pub d90_gy: f64,
    #[serde(rename = "D80_gy")]
    pub d80_gy: f64,
    #[serde(rename = "D70_gy")]
    pub d70_gy: f64,
    #[serde(rename = "D60_gy")]
    pub d60_gy: f64,
    #[serde(rename = "D50_gy")]
    pub d50_gy: f64,
    #[serde(rename = "D40_gy")]
    pub d40_gy: f64,
    #[serde(rename = "D30_gy")]
    pub d30_gy: f64,
    #[serde(rename = "D20_gy")]
    pub d20_gy: f64,
    #[serde(rename = "D10_gy")]
    pub d10_gy: f64,
    #[serde(rename = "D5_gy")]
    pub d5_gy: f64,
    #[serde(rename = "D2_gy")]
    pub d2_gy: f64,
    #[serde(rename = "D1_gy")]
    pub d1_gy: f64,
    #[serde(rename = "D0_gy")]
    pub d0_gy: f64,
    pub homogeneity_index: f64,
}

/// Anatomical direction for margin calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarginDirection {
    /// Margin in all directions (default)
    Uniform,
    /// Posterior direction (toward back)
    Posterior,
    /// Anterior direction (toward front)
    Anterior,
    /// Left lateral direction
    Left,
    /// Right lateral direction
    Right,
    /// Superior direction (toward head)
    Superior,
    /// Inferior direction (toward feet)
    Inferior,
}

/// ROI (Region of Interest) structure
#[derive(Debug, Clone)]
pub struct Roi {
    pub id: i32,
    pub name: String,
    /// Z position -> list of contours (each contour is a polygon)
    pub planes: BTreeMap<OrderedFloat, Vec<Contour>>,
    pub thickness_mm: f64,
}

/// A single contour (polygon) in a plane
#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<[f64; 2]>, // [x, y] coordinates in patient space
    pub contour_type: ContourType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourType {
    External,
    Cavity,
}

/// Dose grid information
#[derive(Debug)]
pub struct DoseGrid {
    /// DoseGridScaling factor to convert to Gy
    pub scale_to_gy: f64,

    /// Grid dimensions
    pub rows: usize,
    pub cols: usize,

    /// Pixel spacing in mm
    pub pixel_spacing_row_mm: f64,
    pub pixel_spacing_col_mm: f64,

    /// DICOM position and orientation
    pub image_position_patient: [f64; 3],
    pub image_orientation_patient: [f64; 6],

    /// Z-axis positions
    pub grid_frame_offset_vector_mm: Vec<f64>,

    /// 3D dose array (z, row, col)
    pub dose_3d: DoseBacking,

    /// Index for X coordinate (0 or 1, matching Python's x_lut_index)
    pub x_lut_index: u8,

    /// Patient coordinate LUTs
    pub col_lut: Vec<f64>,
    pub row_lut: Vec<f64>,

    /// Patient position from DICOM (e.g., "HFS", "HFP", "FFS", "FFP")
    pub patient_position: Option<String>,
}

/// Backing storage for dose data
#[derive(Debug)]
pub enum DoseBacking {
    Owned(Array3<f32>),
    #[cfg(feature = "memmap")]
    MemMapped {
        data: memmap2::Mmap,
        shape: (usize, usize, usize),
    },
}

impl DoseBacking {
    /// Get a dose plane at a specific z index
    pub fn get_plane(&self, z_index: usize) -> Option<Array2<f32>> {
        match self {
            DoseBacking::Owned(arr) => {
                if z_index < arr.shape()[0] {
                    Some(arr.slice(ndarray::s![z_index, .., ..]).to_owned())
                } else {
                    None
                }
            }
            #[cfg(feature = "memmap")]
            DoseBacking::MemMapped { data, shape } => {
                if z_index < shape.0 {
                    // Parse from memory-mapped data
                    let plane_size = shape.1 * shape.2;
                    let offset = z_index * plane_size * std::mem::size_of::<f32>();
                    let plane_data =
                        &data[offset..offset + plane_size * std::mem::size_of::<f32>()];

                    // Convert bytes to f32 array
                    let values: Vec<f32> = plane_data
                        .chunks_exact(4)
                        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                        .collect();

                    Some(Array2::from_shape_vec((shape.1, shape.2), values).unwrap())
                } else {
                    None
                }
            }
        }
    }
}

/// Wrapper for ordered floats (for BTreeMap keys)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Error types for DVH calculation
#[derive(Error, Debug)]
pub enum DvhError {
    #[error("DICOM parsing error: {0}")]
    DicomError(String),

    #[error("Invalid ROI number: {0}")]
    InvalidRoi(i32),

    #[error("Dose grid error: {0}")]
    DoseGridError(String),

    #[error("Interpolation error: {0}")]
    InterpolationError(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// JSON output format matching Python DVH generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiDvhJson {
    pub roi_name: String,
    pub stats: DvhStats,
    pub doses_gy: Vec<f64>,
    pub volumes_cc: Vec<f64>,
}

/// Batch output for all ROIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOutput {
    pub dvhs: Vec<RoiDvhJson>,
}
