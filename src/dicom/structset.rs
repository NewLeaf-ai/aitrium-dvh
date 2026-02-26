use crate::types::{Contour, ContourType, DvhError, OrderedFloat, Roi};
use dicom::object::{DefaultDicomObject, FileDicomObject, InMemDicomObject, open_file};
use dicom::core::dictionary_std::tags;
use std::collections::BTreeMap;
use std::path::Path;

pub struct StructureSetParser;

impl StructureSetParser {
    /// Parse RTSTRUCT file and extract ROI information
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Vec<Roi>, DvhError> {
        let obj = open_file(path)
            .map_err(|e| DvhError::DicomError(format!("Failed to open RTSTRUCT: {}", e)))?;
        
        Self::parse_object(&obj)
    }
    
    /// Parse RTSTRUCT from DICOM object
    pub fn parse_object(obj: &FileDicomObject<InMemDicomObject>) -> Result<Vec<Roi>, DvhError> {
        // Get StructureSetROISequence
        let roi_sequence = obj
            .element(tags::STRUCTURE_SET_ROI_SEQUENCE)
            .map_err(|_| DvhError::DicomError("Missing StructureSetROISequence".to_string()))?
            .items()
            .map_err(|_| DvhError::DicomError("Invalid StructureSetROISequence".to_string()))?;
        
        // Get ROIContourSequence
        let contour_sequence = obj
            .element(tags::ROI_CONTOUR_SEQUENCE)
            .map_err(|_| DvhError::DicomError("Missing ROIContourSequence".to_string()))?
            .items()
            .map_err(|_| DvhError::DicomError("Invalid ROIContourSequence".to_string()))?;
        
        let mut rois = Vec::new();
        
        // Process each ROI
        for roi_item in roi_sequence {
            let roi_number = roi_item
                .element(tags::ROI_NUMBER)
                .map_err(|_| DvhError::DicomError("Missing ROI number".to_string()))?
                .to_int()
                .map_err(|_| DvhError::DicomError("Invalid ROI number".to_string()))?;
            
            let roi_name = roi_item
                .element(tags::ROI_NAME)
                .ok()
                .and_then(|e| e.to_str().ok())
                .unwrap_or_else(|| format!("ROI_{}", roi_number))
                .to_string();
            
            // Find corresponding contour data
            let contour_data = contour_sequence
                .iter()
                .find(|c| {
                    c.element(tags::REFERENCED_ROI_NUMBER)
                        .ok()
                        .and_then(|e| e.to_int().ok())
                        .map(|n| n == roi_number)
                        .unwrap_or(false)
                });
            
            if let Some(contour_item) = contour_data {
                let planes = Self::extract_contour_planes(contour_item)?;
                
                // Calculate thickness from plane spacing
                let thickness_mm = Self::calculate_plane_thickness(&planes);
                
                rois.push(Roi {
                    id: roi_number,
                    name: roi_name,
                    planes,
                    thickness_mm,
                });
            }
        }
        
        Ok(rois)
    }
    
    /// Extract contour planes from ROIContourSequence item
    fn extract_contour_planes(
        contour_item: &DefaultDicomObject
    ) -> Result<BTreeMap<OrderedFloat, Vec<Contour>>, DvhError> {
        let mut planes = BTreeMap::new();
        
        let contour_sequence = contour_item
            .element(tags::CONTOUR_SEQUENCE)
            .map_err(|_| DvhError::DicomError("Missing ContourSequence".to_string()))?
            .items()
            .map_err(|_| DvhError::DicomError("Invalid ContourSequence".to_string()))?;
        
        for contour in contour_sequence {
            // Get contour type (CLOSED_PLANAR expected)
            let geometric_type = contour
                .element(tags::CONTOUR_GEOMETRIC_TYPE)
                .ok()
                .and_then(|e| e.to_str().ok())
                .unwrap_or("CLOSED_PLANAR");
            
            if geometric_type != "CLOSED_PLANAR" {
                continue; // Skip non-planar contours
            }
            
            // Get contour data (x,y,z coordinates)
            let contour_data = contour
                .element(tags::CONTOUR_DATA)
                .map_err(|_| DvhError::DicomError("Missing ContourData".to_string()))?
                .to_multi_float64()
                .map_err(|_| DvhError::DicomError("Invalid ContourData".to_string()))?;
            
            if contour_data.len() % 3 != 0 {
                return Err(DvhError::DicomError("ContourData length not multiple of 3".to_string()));
            }
            
            // Extract points
            let mut points = Vec::new();
            let mut z_position = None;
            
            for chunk in contour_data.chunks_exact(3) {
                let x = chunk[0];
                let y = chunk[1];
                let z = chunk[2];
                
                if z_position.is_none() {
                    z_position = Some(z);
                }
                
                points.push([x, y]);
            }
            
            if let Some(z) = z_position {
                let contour_type = if contour
                    .element(tags::CONTOUR_IMAGE_SEQUENCE)
                    .is_ok()
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
    
    /// Get a specific ROI by number
    pub fn get_roi(rois: &[Roi], roi_number: i32) -> Option<&Roi> {
        rois.iter().find(|r| r.id == roi_number)
    }
}