use crate::types::{Contour, OrderedFloat};
use std::collections::BTreeMap;

/// Z-plane interpolation for structures
/// Matches dicompylercore's interpolate_between_planes behavior
pub struct ZInterpolator;

impl ZInterpolator {
    /// Interpolate n additional structure planes between existing planes
    /// This duplicates the nearest plane (not true geometric interpolation)
    /// Matches Python: interpolate_between_planes(planes, n)
    pub fn interpolate_planes(
        planes: &BTreeMap<OrderedFloat, Vec<Contour>>,
        segments_between: u32,
    ) -> BTreeMap<OrderedFloat, Vec<Contour>> {
        if segments_between == 0 || planes.is_empty() {
            return planes.clone();
        }

        // Get sorted Z positions
        let z_positions: Vec<f64> = planes.keys().map(|k| k.0).collect();
        if z_positions.len() < 2 {
            return planes.clone();
        }

        // Calculate total number of planes after interpolation
        // Python: num_new_samples = (len(planes.keys()) * (n + 1)) - n
        let num_new_samples =
            (z_positions.len() * (segments_between as usize + 1)) - segments_between as usize;

        // Create evenly spaced Z positions
        let z_min = z_positions[0];
        let z_max = z_positions[z_positions.len() - 1];
        let z_step = (z_max - z_min) / (num_new_samples - 1) as f64;

        let mut interpolated_planes = BTreeMap::new();

        // For each new Z position, find the nearest original plane
        for i in 0..num_new_samples {
            let z_new = z_min + (i as f64 * z_step);

            // Find nearest original Z position
            let nearest_idx = Self::find_nearest_index(&z_positions, z_new);
            let nearest_z = z_positions[nearest_idx];

            // Get the contours from the nearest plane
            if let Some(contours) = planes.get(&OrderedFloat(nearest_z)) {
                interpolated_planes.insert(OrderedFloat(z_new), contours.clone());
            }
        }

        interpolated_planes
    }

    /// Find index of nearest value in sorted array
    fn find_nearest_index(sorted_values: &[f64], target: f64) -> usize {
        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (idx, &val) in sorted_values.iter().enumerate() {
            let dist = (val - target).abs();
            if dist < min_dist {
                min_dist = dist;
                min_idx = idx;
            }
        }

        min_idx
    }

    /// Calculate adjusted thickness after interpolation
    /// Python: structure['thickness'] / (interpolation_segments_between_planes + 1)
    pub fn adjusted_thickness(original_thickness: f64, segments_between: u32) -> f64 {
        if segments_between == 0 {
            original_thickness
        } else {
            original_thickness / (segments_between + 1) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ContourType;

    #[test]
    fn test_no_interpolation() {
        let mut planes = BTreeMap::new();
        planes.insert(OrderedFloat(0.0), vec![]);
        planes.insert(OrderedFloat(10.0), vec![]);

        let result = ZInterpolator::interpolate_planes(&planes, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_single_interpolation() {
        let mut planes = BTreeMap::new();
        let contour1 = Contour {
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            contour_type: ContourType::External,
        };
        let contour2 = contour1.clone();

        planes.insert(OrderedFloat(0.0), vec![contour1]);
        planes.insert(OrderedFloat(10.0), vec![contour2]);

        // With n=1, we should get 3 planes total
        // Python: (2 * (1 + 1)) - 1 = 3
        let result = ZInterpolator::interpolate_planes(&planes, 1);
        assert_eq!(result.len(), 3);

        // Check that intermediate plane exists
        let z_values: Vec<f64> = result.keys().map(|k| k.0).collect();
        assert!((z_values[0] - 0.0).abs() < 1e-6);
        assert!((z_values[1] - 5.0).abs() < 1e-6);
        assert!((z_values[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_thickness_adjustment() {
        assert_eq!(ZInterpolator::adjusted_thickness(3.0, 0), 3.0);
        assert_eq!(ZInterpolator::adjusted_thickness(3.0, 2), 1.0);
        assert_eq!(ZInterpolator::adjusted_thickness(4.0, 1), 2.0);
    }
}
