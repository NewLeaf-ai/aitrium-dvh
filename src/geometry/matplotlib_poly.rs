use ndarray::Array2;

/// Point-in-polygon test that matches matplotlib.path.Path.contains_point behavior
///
/// Matplotlib uses the even-odd rule with specific edge inclusion:
/// - For CCW polygons: Left and Top edges are inclusive, Bottom and Right are exclusive
/// - For CW polygons: Different pattern (but we assume CCW as DICOM standard)
pub struct MatplotlibPolygon;

impl MatplotlibPolygon {
    /// Check if a point is inside a polygon using matplotlib's rules
    /// Delegates to winding number algorithm for consistency
    pub fn contains_point(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
        // Use the same winding algorithm as DVH for consistency
        Self::contains_point_winding(polygon, x, y)
    }

    /// Check if a point is exactly on an edge
    fn point_on_edge(p1: [f64; 2], p2: [f64; 2], x: f64, y: f64) -> bool {
        // Check if point is within bounding box of edge
        let min_x = p1[0].min(p2[0]);
        let max_x = p1[0].max(p2[0]);
        let min_y = p1[1].min(p2[1]);
        let max_y = p1[1].max(p2[1]);

        if x < min_x || x > max_x || y < min_y || y > max_y {
            return false;
        }

        // Check if point is on the line
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];

        if dx.abs() < 1e-10 {
            // Vertical line
            return (x - p1[0]).abs() < 1e-10;
        } else if dy.abs() < 1e-10 {
            // Horizontal line
            return (y - p1[1]).abs() < 1e-10;
        } else {
            // General line - check if point satisfies line equation
            let t = (x - p1[0]) / dx;
            let expected_y = p1[1] + t * dy;
            return (y - expected_y).abs() < 1e-10;
        }
    }

    /// Alternative implementation using winding number algorithm
    /// This might match matplotlib better for complex cases
    pub fn contains_point_winding(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
        let n = polygon.len();
        if n < 3 {
            return false;
        }

        let mut winding = 0i32;

        // Special debug for test point
        // let is_test_point = (x - (-78.075)).abs() < 0.001 && (y - (-53.181)).abs() < 0.001;

        for i in 0..n {
            let p1 = polygon[i];
            let p2 = polygon[(i + 1) % n];

            if p1[1] <= y {
                if p2[1] > y {
                    // Upward crossing
                    let cross = Self::is_left(p1, p2, x, y);
                    if cross > 0.0 {
                        winding += 1;
                        // if is_test_point && i == 42 {
                        //     eprintln!("      Edge {}: upward crossing ({:.1},{:.1})->({:.1},{:.1}), cross={:.3}, wn+1={}",
                        //         i, p1[0], p1[1], p2[0], p2[1], cross, winding);
                        // }
                    }
                }
            } else {
                if p2[1] <= y {
                    // Downward crossing
                    let cross = Self::is_left(p1, p2, x, y);
                    if cross < 0.0 {
                        winding -= 1;
                        // if is_test_point && i < 50 {
                        //     eprintln!("      Edge {}: downward crossing ({:.1},{:.1})->({:.1},{:.1}), cross={:.3}, wn-1={}",
                        //         i, p1[0], p1[1], p2[0], p2[1], cross, winding);
                        // }
                    }
                }
            }
        }

        // if is_test_point {
        //     eprintln!("      Winding for ({:.3}, {:.3}): {} ({})", x, y, winding, if winding != 0 { "INSIDE" } else { "OUTSIDE" });
        // }

        winding != 0
    }

    /// Test if point (x,y) is left/on/right of the line p1p2
    /// Returns: >0 for left, =0 for on, <0 for right
    fn is_left(p1: [f64; 2], p2: [f64; 2], x: f64, y: f64) -> f64 {
        (p2[0] - p1[0]) * (y - p1[1]) - (x - p1[0]) * (p2[1] - p1[1])
    }

    /// Create a mask for points inside the polygon
    pub fn create_mask(
        contour_points: &[[f64; 2]],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
    ) -> Array2<bool> {
        let rows = row_lut.len();
        let cols = col_lut.len();
        let mut mask = Array2::from_elem((rows, cols), false);

        // Ensure polygon is closed
        let mut polygon = contour_points.to_vec();
        if polygon.first() != polygon.last() {
            if let Some(first) = polygon.first() {
                polygon.push(*first);
            }
        }

        // Debug for skin contour (large contour with >900 points)
        let is_skin = polygon.len() > 900;
        let mut debug_count = 0;

        // Match Python's iteration order exactly
        if x_lut_index == 0 {
            // col_lut contains X coordinates, row_lut contains Y coordinates
            for (r, &y_coord) in row_lut.iter().enumerate() {
                for (c, &x_coord) in col_lut.iter().enumerate() {
                    // Use the winding number algorithm which seems to match matplotlib better
                    let inside = Self::contains_point_winding(&polygon, x_coord, y_coord);
                    mask[[r, c]] = inside;

                    // Debug output for skin - check specific points
                    if is_skin {
                        if debug_count < 10 || (debug_count >= 49 && debug_count <= 52) {
                            // eprintln!("    Matplotlib grid point {}: ({:.3}, {:.3}) -> {}",
                            //     debug_count, x_coord, y_coord, if inside { "INSIDE" } else { "OUTSIDE" });
                        }
                        // Test specific point that should be inside
                        if (x_coord - (-78.075)).abs() < 0.001
                            && (y_coord - (-53.181)).abs() < 0.001
                        {
                            // eprintln!("    *** Found test point (-78.075, -53.181) at index {}: {}",
                            //     debug_count, if inside { "INSIDE" } else { "OUTSIDE (WRONG!)" });
                        }
                    }
                    debug_count += 1;
                }
            }

            if is_skin {
                let total_inside = mask.iter().filter(|&&v| v).count();
                // eprintln!("    Matplotlib total voxels inside: {} / {}", total_inside, mask.len());
            }
        } else {
            // row_lut contains X coordinates, col_lut contains Y coordinates
            for (r, &x_coord) in row_lut.iter().enumerate() {
                for (c, &y_coord) in col_lut.iter().enumerate() {
                    mask[[r, c]] = Self::contains_point_winding(&polygon, x_coord, y_coord);
                }
            }
        }

        mask
    }

    /// Apply XOR operation between masks (for hole handling)
    pub fn xor_masks(mask1: &Array2<bool>, mask2: &Array2<bool>) -> Array2<bool> {
        assert_eq!(
            mask1.shape(),
            mask2.shape(),
            "Masks must have same shape for XOR"
        );

        let mut result = Array2::from_elem(mask1.dim(), false);
        for ((r, c), &val1) in mask1.indexed_iter() {
            let val2 = mask2[[r, c]];
            result[[r, c]] = val1 ^ val2;
        }
        result
    }

    /// Process multiple contours in a plane with XOR for hole removal
    pub fn create_plane_mask(
        contours: &[Vec<[f64; 2]>],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
    ) -> Array2<bool> {
        let rows = row_lut.len();
        let cols = col_lut.len();

        // Start with empty mask
        let mut combined_mask = Array2::from_elem((rows, cols), false);

        // Debug for skin (multiple contours with large point counts)
        let is_skin_multi = contours.len() > 1 && contours[0].len() > 300;

        // Process each contour and XOR with combined mask
        for (i, contour) in contours.iter().enumerate() {
            let contour_mask = Self::create_mask(contour, col_lut, row_lut, x_lut_index);

            if is_skin_multi {
                let contour_voxels = contour_mask.iter().filter(|&&v| v).count();
                // eprintln!("    Contour {}: {} voxels", i, contour_voxels);
            }

            combined_mask = Self::xor_masks(&combined_mask, &contour_mask);
        }

        if is_skin_multi {
            let final_voxels = combined_mask.iter().filter(|&&v| v).count();
            // eprintln!("    Final mask after XOR: {} voxels", final_voxels);

            // Check test point (-78.075, -53.181) at index 49
            if x_lut_index == 0 && col_lut.len() > 49 && row_lut.len() > 0 {
                let test_x = col_lut[49];
                let test_y = row_lut[0];
                if (test_x - (-78.075)).abs() < 0.001 && (test_y - (-53.181)).abs() < 0.001 {
                    let is_inside = combined_mask[[0, 49]];
                    // eprintln!("    *** FINAL test point (-78.075, -53.181) after XOR: {}",
                    //     if is_inside { "INSIDE (CORRECT!)" } else { "OUTSIDE (WRONG!)" });
                }
            }
        }

        combined_mask
    }
}
