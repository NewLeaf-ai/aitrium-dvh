use geo::{Contains, Coord, LineString, Polygon};
use ndarray::Array2;

/// Polygon mask generator matching matplotlib.path.Path.contains_points behavior
pub struct PolygonMask;

impl PolygonMask {
    /// Check if a point is inside a polygon using even-odd rule
    /// Matches matplotlib's default behavior
    pub fn contains_point(polygon: &Polygon<f64>, point: Coord<f64>) -> bool {
        // Use geo's contains method directly
        // Note: matplotlib has complex boundary behavior - some edges are inclusive, others exclusive
        polygon.contains(&point)
    }

    /// Generate a boolean mask for a polygon on a grid
    /// This matches Python's matplotlib.path.Path.contains_points behavior
    pub fn create_mask(
        contour_points: &[[f64; 2]],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
    ) -> Array2<bool> {
        let rows = row_lut.len();
        let cols = col_lut.len();

        // Convert contour points to geo Polygon
        let exterior: Vec<Coord<f64>> = contour_points
            .iter()
            .map(|p| Coord { x: p[0], y: p[1] })
            .collect();

        // Close the polygon if not already closed
        let mut exterior = exterior;
        if exterior.first() != exterior.last() {
            if let Some(first) = exterior.first() {
                exterior.push(*first);
            }
        }

        let line_string = LineString::from(exterior);
        let polygon = Polygon::new(line_string, vec![]);

        // Create grid of points and test containment
        let mut mask = Array2::from_elem((rows, cols), false);

        // Generate dosegridpoints matching Python's meshgrid approach
        // Python does:
        //   x, y = np.meshgrid(dd['lut'][x_index], dd['lut'][1-x_index])
        //   grid = path.contains_points(dosegridpoints)
        //   grid = grid.reshape((len(doselut[1]), len(doselut[0])))
        //
        // This means for x_lut_index==0:
        //   - meshgrid creates X varying along columns, Y along rows
        //   - when flattened, it goes row by row: all X values for first Y, then next Y, etc.
        //   - reshape is (Y_count, X_count) = (rows, cols)

        // Match Python's iteration order exactly
        if x_lut_index == 0 {
            // col_lut contains X coordinates, row_lut contains Y coordinates
            // Iterate Y (rows) first, then X (cols) - matching meshgrid flatten order
            let mut debug_count = 0;
            for (r, &y_coord) in row_lut.iter().enumerate() {
                for (c, &x_coord) in col_lut.iter().enumerate() {
                    let point = Coord {
                        x: x_coord,
                        y: y_coord,
                    };
                    let inside = Self::contains_point(&polygon, point);
                    mask[[r, c]] = inside;

                    // Debug first few points and test specific points
                    if contour_points.len() > 900 {
                        if debug_count == 0 {
                            // Test some known points (from Python test)
                            let test_pts = [(0.0, 0.0), (-100.0, 0.0), (100.0, 0.0), (0.0, 100.0)];
                            for (tx, ty) in &test_pts {
                                let tp = Coord { x: *tx, y: *ty };
                                let ti = Self::contains_point(&polygon, tp);
                                eprintln!(
                                    "    Test point ({:.0}, {:.0}): {}",
                                    tx,
                                    ty,
                                    if ti { "INSIDE" } else { "OUTSIDE" }
                                );
                            }
                        }
                        if debug_count < 10 {
                            eprintln!(
                                "    Grid point {}: ({:.3}, {:.3}) -> {}",
                                debug_count,
                                x_coord,
                                y_coord,
                                if inside { "INSIDE" } else { "OUTSIDE" }
                            );
                        }
                    }
                    debug_count += 1;
                }
            }
        } else {
            // row_lut contains X coordinates, col_lut contains Y coordinates (decubitus)
            // Still iterate rows first (which are now X values)
            for (r, &x_coord) in row_lut.iter().enumerate() {
                for (c, &y_coord) in col_lut.iter().enumerate() {
                    let point = Coord {
                        x: x_coord,
                        y: y_coord,
                    };
                    mask[[r, c]] = Self::contains_point(&polygon, point);
                }
            }
        }

        mask
    }

    /// Apply XOR operation between masks (for hole handling)
    /// Matches numpy.logical_xor behavior
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
    /// Matches Python's iterative XOR approach
    pub fn create_plane_mask(
        contours: &[Vec<[f64; 2]>],
        col_lut: &[f64],
        row_lut: &[f64],
        x_lut_index: u8,
    ) -> Array2<bool> {
        let rows = row_lut.len();
        let cols = col_lut.len();

        // eprintln!("Creating mask for {} contours on {}x{} grid", contours.len(), rows, cols);

        // Start with empty mask
        let mut combined_mask = Array2::from_elem((rows, cols), false);

        // Process each contour and XOR with combined mask
        for (i, contour) in contours.iter().enumerate() {
            // eprintln!("  Processing contour {}/{} with {} points", i+1, contours.len(), contour.len());
            let contour_mask = Self::create_mask(contour, col_lut, row_lut, x_lut_index);
            combined_mask = Self::xor_masks(&combined_mask, &contour_mask);
        }

        combined_mask
    }
}
