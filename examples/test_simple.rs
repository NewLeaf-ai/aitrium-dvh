use aitrium_dvh::engine::HistogramCalculator;

fn main() {
    // Test simple histogram
    let differential = vec![0.0, 5.0, 10.0, 15.0, 10.0, 5.0, 0.0];
    println!("Differential: {:?}", differential);

    let cumulative = HistogramCalculator::to_cumulative(&differential);
    println!("Cumulative: {:?}", cumulative);

    // Expected cumulative (volume receiving >= dose):
    // Total = 45
    // [45, 45, 40, 30, 15, 5, 0]

    let total: f64 = differential.iter().sum();
    println!("Total volume: {}", total);

    // Verify cumulative is correct
    for (i, &vol) in cumulative.iter().enumerate() {
        let expected_vol: f64 = differential[i..].iter().sum();
        if (vol - expected_vol).abs() > 0.001 {
            println!("ERROR at bin {}: got {}, expected {}", i, vol, expected_vol);
        }
    }

    // Test trim zeros
    let with_zeros = vec![0.0, 5.0, 10.0, 0.0, 0.0];
    let trimmed = HistogramCalculator::trim_zeros(with_zeros);
    println!("\nTrimmed: {:?}", trimmed);
}
