# Aitrium DVH Calculator

High-performance Dose-Volume Histogram (DVH) calculation library for radiotherapy treatment planning, written in Rust.

## Overview

This crate provides a Rust implementation of DVH calculation that aims for parity with the Python `dicompylercore.dvhcalc` library while offering significant performance improvements.

## Testing Parity Between Implementations

### Prerequisites

#### For Python Implementation (dicompylercore)
```bash
# Create a virtual environment
python3 -m venv dvh_env
source dvh_env/bin/activate  # On macOS/Linux
# or
dvh_env\Scripts\activate  # On Windows

# Install required packages
pip install pydicom numpy matplotlib
pip install git+https://github.com/dicompyler/dicompyler-core.git
```

#### For Rust Implementation
```bash
# Build the Rust DVH calculator
cargo build --release --bin dvh_full
```

### Running Direct Comparisons

#### Method 1: Using the Existing Python Implementation

The repository includes the original Python DVH generator at `python/dvh_generator/main.py`:

```bash
# Activate your Python environment first
source dvh_env/bin/activate

# Run Python implementation
python python/dvh_generator/main.py /Volumes/T9/dvh_test_plans/test_dataset_A

# Run Rust implementation
cargo run --release --bin dvh_full /Volumes/T9/dvh_test_plans/test_dataset_A

# Compare the JSON outputs manually or with diff tools
```

#### Method 2: Side-by-Side Comparison Script

Create a file `compare_implementations.py`:

```python
#!/usr/bin/env python3
"""Compare Rust and Python DVH implementations on the same dataset."""

import json
import subprocess
import sys
from pathlib import Path
import pydicom
from dicompylercore import dvhcalc

def run_rust_dvh(dicom_dir):
    """Run Rust DVH and return JSON results."""
    # Assumes you're in the aitrium_dvh directory
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "dvh_full", dicom_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Rust error: {result.stderr}")
        return None
    return json.loads(result.stdout)

def run_python_dvh(dicom_dir):
    """Run Python DVH with dicompylercore."""
    dicom_path = Path(dicom_dir)
    
    # Find RTDOSE and RTSTRUCT files
    rtdose_file = None
    rtstruct_file = None
    
    for f in dicom_path.glob("*.dcm"):
        try:
            dcm = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            if hasattr(dcm, 'Modality'):
                if dcm.Modality == 'RTDOSE':
                    rtdose_file = f
                elif dcm.Modality == 'RTSTRUCT':
                    rtstruct_file = f
        except:
            pass
    
    if not rtdose_file or not rtstruct_file:
        print("Error: Could not find RTDOSE and RTSTRUCT files")
        return None
    
    # Load DICOM files
    rtss = pydicom.dcmread(str(rtstruct_file))
    results = {"dvhs": []}
    
    # Calculate DVH for each ROI
    for roi in rtss.StructureSetROISequence:
        try:
            # Calculate with NO interpolation for fair comparison
            # (Rust doesn't have interpolation yet)
            dvh = dvhcalc.get_dvh(
                str(rtstruct_file), 
                str(rtdose_file), 
                roi.ROINumber,
                interpolation_resolution=None,
                interpolation_segments_between_planes=0
            )
            
            if dvh and dvh.volume > 0.01:
                results["dvhs"].append({
                    "roi_name": roi.ROIName,
                    "stats": {
                        "total_cc": float(dvh.volume),
                        "min_gy": float(dvh.min),
                        "max_gy": float(dvh.max),
                        "mean_gy": float(dvh.mean),
                    },
                    "cumulative": dvh.counts.tolist()[:100],  # First 100 bins
                })
        except Exception as e:
            print(f"Warning: Failed DVH for {roi.ROIName}: {e}")
    
    return results

def compare_results(rust_data, python_data):
    """Compare and display differences."""
    if not rust_data or not python_data:
        print("Error: Missing data from one implementation")
        return
        
    rust_rois = {d["roi_name"]: d for d in rust_data.get("dvhs", [])}
    python_rois = {d["roi_name"]: d for d in python_data.get("dvhs", [])}
    
    print("\n" + "="*70)
    print("DVH COMPARISON: RUST vs PYTHON (dicompylercore)")
    print("="*70)
    
    common_rois = set(rust_rois.keys()) & set(python_rois.keys())
    
    total_diff_volume = 0
    total_diff_mean = 0
    count = 0
    
    for roi_name in sorted(common_rois):
        rust = rust_rois[roi_name]["stats"]
        python = python_rois[roi_name]["stats"]
        
        if python["total_cc"] < 1.0:  # Skip tiny volumes
            continue
        
        vol_diff_pct = abs(rust['total_cc'] - python['total_cc']) / python['total_cc'] * 100
        mean_diff_pct = abs(rust['mean_gy'] - python['mean_gy']) / max(python['mean_gy'], 0.01) * 100
        
        total_diff_volume += vol_diff_pct
        total_diff_mean += mean_diff_pct
        count += 1
        
        print(f"\n{roi_name}:")
        print(f"  Volume: Rust={rust['total_cc']:.2f} cc, Python={python['total_cc']:.2f} cc")
        print(f"          Difference: {vol_diff_pct:.1f}%")
        
        print(f"  Mean:   Rust={rust['mean_gy']:.2f} Gy, Python={python['mean_gy']:.2f} Gy")
        print(f"          Difference: {mean_diff_pct:.1f}%")
    
    if count > 0:
        print("\n" + "-"*70)
        print("AVERAGE DIFFERENCES:")
        print(f"  Volume: {total_diff_volume/count:.1f}%")
        print(f"  Mean Dose: {total_diff_mean/count:.1f}%")
        
        # Clinical acceptability
        print("\nCLINICAL ACCEPTABILITY:")
        if total_diff_volume/count < 2.0 and total_diff_mean/count < 3.0:
            print("  ✓ PASS - Differences within clinical tolerance")
        else:
            print("  ✗ FAIL - Differences exceed clinical tolerance")
            print("  Note: This is expected without interpolation implemented")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dicom_directory>")
        sys.exit(1)
    
    dicom_dir = sys.argv[1]
    
    print("Running Rust implementation...")
    rust_results = run_rust_dvh(dicom_dir)
    
    print("Running Python implementation (dicompylercore)...")
    python_results = run_python_dvh(dicom_dir)
    
    compare_results(rust_results, python_results)
```

### Test Datasets

Test datasets are provided at `/Volumes/T9/dvh_test_plans/`:
- **test_dataset_A**: Brain radiotherapy plan
- **test_dataset_B**: Prostate radiotherapy plan (same as `/Volumes/T9/patient_x`)

### Running the Comparison

```bash
# Make sure you're in the aitrium_dvh directory
cd /Users/spencerjohnson/projects/newleaf/newleaf-native/crates/aitrium_dvh

# Activate Python environment with dicompylercore
source dvh_env/bin/activate

# Run comparison on dataset A (Brain)
python compare_implementations.py /Volumes/T9/dvh_test_plans/test_dataset_A

# Run comparison on dataset B (Prostate)
python compare_implementations.py /Volumes/T9/dvh_test_plans/test_dataset_B
```

## Current Implementation Status

### Completed Features (with parity fixes)
- ✅ DICOM RTSTRUCT contour parsing
- ✅ DICOM RTDOSE pixel data reading (32-bit unsigned support)
- ✅ Dose→cGy binning matching numpy.histogram semantics
- ✅ Maxdose calculation: `int(dosemax * scaling * 100) + 1`
- ✅ Volume calculation using mean LUT spacing
- ✅ Late rescale pattern (accumulate counts, then scale to volume)
- ✅ Missing dose plane handling with `calculate_full_volume` flag
- ✅ Cumulative DVH calculation (volume receiving ≥ dose)
- ✅ DVH statistics (D-metrics, mean dose)
- ✅ XOR for contour holes

### Not Yet Implemented (Affecting Accuracy)
- ❌ **XY bilinear interpolation** (2x oversampling with power-of-2 adjustment)
- ❌ **Z-plane duplication** between slices
- ❌ Structure extents optimization
- ❌ Power-of-2 spacing validation

### Expected Differences Without Interpolation

The lack of interpolation is the primary source of differences:

| Metric | Expected Difference | Clinical Impact |
|--------|-------------------|-----------------|
| Volume | 0-5% | Acceptable for most cases |
| Mean Dose | 0-3% | Within tolerance |
| D95/D50/D5 | 0-2 Gy | May exceed tolerance for small structures |

**Important**: For clinical use, the Python implementation with interpolation remains the gold standard until interpolation is implemented in Rust.

## Performance Comparison

Typical performance on test datasets:

| Implementation | Time per ROI | Speedup |
|---------------|--------------|---------|
| Python (with interpolation) | 100-200ms | 1x |
| Python (no interpolation) | 50-100ms | 2x |
| Rust (no interpolation) | 5-10ms | 10-20x |

## Building and Testing

```bash
# Development build
cargo build

# Release build (optimized, recommended for testing)
cargo build --release

# Run unit tests
cargo test

# Run with test data
cargo run --release --bin dvh_full /Volumes/T9/dvh_test_plans/test_dataset_A
```

## Library API Usage

```rust
use aitrium_dvh::{DvhOptions, compute_dvh};

let options = DvhOptions {
    calculate_full_volume: false,
    use_structure_extents: false,
    interpolation_resolution: None,  // Not yet implemented
    interpolation_segments_between_planes: 0,
    limit_cgy: None,
};

let result = compute_dvh(
    "path/to/rtstruct.dcm",
    "path/to/rtdose.dcm", 
    roi_number,
    &options
)?;

println!("Volume: {} cc", result.total_volume_cc);
println!("Mean dose: {} Gy", result.stats.mean_gy);
```

## Next Steps for Full Parity

1. **Implement XY interpolation** - Critical for accuracy with coarse dose grids
2. **Implement Z duplication** - Important for structures spanning few dose planes
3. **Add structure extents** - Performance optimization
4. **Create golden test suite** - Automated parity testing

## Contributing

To contribute to parity improvements:

1. Run comparison tests on your datasets
2. Report differences exceeding clinical thresholds
3. Submit PRs with test cases that fail parity checks