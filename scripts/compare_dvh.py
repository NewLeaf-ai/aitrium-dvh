#!/usr/bin/env python3
"""
Compare DVH results between Python (dicompylercore) and Rust implementations
"""
import json
import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

def run_python_dvh(dicom_dir, output_file=None):
    """Run Python DVH generator and return results"""
    print(f"Running Python DVH generator on {dicom_dir}...")
    start_time = time.time()
    
    # Run the Python DVH generator
    python_script = Path(__file__).parent.parent.parent.parent / "python" / "dvh_generator" / "main.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(python_script), str(dicom_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"  Python completed in {elapsed:.2f} seconds")
        
        output = json.loads(result.stdout)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
                print(f"  Saved Python output to {output_file}")
        
        return output, elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"  Python DVH failed: {e}")
        print(f"  stderr: {e.stderr}")
        return None, 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse Python output as JSON: {e}")
        return None, 0

def run_rust_dvh(dicom_dir, output_file=None):
    """Run Rust DVH generator and return results"""
    print(f"Running Rust DVH generator on {dicom_dir}...")
    start_time = time.time()
    
    # Build the Rust binary first
    print("  Building Rust binary...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "dvh_generator_rs"],
        cwd=Path(__file__).parent.parent,
        check=True,
        capture_output=True
    )
    
    rust_binary = Path(__file__).parent.parent / "target" / "release" / "dvh_generator_rs"
    
    try:
        result = subprocess.run(
            [str(rust_binary), str(dicom_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"  Rust completed in {elapsed:.2f} seconds")
        
        output = json.loads(result.stdout)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
                print(f"  Saved Rust output to {output_file}")
        
        return output, elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"  Rust DVH failed: {e}")
        print(f"  stderr: {e.stderr}")
        return None, 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse Rust output as JSON: {e}")
        return None, 0

def compare_dvh_results(python_output, rust_output):
    """Compare DVH results between Python and Rust"""
    if not python_output or not rust_output:
        return False
    
    python_dvhs = python_output.get('dvhs', [])
    rust_dvhs = rust_output.get('dvhs', [])
    
    print(f"\nComparing {len(python_dvhs)} Python ROIs with {len(rust_dvhs)} Rust ROIs")
    
    # Compare each ROI
    for py_roi in python_dvhs:
        roi_name = py_roi['roi_name']
        
        # Find matching Rust ROI
        rust_roi = next((r for r in rust_dvhs if r['roi_name'] == roi_name), None)
        
        if not rust_roi:
            print(f"  ❌ ROI '{roi_name}' not found in Rust output")
            continue
        
        # Compare statistics
        py_stats = py_roi['stats']
        rust_stats = rust_roi['stats']
        
        # Volume comparison
        volume_diff = abs(py_stats['total_cc'] - rust_stats['total_cc'])
        volume_tolerance = max(0.1, py_stats['total_cc'] * 0.005)  # 0.1 cc or 0.5%
        
        if volume_diff > volume_tolerance:
            print(f"  ❌ ROI '{roi_name}' volume mismatch: Python={py_stats['total_cc']:.2f} cc, Rust={rust_stats['total_cc']:.2f} cc, diff={volume_diff:.2f} cc")
        else:
            print(f"  ✅ ROI '{roi_name}' volume match: {py_stats['total_cc']:.2f} cc")
        
        # D-metrics comparison
        for metric in ['D98_gy', 'D95_gy', 'D50_gy', 'D2_gy']:
            if metric in py_stats and metric in rust_stats:
                diff = abs(py_stats[metric] - rust_stats[metric])
                if diff > 0.1:  # 0.1 Gy tolerance
                    print(f"     ❌ {metric}: Python={py_stats[metric]:.2f} Gy, Rust={rust_stats[metric]:.2f} Gy, diff={diff:.2f} Gy")
                else:
                    print(f"     ✅ {metric}: {py_stats[metric]:.2f} Gy")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Compare Python and Rust DVH implementations')
    parser.add_argument('dicom_dir', help='Directory containing DICOM files')
    parser.add_argument('--save-outputs', action='store_true', help='Save outputs to files')
    parser.add_argument('--python-only', action='store_true', help='Run Python implementation only')
    parser.add_argument('--rust-only', action='store_true', help='Run Rust implementation only')
    
    args = parser.parse_args()
    
    dicom_dir = Path(args.dicom_dir)
    if not dicom_dir.exists():
        print(f"Error: Directory {dicom_dir} does not exist")
        sys.exit(1)
    
    # Run Python implementation
    python_output = None
    python_time = 0
    if not args.rust_only:
        python_output_file = "python_output.json" if args.save_outputs else None
        python_output, python_time = run_python_dvh(dicom_dir, python_output_file)
    
    # Run Rust implementation
    rust_output = None
    rust_time = 0
    if not args.python_only:
        rust_output_file = "rust_output.json" if args.save_outputs else None
        rust_output, rust_time = run_rust_dvh(dicom_dir, rust_output_file)
    
    # Compare results
    if python_output and rust_output:
        print("\n" + "="*60)
        compare_dvh_results(python_output, rust_output)
        
        print("\n" + "="*60)
        print("Performance Summary:")
        print(f"  Python: {python_time:.2f} seconds")
        print(f"  Rust:   {rust_time:.2f} seconds")
        if rust_time > 0:
            speedup = python_time / rust_time
            print(f"  Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    main()