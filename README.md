# Aitrium DVH (`aitrium_dvh`)

`aitrium_dvh` is the core Rust engine for radiotherapy DVH computation from DICOM RT datasets.

It is designed for:
- Computing DVH curves and summary metrics from `RTSTRUCT` + `RTDOSE`.
- Powering higher-level tools/services (for example, `aitrium-radiotherapy`).
- Local/offline analysis workflows with no cloud dependency.

## What This Project Contains

- Rust library crate: `aitrium_dvh`
- CLI binaries:
  - `dvh_full` - compute DVHs for all structures in a DICOM directory
  - `dvh_metrics_by_roi` - compute selected metrics for explicit ROI numbers

## Install

### Option 1: Build from source (recommended)

```bash
git clone https://github.com/NewLeaf-ai/aitrium-dvh.git
cd aitrium-dvh
cargo build --release
```

Binaries will be available at:
- `target/release/dvh_full`
- `target/release/dvh_metrics_by_roi`

### Option 2: Install binaries with Cargo

```bash
cargo install --git https://github.com/NewLeaf-ai/aitrium-dvh.git --tag v0.1.0 --bin dvh_full --bin dvh_metrics_by_roi
```

## CLI Usage

### 1) Compute DVHs for all ROIs

```bash
dvh_full /path/to/dicom_dir > dvh.json
```

`dvh_full` expects a directory containing both an `RTSTRUCT` file and an `RTDOSE` file.

Common flags:
- `--interpolate` enable XY interpolation
- `--z-segments <N>` interpolation segments between dose planes
- `--include-outside-dose` include contours outside dose grid
- `--use-extents` use structure extents optimization
- `--debug` enable debug logging

### 2) Compute ROI metrics for explicit ROI numbers

```bash
dvh_metrics_by_roi \
  --rtstruct /path/to/RTSTRUCT.dcm \
  --rtdose /path/to/RTDOSE.dcm \
  --roi 1 --roi 2 --roi 7
```

This returns JSON with key metrics (for example `d95Gy`, `d2Gy`, `meanGy`, `v20Pct`, `v30Pct`) for each requested ROI.

## Library Usage

Add dependency in your `Cargo.toml`:

```toml
[dependencies]
aitrium_dvh = { git = "https://github.com/NewLeaf-ai/aitrium-dvh.git", tag = "v0.1.0", default-features = false, features = ["memmap"] }
```

Example:

```rust
use aitrium_dvh::{compute_dvh, DvhOptions};

let options = DvhOptions::default();
let result = compute_dvh(
    "/path/to/RTSTRUCT.dcm",
    "/path/to/RTDOSE.dcm",
    1,
    &options,
)?;

println!("ROI volume (cc): {}", result.total_volume_cc);
println!("Mean dose (Gy): {}", result.stats.mean_gy);
```

## Input Expectations

- DICOM radiotherapy inputs (`RTSTRUCT`, `RTDOSE`) must be valid and spatially aligned for meaningful results.
- ROI numbers passed to `compute_dvh` / `dvh_metrics_by_roi` must exist in the referenced `RTSTRUCT`.

## Development

```bash
cargo test
cargo check
```

## License

MIT
