use aitrium_dvh::dicom_parser::parse_rtstruct;
use aitrium_dvh::engine::dvh::compute_dvh;
use aitrium_dvh::DvhOptions;
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Compute DVH metrics for explicit ROI numbers"
)]
struct Args {
    /// RTSTRUCT DICOM path
    #[arg(long)]
    rtstruct: PathBuf,

    /// RTDOSE DICOM path
    #[arg(long)]
    rtdose: PathBuf,

    /// ROI numbers (repeatable, or comma-separated list)
    #[arg(long = "roi", value_delimiter = ',', required = true)]
    roi_numbers: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct RoiMetricSet {
    #[serde(rename = "d95Gy")]
    d95_gy: f64,
    #[serde(rename = "d2Gy")]
    d2_gy: f64,
    #[serde(rename = "meanGy")]
    mean_gy: f64,
    #[serde(rename = "v20Pct")]
    v20_pct: f64,
    #[serde(rename = "v30Pct")]
    v30_pct: f64,
}

#[derive(Debug, Serialize)]
struct RoiMetricOutput {
    roi_number: i32,
    roi_name: String,
    metrics: RoiMetricSet,
}

#[derive(Debug, Serialize)]
struct DvhMetricsOutput {
    schema_version: &'static str,
    rtstruct_path: String,
    rtdose_path: String,
    rois: Vec<RoiMetricOutput>,
}

fn volume_percent_at_dose_gy(
    differential_hist_cgy: &[f64],
    total_volume_cc: f64,
    dose_gy: f64,
) -> f64 {
    if differential_hist_cgy.is_empty() || total_volume_cc <= 0.0 {
        return 0.0;
    }
    let dose_cgy = (dose_gy * 100.0).round() as usize;
    if dose_cgy >= differential_hist_cgy.len() {
        return 0.0;
    }
    let volume_above_cc: f64 = differential_hist_cgy[dose_cgy..].iter().sum();
    (volume_above_cc / total_volume_cc * 100.0).clamp(0.0, 100.0)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let options = DvhOptions::default();
    let available_rois = parse_rtstruct(&args.rtstruct)?;

    let mut outputs = Vec::new();
    for roi_number in args.roi_numbers {
        let roi = available_rois
            .iter()
            .find(|entry| entry.id == roi_number)
            .ok_or_else(|| anyhow::anyhow!("ROI {} not found in RTSTRUCT", roi_number))?;
        let result = compute_dvh(&args.rtstruct, &args.rtdose, roi_number, &options)?;
        let total_cc = if result.total_volume_cc > 0.0 {
            result.total_volume_cc
        } else {
            result.stats.total_cc
        };
        outputs.push(RoiMetricOutput {
            roi_number,
            roi_name: roi.name.clone(),
            metrics: RoiMetricSet {
                d95_gy: result.stats.d95_gy,
                d2_gy: result.stats.d2_gy,
                mean_gy: result.stats.mean_gy,
                v20_pct: volume_percent_at_dose_gy(&result.differential_hist_cgy, total_cc, 20.0),
                v30_pct: volume_percent_at_dose_gy(&result.differential_hist_cgy, total_cc, 30.0),
            },
        });
    }
    outputs.sort_by_key(|entry| entry.roi_number);

    let payload = DvhMetricsOutput {
        schema_version: "1.0.0",
        rtstruct_path: args.rtstruct.display().to_string(),
        rtdose_path: args.rtdose.display().to_string(),
        rois: outputs,
    };
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
