# Figures and Provenance Catalog

All figures are generated directly from experimental outputs by `python -m resqcity.experiments.figures.generate`.
Plots adhere to colorblind-safe publication palettes (Okabe-Ito / Wong) and export in dual PDF (vector) and PNG (raster) formats.

| Figure | Filename | Description | Data Provenance |
|---|---|---|---|
| **Figure 1** | `results/figures/primary_comparison.png` | Boxplot comparison of primary response times (time to last required arrival) across all policies (B0, B1, B2, B3, FULL) for the normal scenario. | `results/experiments/summary_normal_*.json` |
| **Figure 2** | `results/figures/response_cdf_normal.png` | Empirical Cumulative Distribution Functions (CDFs) of primary arrival times, highlighting compliance at the 8-minute and 15-minute national benchmarks. | `results/experiments/metrics_normal_*.parquet` |
| **Figure 3** | `results/figures/ward_equity_normal.png` | Distribution of 90th-percentile response times across the 58 Pune municipal electoral wards, showing spatial equity across policies. | `results/experiments/metrics_normal_*.parquet` |
| **Figure 4** | `results/figures/utilisation_normal.png` | Bar chart of fleet utilisation fractions across ambulance, police, and fire services, confirming ~41% target background workload calibration. | `results/experiments/summary_normal_*.json` |
| **Figure 5** | `results/figures/detection_breakdown_normal.png` | Comparison of initial detection delay between automated CCTV camera detections and civilian Dial-112 distress calls. | `results/experiments/metrics_normal_*.parquet` |
