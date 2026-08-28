# Ambient Intelligence Air Quality Exposure Modeling

This repository contains an Ambient Intelligence air-quality exposure modeling
project built on the UCI Air Quality Dataset. It cleans hourly observations,
computes context-aware multi-pollutant exposure, generates personalized risk
alerts, and predicts future exposure with machine-learning models.

The codebase is organized as a small research software project: reusable
utilities live under `src/ambiental/`, runnable workflows live under
`scripts/`, data lives under `data/`, and generated artifacts live under
`outputs/`.

## Repository Structure

```text
ambiental/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── AirQualityUCI.csv
│   │   └── AirQualityUCI.xlsx
│   └── processed/
│       └── processed_exposure_data.csv
├── outputs/
│   ├── plots/
│   │   └── .gitkeep
│   └── results/
│       └── .gitkeep
├── src/
│   └── ambiental/
│       ├── __init__.py
│       ├── config.py
│       ├── modeling_utils.py
│       └── plotting.py
├── scripts/
│   ├── clean_dataset.py
│   ├── exposure_model.py
│   ├── linear_regression_baseline.py
│   ├── predictive_exposure_model.py
│   ├── advanced_predictive_experiments.py
│   ├── summarize_exposure_model.py
│   └── window_experiment.py
└── tests/
    └── __init__.py
```

Generated plots and result tables are preserved locally in `outputs/`, but
future generated files in `outputs/plots/` and `outputs/results/` are ignored by
Git. The `.gitkeep` files keep those directories present in a fresh checkout.

## Path Configuration

All project paths are centralized in `src/ambiental/config.py` using
`pathlib.Path`.

Important paths:

- `RAW_CSV_PATH`: `data/raw/AirQualityUCI.csv`
- `RAW_XLSX_PATH`: `data/raw/AirQualityUCI.xlsx`
- `PROCESSED_DATA_PATH`: `data/processed/processed_exposure_data.csv`
- `PLOTS_DIR`: `outputs/plots`
- `RESULTS_DIR`: `outputs/results`

Scripts can be run from the repository root without depending on hardcoded
current-working-directory paths.

## Dataset

Raw inputs:

- `data/raw/AirQualityUCI.csv`
- `data/raw/AirQualityUCI.xlsx`

Processed dataset:

- `data/processed/processed_exposure_data.csv`

The current processed dataset contains 9,357 hourly observations and 31
columns.

Core pollutant variables:

- `CO(GT)`
- `NO2(GT)`
- `NOx(GT)`
- `C6H6(GT)`

Meteorological variables:

- `T`
- `RH`
- `AH`

Derived exposure and intelligence variables:

- `pollution_score`
- `contextual_pollution_score`
- `multi_pollutant_exposure`
- `risk_level`
- `dominant_pollutant`
- `alert`
- `recommendation`

## Main Scripts

| File | Purpose |
| --- | --- |
| `scripts/clean_dataset.py` | Loads the raw UCI CSV, fixes decimal formatting, replaces missing-value markers, builds a datetime index, and prints a quick data-quality check. |
| `src/ambiental/plotting.py` | Generates exploratory plots for pollutants, meteorology, hourly patterns, distributions, and correlations. |
| `scripts/exposure_model.py` | Computes scaled pollutant scores, context-aware hourly weights, accumulated exposure, profile-adjusted thresholds, risk levels, intelligent alerts, dominant pollutants, and recommendations. |
| `scripts/predictive_exposure_model.py` | One-hour Random Forest future exposure model with proactive future alerts. |
| `scripts/linear_regression_baseline.py` | Linear Regression baseline for comparison against the Random Forest model. |
| `scripts/summarize_exposure_model.py` | Console summary of thresholds, model metrics, coefficients, and feature importances. |
| `scripts/advanced_predictive_experiments.py` | Research runner for meteorological features, multi-horizon forecasting, tsfresh features, feature importance, and advanced evaluation. |
| `scripts/window_experiment.py` | Compares 6-hour, 12-hour, and 24-hour exposure accumulation windows. |

Shared utilities:

- `src/ambiental/config.py`
- `src/ambiental/modeling_utils.py`

## Modeling Workflow

1. Clean and parse the raw UCI dataset.
2. Scale the four pollutant concentrations.
3. Compute a weighted multi-pollutant pollution score.
4. Apply context-aware hour weighting for higher-risk periods.
5. Accumulate exposure over a rolling window.
6. Adjust risk thresholds by user profile.
7. Generate risk levels, alerts, and recommendations.
8. Create future exposure targets using time shifts.
9. Train time-ordered predictive models.
10. Evaluate regression quality and proactive alert quality.
11. Save result tables to `outputs/results/` and plots to `outputs/plots/`.

## Exposure Intelligence

The base pollution score combines scaled pollutant values with fixed weights:

| Pollutant | Weight |
| --- | ---: |
| `CO(GT)` | 0.30 |
| `NO2(GT)` | 0.25 |
| `NOx(GT)` | 0.25 |
| `C6H6(GT)` | 0.20 |

The score is adjusted by hour of day:

| Time period | Hour weight |
| --- | ---: |
| Morning commute, 07:00-10:00 | 1.3 |
| Evening commute, 17:00-21:00 | 1.4 |
| Overnight, 00:00-05:00 | 0.8 |
| Other hours | 1.0 |

The default user profile is `sensitive`, which multiplies risk thresholds by
0.85. This makes alerts more conservative for users who may be more vulnerable
to pollution exposure.

## Predictive Features

The current manual feature set is:

- `CO(GT)`
- `NO2(GT)`
- `NOx(GT)`
- `C6H6(GT)`
- `pollution_score`
- `contextual_pollution_score`
- `T`
- `RH`
- `AH`
- `hour`
- `hour_weight`
- `exposure_trend`

The advanced tsfresh experiments add selected time-series features extracted
from causal sliding windows over:

- `CO(GT)`
- `NO2(GT)`
- `NOx(GT)`
- `C6H6(GT)`
- `pollution_score`
- `multi_pollutant_exposure`

## Time-Series Methodology

The advanced experiments use a chronological split:

- 70% training
- 10% validation
- 20% test

The split preserves time ordering. Future targets are created by shifting
`multi_pollutant_exposure` forward in time:

- `future_exposure_h1`
- `future_exposure_h3`
- `future_exposure_h6`
- `future_exposure_h12`

Separate Random Forest models are trained for each horizon. Alert thresholds
for predictive evaluation are derived from the training split and then applied
to validation and test data. This avoids using future distribution information
when evaluating proactive alerts.

## Current Results

### Legacy One-Hour Models

The updated legacy scripts include meteorological features.

| Model | RMSE | MAE | R^2 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.2715 | 0.1896 | 0.8840 |
| Linear Regression | 0.3464 | 0.2641 | 0.8112 |

The Random Forest remains the stronger one-hour exposure predictor.

### Meteorological Feature Comparison

Adding `T`, `RH`, and `AH` improves the one-hour Random Forest model slightly.

| Experiment | RMSE | MAE | R^2 | Alert Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Manual features only | 0.2783 | 0.1957 | 0.8781 | 0.9065 |
| Manual + meteorology | 0.2745 | 0.1926 | 0.8815 | 0.9092 |

Finding: meteorological variables provide a modest but consistent improvement.

### Multi-Horizon Forecasting

Random Forest models were trained separately for 1-hour, 3-hour, 6-hour, and
12-hour forecasting horizons.

| Horizon | RMSE | MAE | R^2 | Precision | Recall | F1 | Alert Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 0.2745 | 0.1926 | 0.8815 | 0.8748 | 0.8748 | 0.8748 | 0.9092 |
| 3h | 0.2584 | 0.1901 | 0.8949 | 0.8649 | 0.8675 | 0.8662 | 0.9027 |
| 6h | 0.4664 | 0.3492 | 0.6577 | 0.7185 | 0.7894 | 0.7523 | 0.8113 |
| 12h | 0.6053 | 0.4547 | 0.4234 | 0.6125 | 0.8056 | 0.6959 | 0.7442 |

Finding: short-horizon forecasting is reliable, while 6-hour and 12-hour
forecasts are more difficult and show lower R^2 and alert accuracy.

### TSFRESH Feature Experiments

tsfresh features were extracted from 6-hour, 12-hour, and 24-hour causal
sliding windows. Each tsfresh model uses:

- 12 manual/met features
- 80 selected tsfresh features
- 92 total features

| Experiment | RMSE | MAE | R^2 | Precision | Recall | F1 | Alert Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual + meteorology | 0.2745 | 0.1926 | 0.8815 | 0.8748 | 0.8748 | 0.8748 | 0.9092 |
| Manual + meteorology + tsfresh 6h | 0.1019 | 0.0712 | 0.9837 | 0.9502 | 0.9558 | 0.9530 | 0.9658 |
| Manual + meteorology + tsfresh 12h | 0.1448 | 0.1009 | 0.9670 | 0.9134 | 0.9470 | 0.9299 | 0.9481 |
| Manual + meteorology + tsfresh 24h | 0.1939 | 0.1418 | 0.9409 | 0.9216 | 0.9012 | 0.9113 | 0.9363 |

Finding: tsfresh features substantially improve one-hour forecasting. The
6-hour tsfresh window is the strongest configuration in the current
experiments.

### Exposure Window Comparison

The project compares exposure accumulation windows of 6, 12, and 24 hours.

| Exposure Window | Adjusted High Threshold | High Alert Frequency | RMSE | MAE | R^2 | F1 | Alert Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6h | 1.4685 | 70.35% | 0.2745 | 0.1926 | 0.8815 | 0.8756 | 0.9097 |
| 12h | 2.7801 | 80.34% | 0.6460 | 0.4963 | 0.7070 | 0.7825 | 0.8403 |
| 24h | 5.3027 | 79.81% | 1.4173 | 1.1331 | 0.4185 | 0.7163 | 0.7334 |

Finding: longer accumulation windows increase high-risk alert frequency but
reduce one-hour predictive quality. The 6-hour exposure window is currently the
best operational choice.

## Feature Importance Highlights

The strongest manual feature across one-hour Random Forest models is
`pollution_score`. In longer horizons, temporal context such as `hour` becomes
more important. In the best tsfresh model, the top extracted feature is:

- `tsfresh_w6__pollution_score__root_mean_square`

This suggests that short-term time-series structure in the pollution score is
highly informative for near-future exposure.

## Output Files

Main result tables are written to `outputs/results/`:

| File | Description |
| --- | --- |
| `outputs/results/experiment_summary.csv` | Combined advanced experiment metrics across train, validation, and test splits. |
| `outputs/results/feature_importance.csv` | Random Forest feature importances for advanced experiments. |
| `outputs/results/horizon_comparison.csv` | Multi-horizon Random Forest metrics. |
| `outputs/results/meteorological_feature_comparison.csv` | Comparison with and without meteorological variables. |
| `outputs/results/tsfresh_feature_comparison.csv` | Performance of tsfresh-enhanced models. |
| `outputs/results/tsfresh_selected_features.csv` | Final manual and tsfresh feature names used by tsfresh experiments. |
| `outputs/results/tsfresh_selected_features_w6.csv` | Selected features for the 6-hour tsfresh window. |
| `outputs/results/tsfresh_selected_features_w12.csv` | Selected features for the 12-hour tsfresh window. |
| `outputs/results/tsfresh_selected_features_w24.csv` | Selected features for the 24-hour tsfresh window. |
| `outputs/results/window_comparison.csv` | Exposure accumulation window comparison. |

Main plots are written to `outputs/plots/`:

| Plot | Description |
| --- | --- |
| `outputs/plots/contextual_vs_base_score.png` | Base pollution score compared with context-aware score. |
| `outputs/plots/multi_pollutant_exposure.png` | Accumulated multi-pollutant exposure over time. |
| `outputs/plots/exposure_with_intelligent_alerts.png` | Exposure timeline with generated alerts. |
| `outputs/plots/actual_vs_predicted_future_exposure.png` | Legacy Random Forest actual vs predicted one-hour exposure. |
| `outputs/plots/proactive_alerts.png` | Proactive future-alert visualization. |
| `outputs/plots/meteorological_feature_comparison.png` | Impact of adding meteorological variables. |
| `outputs/plots/horizon_rmse_comparison.png` | RMSE by forecast horizon. |
| `outputs/plots/horizon_metrics_comparison.png` | RMSE, MAE, and R^2 by horizon. |
| `outputs/plots/tsfresh_feature_importance.png` | Top feature importances for tsfresh experiments. |
| `outputs/plots/window_thresholds.png` | Risk thresholds by exposure accumulation window. |
| `outputs/plots/window_alert_frequency.png` | Alert frequency by exposure window. |
| `outputs/plots/window_prediction_quality.png` | Prediction quality by exposure window. |

## Reproducing the Experiments

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the full workflow from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\exposure_model.py
.\.venv\Scripts\python.exe scripts\predictive_exposure_model.py
.\.venv\Scripts\python.exe scripts\linear_regression_baseline.py
.\.venv\Scripts\python.exe scripts\advanced_predictive_experiments.py
.\.venv\Scripts\python.exe scripts\window_experiment.py
.\.venv\Scripts\python.exe scripts\summarize_exposure_model.py
```

Run exploratory plotting as a package module:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m ambiental.plotting
```

## Research Takeaways

1. Random Forest outperforms Linear Regression for one-hour future exposure.
2. Meteorological variables improve predictive performance, but the gain is
   modest compared with time-series feature extraction.
3. The best current model is the 6-hour tsfresh-enhanced Random Forest.
4. Forecasting remains strong at 1-hour and 3-hour horizons but degrades at
   6-hour and 12-hour horizons.
5. Longer exposure accumulation windows increase alert frequency but reduce
   prediction quality.
6. The current best operational setup is a 6-hour exposure accumulation window
   with manual, meteorological, and selected tsfresh features for near-term
   forecasting.

## Notes

The project avoids data leakage by preserving time order, creating targets with
future shifts only, selecting tsfresh features on the training split, and
deriving alert thresholds from training data for advanced evaluation.

# ambiental
