import os
import sys

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ambiental.config import PLOTS_DIR, PROJECT_ROOT, RESULTS_DIR

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from ambiental.modeling_utils import (
    MANUAL_FEATURES_WITH_METEOROLOGY,
    available_features,
    chronological_split,
    create_horizon_targets,
    ensure_output_dirs,
    evaluate_random_forest_experiment,
    load_processed_data,
    save_results,
)


# =========================
# CONFIGURATION
# =========================
EXPOSURE_WINDOWS = [6, 12, 24]
USER_PROFILE = "sensitive"

PROFILE_MULTIPLIER = {
    "normal": 1.0,
    "sensitive": 0.85,
    "athlete": 0.80,
    "elderly": 0.75,
}


def derive_window_thresholds(train_df, exposure_col):
    multiplier = PROFILE_MULTIPLIER.get(USER_PROFILE, 1.0)

    low = train_df[exposure_col].quantile(0.25)
    medium = train_df[exposure_col].quantile(0.50)
    high = train_df[exposure_col].quantile(0.75)
    spike = train_df["contextual_pollution_score"].quantile(0.85)

    return {
        "low_threshold": float(low),
        "medium_threshold": float(medium),
        "high_threshold": float(high),
        "adjusted_low_threshold": float(low * multiplier),
        "adjusted_medium_threshold": float(medium * multiplier),
        "adjusted_high_threshold": float(high * multiplier),
        "spike_threshold": float(spike),
    }


def assign_risk_level(value, thresholds):
    if value < thresholds["adjusted_low_threshold"]:
        return "Low"
    if value < thresholds["adjusted_medium_threshold"]:
        return "Moderate"
    if value < thresholds["adjusted_high_threshold"]:
        return "High"
    return "Very High"


def add_window_alert_columns(df, exposure_col, thresholds):
    result = df.copy()

    # These columns measure how often the exposure window would trigger elevated
    # risk under thresholds learned from the training portion only.
    result["window_risk_level"] = result[exposure_col].apply(
        lambda value: assign_risk_level(value, thresholds)
    )
    result["window_high_or_very_high_alert"] = result[
        "window_risk_level"
    ].isin(["High", "Very High"])
    result["window_very_high_alert"] = (
        result["window_risk_level"] == "Very High"
    )
    result["window_pollution_spike"] = (
        result["contextual_pollution_score"] > thresholds["spike_threshold"]
    )

    return result


def build_window_dataset(df, window_hours):
    exposure_col = f"multi_pollutant_exposure_w{window_hours}"
    result = df.copy()

    result[exposure_col] = (
        result["contextual_pollution_score"]
        .rolling(window=window_hours, min_periods=1)
        .sum()
    )
    result["exposure_trend"] = result[exposure_col].diff()
    result = create_horizon_targets(
        result,
        exposure_col=exposure_col,
        horizons=[1],
    )

    return result, exposure_col


def run_window_experiments(df):
    metric_frames = []

    for window_hours in EXPOSURE_WINDOWS:
        work_df, exposure_col = build_window_dataset(df, window_hours)
        features = available_features(work_df, MANUAL_FEATURES_WITH_METEOROLOGY)
        target = "future_exposure_h1"
        model_df = work_df.dropna(subset=features + [target]).copy()

        # Thresholds are estimated from the training slice and then applied to
        # validation/test slices to avoid look-ahead threshold leakage.
        splits = chronological_split(model_df)
        thresholds = derive_window_thresholds(
            splits["train"],
            exposure_col,
        )
        model_df = add_window_alert_columns(
            model_df,
            exposure_col,
            thresholds,
        )

        experiment_name = f"rf_window_{window_hours}h_h1"
        _, metrics, _, _ = evaluate_random_forest_experiment(
            df=model_df,
            features=features,
            target=target,
            experiment_name=experiment_name,
            horizon=1,
        )

        metrics.insert(0, "exposure_window_hours", window_hours)
        metrics["user_profile"] = USER_PROFILE
        metrics["exposure_column"] = exposure_col

        for key, value in thresholds.items():
            metrics[key] = value

        for split_name, split_df in chronological_split(model_df).items():
            split_mask = metrics["split"] == split_name
            metrics.loc[
                split_mask,
                "high_or_very_high_alert_frequency",
            ] = split_df["window_high_or_very_high_alert"].mean()
            metrics.loc[
                split_mask,
                "very_high_alert_frequency",
            ] = split_df["window_very_high_alert"].mean()
            metrics.loc[
                split_mask,
                "pollution_spike_frequency",
            ] = split_df["window_pollution_spike"].mean()

        metric_frames.append(metrics)

    return pd.concat(metric_frames, ignore_index=True)


def plot_window_thresholds(window_metrics):
    plot_df = (
        window_metrics[window_metrics["split"] == "test"]
        .sort_values("exposure_window_hours")
        .drop_duplicates(subset=["exposure_window_hours"])
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_df["exposure_window_hours"],
        plot_df["adjusted_low_threshold"],
        marker="o",
        label="Adjusted Low",
    )
    plt.plot(
        plot_df["exposure_window_hours"],
        plot_df["adjusted_medium_threshold"],
        marker="o",
        label="Adjusted Medium",
    )
    plt.plot(
        plot_df["exposure_window_hours"],
        plot_df["adjusted_high_threshold"],
        marker="o",
        label="Adjusted High",
    )
    plt.title("Risk Thresholds by Exposure Accumulation Window")
    plt.xlabel("Exposure window (hours)")
    plt.ylabel("Exposure score threshold")
    plt.legend()
    plt.grid()
    plt.savefig(
        PLOTS_DIR / "window_thresholds.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_window_alert_frequency(window_metrics):
    plot_df = window_metrics[
        window_metrics["split"] == "test"
    ].sort_values("exposure_window_hours")

    plt.figure(figsize=(10, 6))
    plt.bar(
        plot_df["exposure_window_hours"].astype(str),
        plot_df["high_or_very_high_alert_frequency"],
    )
    plt.title("High-Risk Alert Frequency by Exposure Window")
    plt.xlabel("Exposure window (hours)")
    plt.ylabel("Alert frequency")
    plt.grid(axis="y")
    plt.savefig(
        PLOTS_DIR / "window_alert_frequency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_window_prediction_quality(window_metrics):
    plot_df = window_metrics[
        window_metrics["split"] == "test"
    ].sort_values("exposure_window_hours")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, metric in zip(axes, ["rmse", "mae", "r2"]):
        axis.plot(
            plot_df["exposure_window_hours"],
            plot_df[metric],
            marker="o",
        )
        axis.set_title(metric.upper() if metric != "r2" else "R^2")
        axis.set_xlabel("Exposure window (hours)")
        axis.grid()

    fig.suptitle("Prediction Quality by Exposure Accumulation Window")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "window_prediction_quality.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    ensure_output_dirs()
    df = load_processed_data()

    window_metrics = run_window_experiments(df)
    save_results(window_metrics, "window_comparison.csv")

    plot_window_thresholds(window_metrics)
    plot_window_alert_frequency(window_metrics)
    plot_window_prediction_quality(window_metrics)

    print("\nWindow experiment completed.")
    print("Saved result table:", RESULTS_DIR / "window_comparison.csv")
    print("Saved plots in:", PLOTS_DIR)

    print("\nTest-set window summary:")
    print(
        window_metrics[
            window_metrics["split"] == "test"
        ][
            [
                "exposure_window_hours",
                "adjusted_high_threshold",
                "high_or_very_high_alert_frequency",
                "rmse",
                "mae",
                "r2",
                "precision",
                "recall",
                "f1_score",
                "alert_accuracy",
            ]
        ]
    )


if __name__ == "__main__":
    main()
