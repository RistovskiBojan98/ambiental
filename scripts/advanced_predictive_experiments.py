import ast
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
import numpy as np
import pandas as pd

from ambiental.modeling_utils import (
    BASE_MANUAL_FEATURES,
    HORIZONS,
    MANUAL_FEATURES_WITH_METEOROLOGY,
    TSFRESH_SIGNAL_COLUMNS,
    available_features,
    chronological_split,
    create_horizon_targets,
    ensure_output_dirs,
    evaluate_random_forest_experiment,
    load_processed_data,
    save_results,
)


# =========================
# TSFRESH CONFIGURATION
# =========================
# A compact set of interpretable feature calculators keeps the experiment
# tractable on the UCI dataset while still capturing level, variability, trend,
# short-lag dependence, and local shape.
TSFRESH_FC_PARAMETERS = {
    "mean": None,
    "median": None,
    "minimum": None,
    "maximum": None,
    "standard_deviation": None,
    "variance": None,
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "root_mean_square": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "linear_trend": [
        {"attr": "slope"},
        {"attr": "intercept"},
        {"attr": "rvalue"},
    ],
    "number_peaks": [{"n": 1}, {"n": 3}],
}

TSFRESH_WINDOWS = [6, 12, 24]
MAX_SELECTED_TSFRESH_FEATURES = 80


def plot_single_metric(df, metric, output_path, title, label_col):
    plot_df = df[df["split"] == "test"].copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df[label_col].astype(str), plot_df[metric])
    plt.title(title)
    plt.xlabel(label_col.replace("_", " ").title())
    plt.ylabel(metric.upper() if metric != "r2" else "R^2")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_horizon_metrics(df, output_path):
    plot_df = df[df["split"] == "test"].copy().sort_values("horizon")
    metrics = ["rmse", "mae", "r2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, metric in zip(axes, metrics):
        axis.plot(
            plot_df["horizon"],
            plot_df[metric],
            marker="o",
        )
        axis.set_title(metric.upper() if metric != "r2" else "R^2")
        axis.set_xlabel("Forecast horizon (hours)")
        axis.grid()

    fig.suptitle("Random Forest Performance by Forecast Horizon")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_feature_importance(feature_importance, output_path, title):
    if feature_importance.empty:
        return

    top_features = feature_importance.sort_values(
        by="importance",
        ascending=False,
    ).head(20)

    plt.figure(figsize=(12, 8))
    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_meteorological_feature_comparison(df):
    df_targets = create_horizon_targets(df, horizons=[1])

    experiments = [
        ("rf_manual_no_meteorology_h1", BASE_MANUAL_FEATURES),
        ("rf_manual_with_meteorology_h1", MANUAL_FEATURES_WITH_METEOROLOGY),
    ]

    metric_frames = []
    importance_frames = []

    # The target is identical across both runs; only T/RH/AH are added.
    for experiment_name, requested_features in experiments:
        features = available_features(df_targets, requested_features)
        _, metrics, _, importance = evaluate_random_forest_experiment(
            df=df_targets,
            features=features,
            target="future_exposure_h1",
            experiment_name=experiment_name,
            horizon=1,
        )
        metric_frames.append(metrics)
        importance_frames.append(importance)

    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        importance_frames,
        ignore_index=True,
    )


def run_multi_horizon_experiments(df):
    df_targets = create_horizon_targets(df, horizons=HORIZONS)
    features = available_features(df_targets, MANUAL_FEATURES_WITH_METEOROLOGY)

    metric_frames = []
    importance_frames = []

    # Separate models prevent the long-horizon task from contaminating the
    # short-horizon model and keep each evaluation horizon explicit.
    for horizon in HORIZONS:
        target = f"future_exposure_h{horizon}"
        experiment_name = f"rf_manual_meteorology_h{horizon}"

        _, metrics, _, importance = evaluate_random_forest_experiment(
            df=df_targets,
            features=features,
            target=target,
            experiment_name=experiment_name,
            horizon=horizon,
        )
        metric_frames.append(metrics)
        importance_frames.append(importance)

    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        importance_frames,
        ignore_index=True,
    )


def _endpoint_from_rolled_id(rolled_id):
    if isinstance(rolled_id, tuple):
        return int(rolled_id[-1])

    if isinstance(rolled_id, (np.integer, int)):
        return int(rolled_id)

    try:
        parsed = ast.literal_eval(str(rolled_id))
        if isinstance(parsed, tuple):
            return int(parsed[-1])
        return int(parsed)
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(f"Cannot parse rolled tsfresh id: {rolled_id}")


def extract_tsfresh_window_features(df, window_hours):
    try:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        from tsfresh.utilities.dataframe_functions import roll_time_series
    except ImportError as exc:
        raise ImportError(
            "tsfresh is required for Phase 3. Install dependencies from "
            "requirements.txt before running this experiment."
        ) from exc

    source = df[TSFRESH_SIGNAL_COLUMNS].copy()
    source = source.replace([np.inf, -np.inf], np.nan)
    source = source.interpolate(method="time")
    source = source.ffill().bfill()

    tsfresh_input = source.reset_index(drop=True)
    tsfresh_input.insert(0, "time_index", np.arange(len(tsfresh_input)))
    tsfresh_input.insert(0, "id", "uci_air_quality")

    rolled = roll_time_series(
        tsfresh_input,
        column_id="id",
        column_sort="time_index",
        max_timeshift=window_hours - 1,
        min_timeshift=window_hours - 1,
        rolling_direction=1,
        disable_progressbar=True,
        n_jobs=1,
    )

    features = extract_features(
        rolled,
        column_id="id",
        column_sort="time_index",
        default_fc_parameters=TSFRESH_FC_PARAMETERS,
        disable_progressbar=True,
        n_jobs=1,
    )

    impute(features)

    endpoints = [_endpoint_from_rolled_id(value) for value in features.index]
    features.index = pd.Index(df.index[endpoints], name="Datetime")
    features = features.add_prefix(f"tsfresh_w{window_hours}__")

    return features


def select_tsfresh_features(X_train_tsfresh, y_train, window_hours):
    try:
        from tsfresh.feature_selection.relevance import (
            calculate_relevance_table,
        )
    except ImportError as exc:
        raise ImportError(
            "tsfresh is required for Phase 3 feature selection."
        ) from exc

    X_clean = X_train_tsfresh.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.median(numeric_only=True))
    X_clean = X_clean.fillna(0.0)

    relevance_table = calculate_relevance_table(
        X_clean,
        y_train,
        ml_task="regression",
    )
    relevance_table = relevance_table.sort_values(
        by=["relevant", "p_value"],
        ascending=[False, True],
    )
    relevance_table.insert(0, "window_hours", window_hours)

    selected = relevance_table[relevance_table["relevant"]]["feature"].tolist()

    # If the strict multiple-test correction selects nothing, keep a small
    # lowest-p-value fallback so the experiment still reports a transparent
    # tsfresh-augmented comparison.
    if not selected:
        selected = (
            relevance_table.dropna(subset=["p_value"])
            .head(min(20, len(relevance_table)))
            ["feature"]
            .tolist()
        )

    selected = selected[:MAX_SELECTED_TSFRESH_FEATURES]
    relevance_table["selected_for_model"] = relevance_table["feature"].isin(
        selected
    )

    return selected, relevance_table


def build_selected_feature_table(
    window_hours,
    horizon,
    manual_features,
    selected_tsfresh_features,
):
    manual_rows = [
        {
            "window_hours": window_hours,
            "horizon": horizon,
            "feature_name": feature,
            "feature_type": "manual",
        }
        for feature in manual_features
    ]

    tsfresh_rows = [
        {
            "window_hours": window_hours,
            "horizon": horizon,
            "feature_name": feature,
            "feature_type": "tsfresh",
        }
        for feature in selected_tsfresh_features
    ]

    return pd.DataFrame(manual_rows + tsfresh_rows)


def run_tsfresh_experiments(df):
    df_targets = create_horizon_targets(df, horizons=[1])
    manual_features = available_features(
        df_targets,
        MANUAL_FEATURES_WITH_METEOROLOGY,
    )

    metric_frames = []
    importance_frames = []
    selected_feature_frames = []

    for window_hours in TSFRESH_WINDOWS:
        print(f"\nExtracting tsfresh features for {window_hours}h windows...")
        tsfresh_features = extract_tsfresh_window_features(
            df_targets,
            window_hours,
        )

        combined = df_targets.join(tsfresh_features, how="left")
        tsfresh_columns = list(tsfresh_features.columns)
        target = "future_exposure_h1"
        model_df = combined.dropna(
            subset=manual_features + tsfresh_columns + [target]
        ).copy()

        splits = chronological_split(model_df)
        selected_tsfresh_features, _ = select_tsfresh_features(
            splits["train"][tsfresh_columns],
            splits["train"][target],
            window_hours,
        )

        if not selected_tsfresh_features:
            print(
                f"No usable tsfresh features selected for {window_hours}h "
                "windows; skipping this tsfresh model."
            )
            continue

        features = manual_features + selected_tsfresh_features
        experiment_name = f"rf_manual_meteorology_tsfresh_w{window_hours}_h1"
        selected_features = build_selected_feature_table(
            window_hours=window_hours,
            horizon=1,
            manual_features=manual_features,
            selected_tsfresh_features=selected_tsfresh_features,
        )
        selected_feature_frames.append(selected_features)
        save_results(
            selected_features,
            f"tsfresh_selected_features_w{window_hours}.csv",
        )

        _, metrics, _, importance = evaluate_random_forest_experiment(
            df=model_df,
            features=features,
            target=target,
            experiment_name=experiment_name,
            horizon=1,
        )

        metric_frames.append(metrics)
        importance_frames.append(importance)

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        importance = pd.concat(importance_frames, ignore_index=True)
    else:
        metrics = pd.DataFrame()
        importance = pd.DataFrame()

    if selected_feature_frames:
        selected_features = pd.concat(
            selected_feature_frames,
            ignore_index=True,
        )
    else:
        selected_features = pd.DataFrame(
            columns=[
                "window_hours",
                "horizon",
                "feature_name",
                "feature_type",
            ]
        )

    return metrics, importance, selected_features


def main():
    ensure_output_dirs()
    df = load_processed_data()

    all_metric_frames = []
    all_importance_frames = []

    # =========================
    # PHASE 1: METEOROLOGY
    # =========================
    met_metrics, met_importance = run_meteorological_feature_comparison(df)
    save_results(met_metrics, "meteorological_feature_comparison.csv")
    all_metric_frames.append(met_metrics)
    all_importance_frames.append(met_importance)

    plot_single_metric(
        met_metrics,
        "rmse",
        PLOTS_DIR / "meteorological_feature_comparison.png",
        "Meteorological Feature Comparison",
        "experiment",
    )

    # =========================
    # PHASE 2: MULTI-HORIZON
    # =========================
    horizon_metrics, horizon_importance = run_multi_horizon_experiments(df)
    save_results(horizon_metrics, "horizon_comparison.csv")
    all_metric_frames.append(horizon_metrics)
    all_importance_frames.append(horizon_importance)

    plot_single_metric(
        horizon_metrics,
        "rmse",
        PLOTS_DIR / "horizon_rmse_comparison.png",
        "Forecast Horizon RMSE Comparison",
        "horizon",
    )
    plot_horizon_metrics(
        horizon_metrics,
        PLOTS_DIR / "horizon_metrics_comparison.png",
    )

    # =========================
    # PHASE 3: TSFRESH
    # =========================
    try:
        tsfresh_metrics, tsfresh_importance, tsfresh_selected_features = (
            run_tsfresh_experiments(df)
        )
    except ImportError as exc:
        print("\nPhase 3 skipped:", exc)
        tsfresh_metrics = pd.DataFrame()
        tsfresh_importance = pd.DataFrame()
        tsfresh_selected_features = pd.DataFrame(
            columns=[
                "window_hours",
                "horizon",
                "feature_name",
                "feature_type",
            ]
        )

    save_results(tsfresh_selected_features, "tsfresh_selected_features.csv")
    print(
        "Saved selected tsfresh features to",
        RESULTS_DIR / "tsfresh_selected_features.csv",
    )

    if not tsfresh_metrics.empty:
        save_results(tsfresh_metrics, "tsfresh_feature_comparison.csv")
        all_metric_frames.append(tsfresh_metrics)
        all_importance_frames.append(tsfresh_importance)

        plot_top_feature_importance(
            tsfresh_importance,
            PLOTS_DIR / "tsfresh_feature_importance.png",
            "Top Feature Importances for TSFRESH Experiments",
        )

    # =========================
    # PHASE 5-6: SUMMARY OUTPUTS
    # =========================
    experiment_summary = pd.concat(all_metric_frames, ignore_index=True)
    feature_importance = pd.concat(all_importance_frames, ignore_index=True)

    save_results(experiment_summary, "experiment_summary.csv")
    save_results(feature_importance, "feature_importance.csv")

    print("\nAdvanced predictive experiments completed.")
    print("Saved result tables in:", RESULTS_DIR)
    print("Saved plots in:", PLOTS_DIR)

    print("\nTest-set summary:")
    print(
        experiment_summary[
            experiment_summary["split"] == "test"
        ][
            [
                "experiment",
                "horizon",
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
