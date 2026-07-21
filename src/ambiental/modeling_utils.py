import numpy as np
import pandas as pd

from ambiental.config import (
    PLOTS_DIR,
    PROCESSED_DATA_PATH,
    RESULTS_DIR,
    ensure_project_directories,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# =========================
# FEATURE CONFIGURATION
# =========================
BASE_MANUAL_FEATURES = [
    "CO(GT)",
    "NO2(GT)",
    "NOx(GT)",
    "C6H6(GT)",
    "pollution_score",
    "contextual_pollution_score",
    "hour",
    "hour_weight",
    "exposure_trend",
]

METEOROLOGICAL_FEATURES = [
    "T",
    "RH",
    "AH",
]

MANUAL_FEATURES_WITH_METEOROLOGY = (
    BASE_MANUAL_FEATURES + METEOROLOGICAL_FEATURES
)

POLLUTANT_COLUMNS = [
    "CO(GT)",
    "NO2(GT)",
    "NOx(GT)",
    "C6H6(GT)",
]

TSFRESH_SIGNAL_COLUMNS = [
    "CO(GT)",
    "NO2(GT)",
    "NOx(GT)",
    "C6H6(GT)",
    "pollution_score",
    "multi_pollutant_exposure",
]

HORIZONS = [1, 3, 6, 12]
EXPOSURE_WINDOWS = [6, 12, 24]


def ensure_output_dirs():
    ensure_project_directories()


def load_processed_data(data_path=PROCESSED_DATA_PATH):
    df = pd.read_csv(data_path)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    # Older result files may not contain this column if they were edited by hand.
    if "exposure_trend" not in df.columns:
        df["exposure_trend"] = df["multi_pollutant_exposure"].diff()

    return df


def available_features(df, requested_features):
    missing_features = [
        feature for feature in requested_features if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: " + ", ".join(missing_features)
        )

    return list(requested_features)


def create_horizon_targets(
    df,
    exposure_col="multi_pollutant_exposure",
    horizons=None,
):
    horizons = HORIZONS if horizons is None else horizons
    df_targets = df.copy()

    # Negative shifts keep every row's features at time t and move only labels
    # into the future, preserving the causal forecasting setup.
    for horizon in horizons:
        target = f"future_exposure_h{horizon}"
        df_targets[target] = df_targets[exposure_col].shift(-horizon)

    return df_targets


def chronological_split(
    df,
    train_fraction=0.70,
    validation_fraction=0.10,
):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    train_end = int(len(df) * train_fraction)
    validation_end = int(len(df) * (train_fraction + validation_fraction))

    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
    )


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def derive_alert_thresholds(y_train):
    return {
        "future_high_threshold": float(y_train.quantile(0.75)),
        "future_critical_threshold": float(y_train.quantile(0.90)),
    }


def alert_metrics(y_true, y_pred, high_threshold):
    actual_high = y_true >= high_threshold
    predicted_high = y_pred >= high_threshold

    return {
        "precision": float(
            precision_score(actual_high, predicted_high, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_high, predicted_high, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual_high, predicted_high, zero_division=0)
        ),
        "alert_accuracy": float(accuracy_score(actual_high, predicted_high)),
        "actual_alert_frequency": float(actual_high.mean()),
        "predicted_alert_frequency": float(predicted_high.mean()),
    }


def evaluate_random_forest_experiment(
    df,
    features,
    target,
    experiment_name,
    horizon,
):
    model_df = df.dropna(subset=features + [target]).copy()

    if model_df.empty:
        raise ValueError(f"No rows available for experiment {experiment_name}.")

    splits = chronological_split(model_df)

    X_train = splits["train"][features]
    y_train = splits["train"][target]

    model = build_random_forest()
    model.fit(X_train, y_train)

    thresholds = derive_alert_thresholds(y_train)
    rows = []
    predictions = {}

    # Evaluation uses the training-derived threshold for every split. This keeps
    # alert classification from learning distribution cutoffs from validation or
    # test observations.
    for split_name, split_df in splits.items():
        X_split = split_df[features]
        y_split = split_df[target]
        y_pred = model.predict(X_split)

        row = {
            "experiment": experiment_name,
            "horizon": horizon,
            "target": target,
            "split": split_name,
            "n_train": len(splits["train"]),
            "n_validation": len(splits["validation"]),
            "n_test": len(splits["test"]),
            "n_eval": len(split_df),
            "n_features": len(features),
            **thresholds,
            **regression_metrics(y_split, y_pred),
            **alert_metrics(
                y_split,
                y_pred,
                thresholds["future_high_threshold"],
            ),
        }

        rows.append(row)
        predictions[split_name] = pd.DataFrame(
            {
                "actual": y_split,
                "predicted": y_pred,
            },
            index=split_df.index,
        )

    importance = pd.DataFrame(
        {
            "experiment": experiment_name,
            "horizon": horizon,
            "target": target,
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        by=["experiment", "horizon", "importance"],
        ascending=[True, True, False],
    )

    return model, pd.DataFrame(rows), predictions, importance


def save_results(df, filename):
    ensure_output_dirs()
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path
