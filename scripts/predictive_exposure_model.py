import os
import sys

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ambiental.config import (
    PLOTS_DIR,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_project_directories,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# CONFIGURATION
# =========================
ensure_project_directories()


# =========================
# STEP 1: LOAD PROCESSED DATA
# =========================
df = pd.read_csv(PROCESSED_DATA_PATH)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)


# =========================
# STEP 2: CREATE FUTURE TARGET
# =========================
# Predict exposure 1 hour ahead
df["future_exposure"] = df["multi_pollutant_exposure"].shift(-1)


# =========================
# STEP 3: FEATURE SELECTION
# =========================
features = [
    "CO(GT)",
    "NO2(GT)",
    "NOx(GT)",
    "C6H6(GT)",
    "pollution_score",
    "contextual_pollution_score",
    "T",
    "RH",
    "AH",
    # "multi_pollutant_exposure",
    "hour",
    "hour_weight",
    "exposure_trend"
]

target = "future_exposure"

df_model = df.dropna(subset=features + [target]).copy()

X = df_model[features]
y = df_model[target]


# =========================
# STEP 4: TIME-SERIES TRAIN/TEST SPLIT
# =========================
split_index = int(len(df_model) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# =========================
# STEP 5: TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================
# STEP 6: PREDICT FUTURE EXPOSURE
# =========================
y_pred = model.predict(X_test)

df_model.loc[X_test.index, "predicted_future_exposure"] = y_pred


# =========================
# STEP 7: MODEL EVALUATION
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPrediction Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R^2:   {r2:.4f}")


# =========================
# STEP 8: PLOT ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(12, 6))

plt.plot(y_test.index, y_test.values, label="Actual Future Exposure")
plt.plot(y_test.index, y_pred, label="Predicted Future Exposure", alpha=0.8)

plt.title("Actual vs Predicted Future Exposure")
plt.xlabel("Time")
plt.ylabel("Exposure Score")
plt.legend()
plt.grid()

plt.savefig(
    PLOTS_DIR / "actual_vs_predicted_future_exposure.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()


# =========================
# STEP 9: FEATURE IMPORTANCE
# =========================
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.title("Feature Importance for Future Exposure Prediction")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")

plt.savefig(
    PLOTS_DIR / "future_exposure_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()


# =========================
# STEP 10: PROACTIVE ALERT SYSTEM
# =========================
future_high_threshold = df_model["multi_pollutant_exposure"].quantile(0.75)
future_critical_threshold = df_model["multi_pollutant_exposure"].quantile(0.90)


def proactive_alert(predicted_exposure):
    if predicted_exposure >= future_critical_threshold:
        return "PROACTIVE CRITICAL: Very high exposure expected in next hour"
    elif predicted_exposure >= future_high_threshold:
        return "PROACTIVE WARNING: High exposure expected in next hour"
    else:
        return "NO FUTURE RISK"


df_model.loc[:, "proactive_alert"] = "NO PREDICTION"
df_model.loc[X_test.index, "proactive_alert"] = [
    proactive_alert(value) for value in y_pred
]


# =========================
# STEP 11: PLOT PROACTIVE ALERTS
# =========================
test_results = df_model.loc[X_test.index].copy()

warning_events = test_results[
    test_results["proactive_alert"].str.startswith("PROACTIVE WARNING")
]

critical_events = test_results[
    test_results["proactive_alert"].str.startswith("PROACTIVE CRITICAL")
]

plt.figure(figsize=(12, 6))

plt.plot(
    test_results.index,
    test_results["predicted_future_exposure"],
    label="Predicted Future Exposure"
)

plt.scatter(
    warning_events.index,
    warning_events["predicted_future_exposure"],
    label="Proactive Warning",
    alpha=0.6
)

plt.scatter(
    critical_events.index,
    critical_events["predicted_future_exposure"],
    label="Proactive Critical",
    alpha=0.8
)

plt.axhline(
    future_high_threshold,
    linestyle="--",
    label="Future High Threshold"
)

plt.axhline(
    future_critical_threshold,
    linestyle=":",
    label="Future Critical Threshold"
)

plt.title("Proactive Alerts Based on Predicted Exposure")
plt.xlabel("Time")
plt.ylabel("Predicted Exposure Score")
plt.legend()
plt.grid()

plt.savefig(
    PLOTS_DIR / "proactive_alerts.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# ALERT ACCURACY EVALUATION
# =========================

# Ground truth: was future exposure actually high?
df_model.loc[:, "actual_high"] = (
    df_model["future_exposure"] >= future_high_threshold
)

# Prediction: did system raise warning or critical alert?
df_model.loc[:, "predicted_high"] = df_model["proactive_alert"].isin([
    "PROACTIVE WARNING: High exposure expected in next hour",
    "PROACTIVE CRITICAL: Very high exposure expected in next hour"
])

# Only evaluate on test set (where predictions exist)
eval_df = df_model.loc[X_test.index].copy()

from sklearn.metrics import classification_report, confusion_matrix

y_true = eval_df["actual_high"]
y_pred = eval_df["predicted_high"]

print("\n=== ALERT SYSTEM EVALUATION ===")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))


# =========================
# STEP 12: PROACTIVE RECOMMENDATIONS
# =========================
def proactive_recommendation(row):
    if row["proactive_alert"].startswith("PROACTIVE CRITICAL"):
        return (
            "Very high air quality risk is expected in the next hour. "
            "Take preventive action immediately: reduce exposure, close windows, "
            "and activate filtration if available."
        )

    elif row["proactive_alert"].startswith("PROACTIVE WARNING"):
        return (
            "High air quality risk is expected in the next hour. "
            "Limit outdoor activity and monitor air quality."
        )

    else:
        return "No preventive action required."


df_model.loc[:, "proactive_recommendation"] = df_model.apply(
    proactive_recommendation,
    axis=1
)


# =========================
# STEP 13: SAVE RESULTS
# =========================
predictive_results_path = RESULTS_DIR / "predictive_exposure_results.csv"
df_model.to_csv(predictive_results_path)

print("\nProactive alert counts:")
print(df_model["proactive_alert"].value_counts())

print("\nExample proactive system output:")
print(
    df_model[
        [
            "multi_pollutant_exposure",
            "future_exposure",
            "predicted_future_exposure",
            "proactive_alert",
            "proactive_recommendation"
        ]
    ].dropna().head(10)
)

print("\nSaved predictive results as:", predictive_results_path)
print("Saved plots in:", PLOTS_DIR)
