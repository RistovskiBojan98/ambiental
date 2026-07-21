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
    RAW_CSV_PATH,
    ensure_project_directories,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# =========================
# CONFIGURATION
# =========================
USER_PROFILE = "sensitive"  # normal, sensitive, athlete, elderly

ensure_project_directories()


# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_csv(RAW_CSV_PATH, sep=";")

df = df.iloc[:, :-2]

for col in df.columns[2:]:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace(-200, np.nan, inplace=True)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

df = df.dropna(subset=["Datetime"])

df.drop(["Date", "Time"], axis=1, inplace=True)
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)

df.interpolate(method="linear", inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)


# =========================
# MULTI-POLLUTANT MODEL
# =========================
pollutants = ["CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)"]

scaler = MinMaxScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df[pollutants]),
    columns=[p + "_scaled" for p in pollutants],
    index=df.index
)

df = pd.concat([df, df_scaled], axis=1)


# =========================
# WEIGHTED POLLUTION SCORE
# =========================
weights = {
    "CO(GT)_scaled": 0.30,
    "NO2(GT)_scaled": 0.25,
    "NOx(GT)_scaled": 0.25,
    "C6H6(GT)_scaled": 0.20
}

df["pollution_score"] = (
    df["CO(GT)_scaled"] * weights["CO(GT)_scaled"] +
    df["NO2(GT)_scaled"] * weights["NO2(GT)_scaled"] +
    df["NOx(GT)_scaled"] * weights["NOx(GT)_scaled"] +
    df["C6H6(GT)_scaled"] * weights["C6H6(GT)_scaled"]
)


# =========================
# CONTEXT-AWARE HOUR WEIGHTING
# =========================
def hour_weight(hour):
    if 7 <= hour <= 10:
        return 1.3
    elif 17 <= hour <= 21:
        return 1.4
    elif 0 <= hour <= 5:
        return 0.8
    else:
        return 1.0


df["hour"] = df.index.hour
df["hour_weight"] = df["hour"].apply(hour_weight)

df["contextual_pollution_score"] = (
    df["pollution_score"] * df["hour_weight"]
)


# =========================
# ROLLING EXPOSURE SCORE
# =========================
window_hours = 6

df["multi_pollutant_exposure"] = (
    df["contextual_pollution_score"]
    .rolling(window=window_hours, min_periods=1)
    .sum()
)


# =========================
# USER PROFILE THRESHOLD ADJUSTMENT
# =========================
profile_multiplier = {
    "normal": 1.0,
    "sensitive": 0.85,
    "athlete": 0.80,
    "elderly": 0.75
}

threshold_multiplier = profile_multiplier.get(USER_PROFILE, 1.0)


# =========================
# DATA-DRIVEN RISK THRESHOLDS
# =========================
low_threshold = df["multi_pollutant_exposure"].quantile(0.25)
medium_threshold = df["multi_pollutant_exposure"].quantile(0.50)
high_threshold = df["multi_pollutant_exposure"].quantile(0.75)

adjusted_low_threshold = low_threshold * threshold_multiplier
adjusted_medium_threshold = medium_threshold * threshold_multiplier
adjusted_high_threshold = high_threshold * threshold_multiplier

spike_threshold = df["contextual_pollution_score"].quantile(0.85)


def exposure_risk_level(x):
    if x < adjusted_low_threshold:
        return "Low"
    elif x < adjusted_medium_threshold:
        return "Moderate"
    elif x < adjusted_high_threshold:
        return "High"
    else:
        return "Very High"


df["risk_level"] = df["multi_pollutant_exposure"].apply(exposure_risk_level)


# =========================
# TREND DETECTION
# =========================
df["exposure_trend"] = df["multi_pollutant_exposure"].diff()
df["rising_exposure"] = df["exposure_trend"] > 0


# =========================
# PERSISTENCE DETECTION
# =========================
df["high_exposure_flag"] = (
    df["multi_pollutant_exposure"] >= adjusted_high_threshold
)

df["persistent_high_exposure"] = (
    df["high_exposure_flag"]
    .rolling(window=3, min_periods=1)
    .sum() >= 2
)


# =========================
# DOMINANT POLLUTANT DETECTION
# =========================
scaled_pollutants = [
    "CO(GT)_scaled",
    "NO2(GT)_scaled",
    "NOx(GT)_scaled",
    "C6H6(GT)_scaled"
]

dominant_map = {
    "CO(GT)_scaled": "CO",
    "NO2(GT)_scaled": "NO2",
    "NOx(GT)_scaled": "NOx",
    "C6H6(GT)_scaled": "C6H6"
}

df["dominant_pollutant"] = (
    df[scaled_pollutants]
    .idxmax(axis=1)
    .map(dominant_map)
)


# =========================
# ALERT SYSTEM
# =========================
def generate_alert(row):
    current_spike = row["contextual_pollution_score"] > spike_threshold

    if row["persistent_high_exposure"] and current_spike:
        return "CRITICAL: Persistent exposure and current pollution spike"

    elif row["persistent_high_exposure"]:
        return "HIGH: Persistent accumulated exposure"

    elif row["risk_level"] == "Very High" and row["rising_exposure"]:
        return "HIGH: Exposure is increasing"

    elif current_spike:
        return "MODERATE: Short-term pollution spike"

    else:
        return "NORMAL"


df["alert"] = df.apply(generate_alert, axis=1)


# =========================
# RECOMMENDATION LAYER
# =========================
def recommendation(row):
    pollutant = row["dominant_pollutant"]

    if row["alert"].startswith("CRITICAL"):
        return (
            f"Critical risk mainly caused by {pollutant}. "
            "Avoid outdoor activity, close windows, and activate air purification."
        )

    elif row["alert"] == "HIGH: Persistent accumulated exposure":
        return (
            f"Persistent high exposure mainly caused by {pollutant}. "
            "Reduce exposure and avoid unnecessary outdoor movement."
        )

    elif row["alert"] == "HIGH: Exposure is increasing":
        return (
            f"Exposure is rising, mainly caused by {pollutant}. "
            "Monitor conditions closely and limit prolonged exposure."
        )

    elif row["alert"].startswith("MODERATE"):
        return (
            f"Short-term pollution spike mainly caused by {pollutant}. "
            "Monitor air quality and avoid prolonged exposure."
        )

    return "Air quality exposure is acceptable."

df["recommendation"] = df.apply(recommendation, axis=1)

# =========================
# PLOTS
# =========================

plt.figure(figsize=(12, 6))
df["pollution_score"].plot(label="Base Pollution Score")
df["contextual_pollution_score"].plot(
    label="Context-Aware Pollution Score",
    alpha=0.8
)
plt.title("Base Pollution Score vs Context-Aware Pollution Score")
plt.xlabel("Time")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.savefig(
    PLOTS_DIR / "contextual_vs_base_score.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()


plt.figure(figsize=(12, 6))
df["multi_pollutant_exposure"].plot()
plt.title("Context-Aware Multi-Pollutant Exposure Score")
plt.xlabel("Time")
plt.ylabel("Exposure Score")
plt.grid()
plt.savefig(
    PLOTS_DIR / "multi_pollutant_exposure.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()


hourly_exposure = df.groupby("hour")["multi_pollutant_exposure"].mean()

plt.figure(figsize=(10, 6))
hourly_exposure.plot(marker="o")
plt.title("Average Multi-Pollutant Exposure by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Exposure Score")
plt.grid()
plt.savefig(
    PLOTS_DIR / "hourly_exposure_score.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()


plt.figure(figsize=(12, 6))

df["multi_pollutant_exposure"].plot(
    label="Exposure Score",
    alpha=0.8
)

critical_events = df[df["alert"].str.startswith("CRITICAL")]
high_events = df[df["alert"].str.startswith("HIGH")]
moderate_events = df[df["alert"].str.startswith("MODERATE")]

plt.scatter(
    moderate_events.index,
    moderate_events["multi_pollutant_exposure"],
    label="Moderate Alert",
    alpha=0.4
)

plt.scatter(
    high_events.index,
    high_events["multi_pollutant_exposure"],
    label="High Alert",
    alpha=0.6
)

plt.scatter(
    critical_events.index,
    critical_events["multi_pollutant_exposure"],
    label="Critical Alert",
    alpha=0.8
)

plt.axhline(
    y=adjusted_high_threshold,
    linestyle="--",
    label="Adjusted High Risk Threshold"
)

plt.title("Exposure Score with Intelligent Alerts")
plt.xlabel("Time")
plt.ylabel("Exposure Score")
plt.legend()
plt.grid()

plt.savefig(
    PLOTS_DIR / "exposure_with_intelligent_alerts.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()


# =========================
# SAVE FINAL DATA
# =========================
df.to_csv(PROCESSED_DATA_PATH)


# =========================
# SUMMARY OUTPUT
# =========================
print("\nExposure model completed successfully.")
print("User profile:", USER_PROFILE)

print("\nThresholds:")
print(f"Low threshold: {adjusted_low_threshold:.3f}")
print(f"Medium threshold: {adjusted_medium_threshold:.3f}")
print(f"High threshold: {adjusted_high_threshold:.3f}")
print(f"Spike threshold: {spike_threshold:.3f}")

print("\nRisk level counts:")
print(df["risk_level"].value_counts())

print("\nAlert counts:")
print(df["alert"].value_counts())

print("\nDominant pollutant counts:")
print(df["dominant_pollutant"].value_counts())

print("\nExample intelligent system output:")
print(
    df[
        [
            "multi_pollutant_exposure",
            "risk_level",
            "dominant_pollutant",
            "alert",
            "recommendation"
        ]
    ].head(10)
)

print("\nSaved plots in:", PLOTS_DIR)
print("Saved processed data as:", PROCESSED_DATA_PATH)
