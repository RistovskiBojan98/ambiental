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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# CONFIGURATION
# =========================
ensure_project_directories()


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(PROCESSED_DATA_PATH)

df["Datetime"] = pd.to_datetime(df["Datetime"])

df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)


# =========================
# CREATE FUTURE TARGET
# =========================
df["future_exposure"] = (
    df["multi_pollutant_exposure"].shift(-1)
)


# =========================
# FEATURES
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
    "hour",
    "hour_weight",
    "exposure_trend"
]

target = "future_exposure"

df_model = df.dropna(
    subset=features + [target]
).copy()


X = df_model[features]
y = df_model[target]


# =========================
# TRAIN / TEST SPLIT
# =========================
split_index = int(len(df_model) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# =========================
# TRAIN LINEAR REGRESSION
# =========================
model = LinearRegression()

model.fit(X_train, y_train)


# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

df_model.loc[
    X_test.index,
    "predicted_future_exposure"
] = y_pred


# =========================
# EVALUATION
# =========================
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("\n=== LINEAR REGRESSION RESULTS ===")

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R^2:   {r2:.4f}")


# =========================
# COEFFICIENT IMPORTANCE
# =========================
coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_
})

coefficients["abs_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="abs_coefficient",
    ascending=False
)

print("\nFeature Coefficients:")
print(coefficients)


# =========================
# PLOT ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Future Exposure"
)

plt.plot(
    y_test.index,
    y_pred,
    label="Predicted Future Exposure",
    alpha=0.8
)

plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Future Exposure")
plt.legend()
plt.grid()

plt.savefig(
    PLOTS_DIR / "linear_regression_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# PLOT FEATURE COEFFICIENTS
# =========================
plt.figure(figsize=(10, 6))

plt.bar(
    coefficients["feature"],
    coefficients["coefficient"]
)

plt.title("Linear Regression Feature Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")

plt.savefig(
    PLOTS_DIR / "linear_regression_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# SAVE RESULTS
# =========================
linear_results_path = RESULTS_DIR / "linear_regression_results.csv"
df_model.to_csv(
    linear_results_path
)

print("\nSaved results:")
print(linear_results_path)

print("\nSaved plots:")
print("linear_regression_predictions.png")
print("linear_regression_coefficients.png")
