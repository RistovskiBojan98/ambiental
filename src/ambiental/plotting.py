import os
import sys

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ambiental.config import (
    PLOTS_DIR,
    PROJECT_ROOT,
    RAW_CSV_PATH,
    ensure_project_directories,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_plotting_data():
    df = pd.read_csv(RAW_CSV_PATH, sep=";")

    # Remove empty columns and convert comma decimals to numeric values.
    df = df.iloc[:, :-2]
    for col in df.columns[2:]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace(-200, np.nan, inplace=True)

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    df = df.dropna(subset=["Datetime"])
    df.drop(["Date", "Time"], axis=1, inplace=True)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    df.interpolate(method="time", inplace=True)

    return df


def main():
    ensure_project_directories()
    df = load_plotting_data()
    pollutants = ["CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)"]

    # =========================
    # 1. CO LEVELS OVER TIME
    # =========================
    plt.figure()
    df["CO(GT)"].plot()
    plt.title("CO Levels Over Time")
    plt.xlabel("Time")
    plt.ylabel("CO (mg/m^3)")
    plt.grid()
    plt.savefig(PLOTS_DIR / "co_levels.png")
    plt.close()

    # =========================
    # 2. MULTIPLE POLLUTANTS
    # =========================
    plt.figure()
    df[pollutants].plot()
    plt.title("Air Pollutants Over Time")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid()
    plt.savefig(PLOTS_DIR / "multiple_pollutants.png")
    plt.close()

    # =========================
    # 3. DAILY PATTERN (HOURLY)
    # =========================
    df["hour"] = df.index.hour
    hourly_avg = df.groupby("hour")[pollutants].mean()

    plt.figure()
    hourly_avg.plot(marker="o")
    plt.title("Average Pollution Levels by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average Concentration")
    plt.grid()
    plt.savefig(PLOTS_DIR / "hourly_pattern.png")
    plt.close()

    # =========================
    # 4. TEMPERATURE & HUMIDITY
    # =========================
    env_vars = ["T", "RH", "AH"]

    plt.figure()
    df[env_vars].plot()
    plt.title("Environmental Variables Over Time")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.savefig(PLOTS_DIR / "environmental_variables.png")
    plt.close()

    # =========================
    # 5. CORRELATION HEATMAP
    # =========================
    correlation = df.corr()

    fig, axis = plt.subplots(figsize=(10, 8))
    heatmap = axis.imshow(correlation, cmap="coolwarm", aspect="auto")
    axis.set_title("Correlation Between Variables")
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90)
    axis.set_yticklabels(correlation.columns)
    fig.colorbar(heatmap, ax=axis)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    # =========================
    # 6. DISTRIBUTION OF POLLUTANTS
    # =========================
    plt.figure()
    df[pollutants].hist(bins=30)
    plt.suptitle("Distribution of Pollutants")
    plt.savefig(PLOTS_DIR / "distributions.png")
    plt.close()

    # =========================
    # 7. WEEKLY TREND (SMOOTHED)
    # =========================
    weekly_avg = df[pollutants].resample("D").mean()

    plt.figure()
    weekly_avg.plot()
    plt.title("Daily Average Pollution Trend")
    plt.xlabel("Date")
    plt.ylabel("Concentration")
    plt.grid()
    plt.savefig(PLOTS_DIR / "daily_trend.png")
    plt.close()

    print("Saved exploratory plots in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
