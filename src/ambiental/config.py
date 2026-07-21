from pathlib import Path


# =========================
# PROJECT PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

RAW_CSV_PATH = RAW_DATA_DIR / "AirQualityUCI.csv"
RAW_XLSX_PATH = RAW_DATA_DIR / "AirQualityUCI.xlsx"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_exposure_data.csv"


def ensure_project_directories() -> None:
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PLOTS_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

