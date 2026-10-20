"""Central configuration for the Green ML project.

Every notebook imports this module, so paths, column names, seeds and
experiment settings are defined once here.
"""
import os
from pathlib import Path


def _find_project_root() -> Path:
    """Locate Green_ML_Project on this machine (local Drive mount or Colab)."""
    candidates = [
        os.environ.get("GREEN_ML_ROOT"),
        r"H:\My Drive\Green_ML_Project",                     # Windows, Google Drive Desktop
        "/content/drive/MyDrive/Green_ML_Project",           # Google Colab
        str(Path.home() / "Google Drive" / "My Drive" / "Green_ML_Project"),
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c)
    # Fallback: this file lives in <root>/scripts/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------- paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_VM_DIR = DATA_RAW / "fastStorage" / "2013-8"
RAW_ZIP = DATA_RAW / "bitbrains_fastStorage.zip"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = DATA_PROCESSED / "models"
RESULTS = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS / "tables"
CHARTS_DIR = RESULTS / "charts"
LOGS_DIR = RESULTS / "logs"
SUMMARY_MD = PROJECT_ROOT / "green_project_details.md"

for _d in (DATA_PROCESSED, MODELS_DIR, TABLES_DIR, CHARTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- raw format
# Verified 2026-10-04 on all 1,250 files: one header row, fields separated by
# ";\t", 11 fields per row, timestamps in SECONDS (header label "[ms]" is wrong).
RAW_SEP = ";"
RAW_SKIP_HEADER_ROWS = 1
TIMESTAMP_UNIT = "s"
RAW_COLUMNS = [
    "timestamp_ms", "cpu_cores", "cpu_capacity_provisioned_mhz", "cpu_usage_mhz",
    "cpu_usage_pct", "memory_provisioned_kb", "memory_usage_kb", "disk_read_kbps",
    "disk_write_kbps", "net_received_kbps", "net_transmitted_kbps",
]
IO_COLUMNS = ["disk_read_kbps", "disk_write_kbps", "net_received_kbps", "net_transmitted_kbps"]

# ---------------------------------------------------------------- sampling
SAMPLE_SEED = 42
# Decided 2026-10-04 after the Colab feasibility check (Phase 1b): 50 main VMs keeps
# Phases 3-6 within free-Colab limits; the generalization group is prediction-only,
# so it uses all 100 cleaned VMs at almost no cost.
N_TRAIN_VMS = 50          # primary experiment (train/tune/test)
N_GEN_VMS = 100           # generalization hold-out, never used for training/tuning
# Phase 1 cleans a larger pool (up to 100 + 100) so the sample can be scaled up
# later without redoing Phase 1. Phases 2+ use only the first N_TRAIN_VMS /
# N_GEN_VMS VMs of each pool (by pool_rank).
MAX_TRAIN_VMS = 100
MAX_GEN_VMS = 100
CANDIDATE_BUFFER = 50     # extra candidates loaded in case some VMs fail the cleaning rules
MIN_VALID_READINGS = 100
# Team decisions 2026-10-04:
# Q1 - rows with memory_provisioned = 0 and cpu_cores = 0 are powered-off periods. They are
#      removed, and VMs powered off for more than this share of their readings are excluded.
MAX_POWERED_OFF_SHARE = 0.10
# Q2 - CPU% slightly above 100 (max ~108.5) is real over-commit; clip to 100 instead of deleting.
CPU_PCT_CLIP_MAX = 100.0
EXPECTED_INTERVAL_SEC = 300

# ---------------------------------------------------------------- features
TARGET_SOURCE = "cpu_usage_pct"
TARGET = "target_cpu_next"
LAGS = [1, 2, 3]
ROLL_WINDOW = 6
TEST_FRACTION = 0.20

# MAPE is undefined when actual CPU% = 0 (idle VMs are common in Bitbrains),
# so it is computed only over test points with actual CPU% >= this threshold.
MAPE_MIN_ACTUAL = 1.0

# ---------------------------------------------------------------- models / experiments
RANDOM_SEED = 42
STABILITY_SEEDS = [42, 7, 123, 2024, 99]
N_JOBS = -1
CV_SPLITS = 3

MODEL_NAMES = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "LSTM"]

# ---------------------------------------------------------------- TOPSIS
# True = higher is better (benefit criterion); False = lower is better (cost).
TOPSIS_CRITERIA = {
    "MAE": False, "RMSE": False, "MAPE": False, "R2": True,
    "train_time_sec": False, "co2_emissions_kg": False,
}
TOPSIS_PRIMARY_WEIGHTS = {
    "MAE": 0.15, "RMSE": 0.15, "MAPE": 0.15, "R2": 0.20,
    "train_time_sec": 0.15, "co2_emissions_kg": 0.20,
}
TOPSIS_SCENARIOS = {
    "Balanced":          {"MAE": 1/6, "RMSE": 1/6, "MAPE": 1/6, "R2": 1/6, "train_time_sec": 1/6, "co2_emissions_kg": 1/6},
    "Accuracy-Priority": {"MAE": 0.20, "RMSE": 0.20, "MAPE": 0.15, "R2": 0.25, "train_time_sec": 0.10, "co2_emissions_kg": 0.10},
    "Energy-Priority":   {"MAE": 0.10, "RMSE": 0.10, "MAPE": 0.10, "R2": 0.10, "train_time_sec": 0.15, "co2_emissions_kg": 0.45},
    "Speed-Priority":    {"MAE": 0.10, "RMSE": 0.10, "MAPE": 0.10, "R2": 0.10, "train_time_sec": 0.45, "co2_emissions_kg": 0.15},
}

# ---------------------------------------------------------------- plotting
CHART_DPI = 200
