"""Shared helpers used by the notebooks: logging, raw-data loading, feature
building, train/test splitting, metrics and table export.

Keeping these in one file means every notebook does things the same way,
e.g. Phase 2 and Phase 9 build features with the exact same function.
"""
import logging
import platform
import sys
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import config as C


# ====================================================================== logging
class _AppendAndCloseHandler(logging.Handler):
    """Opens the log file, appends one line and closes it again for every message.

    A normal FileHandler keeps the file open, and on Colab's Drive mount an open
    file is not uploaded until it is closed, so the log never reached Drive.
    """
    def __init__(self, path):
        super().__init__()
        self.path = path

    def emit(self, record):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def get_logger(phase_tag: str) -> logging.Logger:
    """Logger that prints to the notebook AND appends to results/logs/<phase_tag>.log."""
    logger = logging.getLogger(phase_tag)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = _AppendAndCloseHandler(C.LOGS_DIR / f"{phase_tag}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("=" * 70)
    logger.info(f"RUN STARTED {datetime.now():%Y-%m-%d %H:%M:%S} | {phase_tag}")
    logger.info(f"python {platform.python_version()} | pandas {pd.__version__} | "
                f"numpy {np.__version__} | machine: {platform.platform()}")
    logger.info(f"project root: {C.PROJECT_ROOT}")
    return logger


def log_df(logger, df: pd.DataFrame, title: str = None):
    """Write a whole DataFrame into the log (and print it)."""
    if title:
        logger.info(f"--- {title} ---")
    for line in df.to_string().splitlines():
        logger.info(line)


class Timer:
    """Usage:  with Timer() as t: ...   then t.seconds gives the elapsed time."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start


# ====================================================================== raw data
def open_raw_source():
    """Return (reader, vm_ids).

    Reads from the extracted folder if it is present, otherwise straight from
    the zip (much faster on Colab because the zip is a single 129 MB file).
    """
    if C.RAW_ZIP.exists():
        zf = zipfile.ZipFile(C.RAW_ZIP)
        members = {n.rsplit("/", 1)[-1][:-4]: n for n in zf.namelist() if n.endswith(".csv")}
        reader = lambda vm_id: zf.open(members[vm_id])
    elif C.RAW_VM_DIR.is_dir():
        members = {p.stem: p for p in C.RAW_VM_DIR.glob("*.csv")}
        reader = lambda vm_id: open(members[vm_id], "rb")
    else:
        raise FileNotFoundError(f"No raw data found at {C.RAW_ZIP} or {C.RAW_VM_DIR}")
    vm_ids = sorted(members, key=int)
    return reader, vm_ids


def read_vm(reader, vm_id: str) -> pd.DataFrame:
    """Read one VM file: skip the header line, apply our 11 column names,
    force every column to numeric (anything unparseable becomes NaN)."""
    with reader(vm_id) as f:
        df = pd.read_csv(f, sep=C.RAW_SEP, skiprows=C.RAW_SKIP_HEADER_ROWS,
                         header=None, names=C.RAW_COLUMNS, dtype=str)
    for col in C.RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
    df.insert(0, "vm_id", int(vm_id))
    return df


# ====================================================================== features
FEATURE_COLUMNS = (
    ["cpu_usage_pct", "mem_usage_pct", "cpu_cores"]
    + [f"cpu_lag_{k}" for k in C.LAGS]
    + [f"mem_lag_{k}" for k in C.LAGS]
    + [f"cpu_roll_mean_{C.ROLL_WINDOW}", f"cpu_roll_std_{C.ROLL_WINDOW}"]
    + C.IO_COLUMNS
    + ["hour", "minute"]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features for each VM separately (no information crosses VMs).

    At time t we know the readings at t, t-1, t-2, ... and we predict CPU% at t+1.
    - cpu/mem lags: values at t-1, t-2, t-3
    - rolling mean/std of CPU% over the 6 readings t-5..t (only past + current,
      never the future, so there is no leakage into the target)
    - hour / minute of reading t
    - target: CPU% at t+1 (rows without a next reading are dropped)
    """
    df = df.sort_values(["vm_id", "timestamp"]).copy()
    df["mem_usage_pct"] = 100.0 * df["memory_usage_kb"] / df["memory_provisioned_kb"]
    g = df.groupby("vm_id", sort=False)
    for k in C.LAGS:
        df[f"cpu_lag_{k}"] = g["cpu_usage_pct"].shift(k)
        df[f"mem_lag_{k}"] = g["mem_usage_pct"].shift(k)
    roll = g["cpu_usage_pct"].rolling(C.ROLL_WINDOW, min_periods=C.ROLL_WINDOW)
    df[f"cpu_roll_mean_{C.ROLL_WINDOW}"] = roll.mean().reset_index(level=0, drop=True)
    df[f"cpu_roll_std_{C.ROLL_WINDOW}"] = roll.std().reset_index(level=0, drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df[C.TARGET] = g["cpu_usage_pct"].shift(-1)
    df = df.dropna(subset=FEATURE_COLUMNS + [C.TARGET])
    return df.reset_index(drop=True)


def test_mask(df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION) -> np.ndarray:
    """True for rows in the last 20% of each VM's (time-sorted) rows."""
    pos = df.groupby("vm_id").cumcount()
    n = df.groupby("vm_id")["vm_id"].transform("size")
    return (pos >= np.floor(n * (1 - test_fraction))).to_numpy()


def chrono_split(df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION):
    """Per-VM chronological split: first 80% of each VM's rows -> train, last 20% -> test."""
    is_test = test_mask(df, test_fraction)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


# ====================================================================== metrics
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mask = np.abs(y_true) >= C.MAPE_MIN_ACTUAL
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err[mask]) / np.abs(y_true[mask])) * 100),
        "R2": float(1 - ss_res / ss_tot),
    }


# ====================================================================== TOPSIS
def topsis(decision: pd.DataFrame, weights: dict, benefit: dict) -> dict:
    """TOPSIS (Hwang & Yoon, 1981) with vector normalisation.

    decision: rows = alternatives (models), columns = criteria
    weights:  {criterion: weight}, normalised to sum to 1 here
    benefit:  {criterion: True if higher is better, False if lower is better}
    Returns every intermediate step so notebooks can show and save them.
    """
    cols = list(weights)
    X = decision[cols].astype(float)
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    norm = X / np.sqrt((X ** 2).sum(axis=0))                      # step 1: vector normalisation
    weighted = norm * w                                            # step 2: apply weights
    is_benefit = pd.Series({c: bool(benefit[c]) for c in cols})
    ideal_best = pd.Series({c: weighted[c].max() if is_benefit[c] else weighted[c].min() for c in cols})
    ideal_worst = pd.Series({c: weighted[c].min() if is_benefit[c] else weighted[c].max() for c in cols})
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))  # step 4: distances
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = d_worst / (d_best + d_worst)                       # step 5: closeness coefficient
    rank = closeness.rank(ascending=False, method="min").astype(int)
    return {"normalized": norm, "weights": w, "weighted": weighted, "ideal_best": ideal_best,
            "ideal_worst": ideal_worst, "d_best": d_best, "d_worst": d_worst,
            "closeness": closeness, "rank": rank}


# ====================================================================== table export
def save_table(df: pd.DataFrame, name: str, title: str = None, float_fmt: str = "{:.4f}",
               index: bool = False):
    """Save a results table as CSV and as a styled PNG
    (coloured header row, alternating row shading)."""
    import matplotlib.pyplot as plt

    df.to_csv(C.TABLES_DIR / f"{name}.csv", index=index)
    shown = df.reset_index() if index else df.copy()
    cells = [[float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in shown.itertuples(index=False)]
    ncols, nrows = len(shown.columns), len(cells)
    fig_w = max(6, min(22, 1.25 * ncols + 1))
    fig_h = 0.42 * (nrows + 1) + (0.5 if title else 0.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=[str(c) for c in shown.columns],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c8ccd2")
        if r == 0:
            cell.set_facecolor("#2f5d8a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef2f7")
        else:
            cell.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=11, weight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(C.TABLES_DIR / f"{name}.png", dpi=C.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
