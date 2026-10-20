"""Single-column version of the actual-vs-predicted figure for the paper.

Same data as Phase 10 (official Colab predictions saved in data/processed); only the
layout changes: 3.5 in wide, larger text, first 24 h of the test period of the same VM.
Run: python scripts/make_fig_actual_vs_predicted_col.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import model_definitions as M  # noqa: E402
import plot_style as P  # noqa: E402

P.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

preds = pd.read_parquet(C.DATA_PROCESSED / "test_predictions.parquet")
for m in P.ORDER:
    preds[f"tuned_{M.slug(m)}"] = np.load(C.MODELS_DIR / f"pred_test_tuned_{M.slug(m)}.npy")
vm = preds.groupby("vm_id")["actual"].std().idxmax()          # same VM selection rule as Phase 10
p = preds[preds.vm_id == vm].sort_values("timestamp")
p = p[p["timestamp"] < p["timestamp"].iloc[0] + pd.Timedelta(hours=24)]

ranking = pd.read_csv(C.TABLES_DIR / "phase7_topsis_ranking.csv")
winner = ranking.sort_values("Rank").Model.iloc[0]
show = [winner] + [m for m in ["LSTM", "Linear Regression"] if m != winner]

fig, ax = plt.subplots(figsize=(P.ONE_COL, 2.2))
ax.plot(p["timestamp"], p["actual"], color=P.INK, lw=1.3, label="Actual")
for m in show:
    ax.plot(p["timestamp"], p[f"tuned_{M.slug(m)}"], color=P.COLOR[m], lw=1.0, label=P.SHORT[m])
ax.set_ylabel("CPU usage (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.set_xlabel(f"Time of day, {p['timestamp'].iloc[0]:%d %b %Y} (VM {vm})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(show) + 1, frameon=False)
fig.tight_layout()
out = C.CHARTS_DIR / "fig_actual_vs_predicted_col"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=C.CHART_DPI, bbox_inches="tight")
dst = C.PROJECT_ROOT / "paper" / "final" / "figures" / "fig_actual_vs_predicted_col.pdf"
fig.savefig(dst, bbox_inches="tight")
print("VM", vm, "models", show, "->", dst)
