"""Column-width (3.5 in) versions of two paper figures, with text sized to stay readable
in a two-column IEEE layout. Same official data as Phases 2 and 8; only the layout changes.

Run: python scripts/make_col_figures.py
Writes results/charts/fig_split_col.{png,pdf}, fig_topsis_sensitivity_col.{png,pdf}
and copies the PDFs into paper/final/figures/.
"""
import shutil
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
import plot_style as P  # noqa: E402

P.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
FIG_DIR = C.PROJECT_ROOT / "paper" / "final" / "figures"


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.CHARTS_DIR / f"{name}.{ext}", dpi=C.CHART_DPI, bbox_inches="tight")
    shutil.copy2(C.CHARTS_DIR / f"{name}.pdf", FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


# ---- Fig. 1: chronological split of one VM (same VM rule as Phase 2: largest CPU std)
feat = pd.read_parquet(C.DATA_PROCESSED / "features_main.parquet", columns=["vm_id", "timestamp", "cpu_usage_pct", "split"])
vm = feat.groupby("vm_id")["cpu_usage_pct"].std().idxmax()
ex = feat[feat.vm_id == vm]
fig, ax = plt.subplots(figsize=(P.ONE_COL, 1.9))
for part, color, label in (("train", "#2f5d8a", "Training (first 80%)"), ("test", "#d1793b", "Test (last 20%)")):
    d = ex[ex.split == part]
    ax.plot(d["timestamp"], d["cpu_usage_pct"], lw=0.5, color=color, label=label)
ax.axvline(ex.loc[ex.split == "test", "timestamp"].min(), color="black", ls="--", lw=0.9)
ax.set_ylabel("CPU usage (%)")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_xlabel(f"Date (2013), VM {vm}")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save(fig, "fig_split_col")

# ---- Fig. 5b: TOPSIS closeness under the five scenarios
sens = pd.read_csv(C.TABLES_DIR / "phase8_sensitivity_raw.csv").set_index("Model").loc[P.ORDER]
scen = [c[:-2] for c in sens.columns if c.endswith(" C")]
short_scen = {"Primary": "Primary", "Balanced": "Balanced", "Accuracy-Priority": "Accuracy",
              "Energy-Priority": "Energy", "Speed-Priority": "Speed"}
fig, ax = plt.subplots(figsize=(P.ONE_COL, 2.3))
x = np.arange(len(scen))
for m in P.ORDER:
    ax.plot(x, [sens.loc[m, f"{s} C"] for s in scen], marker="o", ms=3.5, color=P.COLOR[m], lw=1.4, mec="white", mew=0.8)
P.end_labels(ax, x[-1] + 0.2, {m: sens.loc[m, f"{scen[-1]} C"] for m in P.ORDER}, min_gap=0.075)
ax.set_xticks(x, [short_scen[s] for s in scen])
ax.set_xlim(-0.2, len(scen) - 0.25)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Weighting scenario (priority)")
ax.set_ylabel("Closeness coefficient $C$")
save(fig, "fig_topsis_sensitivity_col")
