"""Chart style shared by the figure-producing notebooks (same look as Phase 10).

Each model keeps one colour everywhere (palette validated for colour-blind readers);
models are always also named on an axis or by a direct label.
"""
import matplotlib.pyplot as plt
import numpy as np

import config as C

ORDER = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "LSTM"]
SHORT = {"Linear Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF",
         "XGBoost": "XGB", "LightGBM": "LGBM", "LSTM": "LSTM"}
COLOR = {"Linear Regression": "#2a78d6", "Decision Tree": "#eb6834", "Random Forest": "#1baf7a",
         "XGBoost": "#eda100", "LightGBM": "#e87ba4", "LSTM": "#008300"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ONE_COL, TWO_COL = 3.5, 7.16


def apply():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    })


def save_fig(fig, name, log=None, show=True):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.CHARTS_DIR / f"{name}.{ext}", dpi=C.CHART_DPI, bbox_inches="tight")
    if log:
        log.info(f"Saved figure: {name}.png / .pdf")
    if show:
        plt.show()
    plt.close(fig)


def end_labels(ax, x_end, y_values, min_gap, y_top=1.0, names=None):
    """Direct labels at the right end of lines, spaced apart and kept below y_top."""
    names = names or SHORT
    items = sorted(y_values.items(), key=lambda kv: kv[1])
    lab = [y for _, y in items]
    for i in range(1, len(lab)):
        lab[i] = max(lab[i], lab[i - 1] + min_gap)
    if lab and lab[-1] > y_top:
        lab[-1] = y_top
        for i in range(len(lab) - 2, -1, -1):
            lab[i] = min(lab[i], lab[i + 1] - min_gap)
    for (m, y), y_lab in zip(items, lab):
        ax.annotate(names.get(m, m), (x_end, y), xytext=(x_end, y_lab), textcoords="data",
                    va="center", fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5) if abs(y_lab - y) > 0.005 else None)
