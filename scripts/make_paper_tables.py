"""Build the LaTeX tables and figure folder for the paper directly from the official results.

Run:  python scripts/make_paper_tables.py
Writes paper/final/tables/*.tex and copies figures into paper/final/figures/.
No number in the paper's tables is typed by hand; everything comes from results/ and the
saved JSON result files, so the paper always matches the Colab runs.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

T = C.TABLES_DIR
OUT = C.PROJECT_ROOT / "paper" / "final"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "LSTM"]
SHORT = {"Linear Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF",
         "XGBoost": "XGB", "LightGBM": "LGBM", "LSTM": "LSTM"}


def short(name: str) -> str:
    base = name.replace(" (VM-agnostic)", "")
    return SHORT.get(base, base) + ("-A" if "agnostic" in name else "")


def f(x, d=3):
    return f"{x:.{d}f}"


def bold_best(values, fmt, higher=False):
    """Format a list of numbers, bolding the best one."""
    best = max(values) if higher else min(values)
    return [("\\textbf{" + fmt(v) + "}") if v == best else fmt(v) for v in values]


def esc(text: str) -> str:
    """Escape LaTeX special characters in plain-text cells."""
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        text = text.replace(a, b)
    return text.replace("<", "$<$").replace(">", "$>$")


def write(name, caption, label, colspec, header, rows, note=None, wide=False, size="\\footnotesize", pos="!htb"):
    """Table in the style of the team's previous paper: caption above, booktabs rules,
    centred small note below. The tabular is wrapped in an adjustbox that only ever
    shrinks it, so a table can never run into the neighbouring column."""
    env = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    lines = [f"\\begin{{{env}}}[{pos}]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{adjustbox}}{{max width={width}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines += ["\\par\\vspace{3pt}", f"{{\\scriptsize {note}\\par}}"]
    lines += [f"\\end{{{env}}}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------- Table: cleaning steps
d = pd.read_csv(T / "phase1_cleaning_steps.csv")
rows = []
for _, r in d.iterrows():
    note = "" if pd.isna(r["Note"]) else esc(str(r["Note"]).split(" [")[0].split(" (")[0])
    rows.append([int(r["Step"]), esc(str(r["Rule"])), f"{int(r['Rows removed']):,}", int(r["VMs after"]), note])
write("tab_cleaning", "Data Cleaning Steps (300 Candidate VMs)",
      "tab:cleaning", "cp{3.0cm}rrp{1.55cm}", ["\\#", "Rule", "Rows removed", "VMs", "Note"], rows)

# ---------------------------------------------------------------- Table: hyperparameter grids
p4 = json.load(open(C.MODELS_DIR / "phase4_results.json"))
grids = {
    "DT": ("max\\_depth $\\in\\{5,10,15,\\infty\\}$, min\\_samples\\_split $\\in\\{2,10,50\\}$", 12),
    "RF": ("(n\\_est.\\,50, depth $\\{10,20,\\infty\\}$); (100, $\\{10,20\\}$)", 5),
    "XGB": ("n\\_est. $\\{100,300\\}$, depth $\\{4,6,8\\}$, lr $\\{0.05,0.1\\}$", 12),
    "LGBM": ("n\\_est. $\\{100,300\\}$, depth $\\{4,8,\\infty\\}$, lr $\\{0.05,0.1\\}$", 12),
    "LSTM": ("hidden $\\{32,64\\}$, history $\\{6,12\\}$ steps", 4),
}
best_fmt = {
    "DT": lambda b: f"depth {b['max_depth']}, split {b['min_samples_split']}",
    "RF": lambda b: f"{b['n_estimators']} trees, depth {b['max_depth']}",
    "XGB": lambda b: f"{b['n_estimators']}, depth {b['max_depth']}, lr {b['learning_rate']}",
    "LGBM": lambda b: f"{b['n_estimators']}, depth {'$\\infty$' if b['max_depth'] == -1 else b['max_depth']}, lr {b['learning_rate']}",
    "LSTM": lambda b: f"hidden {b['hidden_size']}, history {b['seq_len']}",
}
rows = []
for name in ORDER[1:]:
    s = SHORT[name]
    rows.append([s, grids[s][0], grids[s][1], best_fmt[s](p4[name]["best_params"]), f(p4[name]["tuning_time_sec"] / 60, 1)])
write("tab_grids", "Hyperparameter Search Spaces and Selected Settings",
      "tab:grids", "lp{3.6cm}cp{2.3cm}r", ["Model", "Search space", "\\#", "Selected", "Min"], rows,
      note="Tree models: GridSearchCV with time-ordered TimeSeriesSplit (3 folds), RMSE criterion. "
           "LSTM: validation-based search with early stopping. Linear Regression has no hyperparameters.")

# ---------------------------------------------------------------- Table: main results (tuned)
full = pd.read_csv(T / "table_full_results.csv").set_index("Model").loc[ORDER]
p3 = pd.read_csv(T / "phase3_results_raw.csv").set_index("Model")
pers = p3.loc["Persistence (reference)"]
cols = {
    "MAE": (lambda v: f(v, 3), False), "RMSE": (lambda v: f(v, 3), False), "MAPE (%)": (lambda v: f(v, 2), False),
    "R2": (lambda v: f(v, 4), True), "Train time (s)": (lambda v: f"{v:.3g}", False),
    "Inference (ms/1k)": (lambda v: f"{v:.3g}", False), "Energy (Wh)": (lambda v: f"{v:.2g}", False),
    "CO2 (g)": (lambda v: f"{v:.2g}", False), "TOPSIS C": (lambda v: f(v, 3), True),
}
colvals = {c: bold_best(full[c].tolist(), fmt, higher) for c, (fmt, higher) in cols.items()}
rows = [[SHORT[m]] + [colvals[c][i] for c in cols] + [int(full.loc[m, "TOPSIS rank"])] for i, m in enumerate(ORDER)]
rows.append(["\\textit{Persist.}", f(pers["MAE"]), f(pers["RMSE"]), f(pers["MAPE"], 2), f(pers["R2"], 4),
             "0", "--", "0", "0", "--", "--"])
write("tab_results", "Accuracy, Cost, Energy and TOPSIS Ranking of the Tuned Models (Test Set)",
      "tab:results", "lrrrrrrrrrr",
      ["Model", "MAE", "RMSE", "MAPE\\%", "R$^2$", "Train s", "Inf.\\,ms/1k", "Wh", "g\\,CO$_2$", "$C_i$", "Rank"], rows,
      note="Best value per column in bold. Energy and CO$_2$ are per training run (CodeCarbon; USA/Iowa grid, 0.369 kg/kWh). "
           "Inference time per 1{,}000 predictions. Persist.: persistence reference ($\\hat{y}_{t+1}=y_t$), not ranked.",
      wide=True, pos="!b")  # bottom of page (stfloats): keeps this full-width table inside the Results section

# ---------------------------------------------------------------- Table: tuning cost
tb = pd.read_csv(T / "phase4_tuning_before_after.csv").set_index("Model").loc[ORDER]
te = pd.read_csv(T / "phase5_tuning_energy_estimate.csv").set_index("Model").loc[ORDER]
rows = []
for m in ORDER:
    r = tb.loc[m]
    dm = r["Δ MAE %"]
    rows.append([SHORT[m], f(r["Default MAE"]), f(r["Tuned MAE"]), f"{dm:+.1f}", f(r["Default RMSE"]), f(r["Tuned RMSE"]),
                 f(r["Tuning time (min)"], 1), f"{te.loc[m, 'Est. tuning energy (kWh)'] * 1000:.2g}" if m != "Linear Regression" else "--"])
write("tab_tuning", "Effect and Cost of Hyperparameter Tuning (Test Set)", "tab:tuning", "lrrrrrrr",
      ["Model", "MAE$_{def}$", "MAE$_{tun}$", "$\\Delta$\\%", "RMSE$_{def}$", "RMSE$_{tun}$", "Min", "Wh"], rows,
      note="Wh: estimated tuning energy = tuning time $\\times$ measured mean power of that model.")

# ---------------------------------------------------------------- Table: stability
st = pd.read_csv(T / "phase6_stability_summary_raw.csv").set_index("model").loc[ORDER]
rows = [[SHORT[m], f"{st.loc[m, 'MAE_mean']:.3f} $\\pm$ {st.loc[m, 'MAE_std']:.3f}",
         f"{st.loc[m, 'RMSE_mean']:.3f} $\\pm$ {st.loc[m, 'RMSE_std']:.3f}",
         f"{st.loc[m, 'R2_mean']:.4f} $\\pm$ {st.loc[m, 'R2_std']:.4f}", f"{st.loc[m, 'MAE_cv_pct']:.2f}"] for m in ORDER]
write("tab_stability", "Stability Over Five Random Seeds (Mean $\\pm$ Std.)", "tab:stability",
      "lcccr", ["Model", "MAE", "RMSE", "R$^2$", "CV\\%"], rows,
      note="Seeds 42, 7, 123, 2024, 99. CV: coefficient of variation of MAE.")

# ---------------------------------------------------------------- Table: sensitivity ranks
se = pd.read_csv(T / "phase8_sensitivity_raw.csv").set_index("Model").loc[ORDER]
scen = ["Primary", "Balanced", "Accuracy-Priority", "Energy-Priority", "Speed-Priority"]
rows = [[SHORT[m]] + [f"{se.loc[m, s + ' C']:.3f} ({int(se.loc[m, s + ' rank'])})" for s in scen] for m in ORDER]
write("tab_sensitivity", "TOPSIS Closeness (Rank) Under Five Weighting Scenarios", "tab:sensitivity", "lccccc",
      ["Model", "Primary", "Balanced", "Accuracy", "Energy", "Speed"], rows)

# ---------------------------------------------------------------- Table: generalization (Phase 9)
g = pd.read_csv(T / "phase9_generalization_tuned.csv").set_index("Model").loc[ORDER]
pv = pd.read_csv(T / "phase9_per_vm_spread.csv")
pv = pv[pv.Version == "tuned"].set_index("Model").loc[ORDER]
rows = [[SHORT[m], f(g.loc[m, "Test MAE"]), f(g.loc[m, "Unseen MAE"]), f"{g.loc[m, 'MAE change (%)']:+.1f}",
         f(g.loc[m, "Unseen RMSE"]), f(g.loc[m, "Unseen R2"], 4), f(pv.loc[m, "Median VM MAE"]),
         f(pv.loc[m, "Worst VM MAE"], 1)] for m in ORDER]
gen_raw = pd.read_csv(T / "phase9_generalization_raw.csv")
rows.append(["\\textit{Persist.}", f(pers["MAE"]), "0.506", "--", "3.315", "0.9620", "--", "--"])
write("tab_generalization", "Generalization of the Tuned Models to 100 Unseen VMs",
      "tab:generalization", "lrrrrrrr",
      ["Model", "MAE$_{test}$", "MAE$_{unseen}$", "$\\Delta$\\%", "RMSE", "R$^2$", "Med. VM", "Worst VM"], rows,
      note="Med./Worst VM: median and maximum per-VM MAE over the 100 unseen VMs.")

# ---------------------------------------------------------------- Table: VM-agnostic ablation (Phase 11)
ab = pd.read_csv(T / "table_ablation_vm_agnostic.csv").set_index("Model")
order11 = []
for m in ORDER[:-1]:
    order11 += [m, f"{m} (VM-agnostic)"]
order11.append("LSTM")
rows = [[short(m), int(ab.loc[m, "Features"]), f(ab.loc[m, "Test MAE"]), f(ab.loc[m, "Unseen MAE"]),
         f(ab.loc[m, "Unseen R2"], 4), f(ab.loc[m, "Worst-VM MAE"], 2), f(ab.loc[m, "VM387 MAE"], 2)] for m in order11]
write("tab_ablation", "Full vs. VM-Agnostic Feature Sets",
      "tab:ablation", "lrrrrrr", ["Model", "Feat.", "MAE$_{test}$", "MAE$_{unseen}$", "R$^2_{unseen}$", "Worst VM", "VM 387"], rows,
      note="-A: VM-agnostic variant (no core count, disk or network features). All variants retrained in one session with "
           "the Phase-4 tuned hyperparameters.")

# ---------------------------------------------------------------- Table: robust TOPSIS
rt = pd.read_csv(T / "table_topsis_robust.csv")
rows = [[short(r.Model), f(r["Standard C"]), int(r["Standard rank"]), f(r["Robust C"]), int(r["Robust rank"])]
        for _, r in rt.iterrows()]
write("tab_robust", "Standard vs. Robustness-Aware TOPSIS (11 Variants)", "tab:robust", "lrrrr",
      ["Variant", "$C_i$ std.", "Rank", "$C_i$ robust", "Rank"], rows,
      note="Robust weights: MAE .10, RMSE .10, MAPE .10, R$^2$ .15, unseen MAE .20, training time .15, CO$_2$ .20.")

# ---------------------------------------------------------------- Table: fleet projection
fl = pd.read_csv(T / "table_fleet_projection.csv").set_index("Model").loc[order11]
rows = [[short(m), f"{fl.loc[m, 'Training (kWh/yr)']:.3g}", f"{fl.loc[m, 'Inference (kWh/yr)']:.2g}",
         f"{fl.loc[m, 'Total CO2 (kg/yr)']:.3g}", f"{fl.loc[m, 'vs lowest (x)']:.0f}$\\times$"] for m in order11]
write("tab_fleet", "Projected Annual Energy and CO$_2$ for a 10{,}000-VM Fleet", "tab:fleet", "lrrrr",
      ["Variant", "Train kWh", "Infer. kWh", "kg CO$_2$", "vs. min"], rows,
      note="Assumes 200 models (one per 50 VMs) retrained daily and one forecast per VM every 5 min "
           "($1.05\\times10^9$ predictions/yr); grid 0.369 kg/kWh; excludes data handling and facility overhead (PUE).")

# ---------------------------------------------------------------- figures
figs = ["phase2_example_split.png", "fig_prediction_metrics_combined.pdf", "fig_tradeoff_accuracy_vs_energy.pdf",
        "fig_tradeoff_accuracy_vs_time.pdf", "fig_topsis_ranking.pdf", "fig_topsis_sensitivity.pdf",
        "fig_stability_errorbars.pdf", "fig_ablation_generalization.pdf", "fig_topsis_robust_sweep.pdf",
        "fig_fleet_projection.pdf", "fig_actual_vs_predicted.pdf", "fig_training_time.pdf", "fig_co2.pdf"]
for name in figs:
    shutil.copy2(C.CHARTS_DIR / name, FIG / name)
print("copied", len(figs), "figures")
