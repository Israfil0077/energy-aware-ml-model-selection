"""Recompute every key number quoted in the paper text from the official result files
and compare with the value written in paper/final/main.tex.

Run: python scripts/verify_paper_numbers.py   (prints OK / MISMATCH per claim)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

T, MD = C.TABLES_DIR, C.MODELS_DIR
tex = (C.PROJECT_ROOT / "paper" / "final" / "main.tex").read_text(encoding="utf-8")
p3 = json.load(open(MD / "phase3_results.json"))
p4 = json.load(open(MD / "phase4_results.json"))
p5 = json.load(open(MD / "phase5_results.json"))
p11 = json.load(open(MD / "phase11_results.json"))
q = pd.read_csv(T / "phase1_data_quality_before.csv").set_index("Check")["Count"]
steps = pd.read_csv(T / "phase1_cleaning_steps.csv")
split = pd.read_csv(T / "phase2_split_summary.csv").set_index("Set")
stab = pd.read_csv(T / "phase6_stability_summary_raw.csv").set_index("model")
rank = pd.read_csv(T / "phase7_topsis_ranking.csv").set_index("Model")
sweep = pd.read_csv(T / "phase8_weight_sweep.csv")
gen = pd.read_csv(T / "phase9_generalization_tuned.csv").set_index("Model")
pv = pd.read_csv(T / "phase9_per_vm_spread.csv")
rob = pd.read_csv(T / "table_topsis_robust.csv").set_index("Model")
rsw = pd.read_csv(T / "phase11_robust_weight_sweep.csv")
sig = pd.read_csv(T / "table_significance_tests.csv")
fleet = pd.read_csv(T / "table_fleet_projection.csv").set_index("Model")
tun = pd.read_csv(T / "phase4_tuning_before_after.csv").set_index("Model")
tue = pd.read_csv(T / "phase5_tuning_energy_estimate.csv").set_index("Model")
raw5 = pd.read_csv(T / "phase5_energy_all_raw.csv")

E = lambda m, v="tuned": p5[f"{m} | {v}"]
claims = []
def claim(text_in_paper, computed):
    claims.append((text_in_paper, computed, text_in_paper in tex))

# ---- dataset
claim("2{,}663{,}335", int(q["Total rows"]))
claim("103{,}458", int(q["Duplicate rows (all columns)"]))
claim("13.8\\%", round(100 * q["Powered-off rows (cores = 0 and memory = 0)"] / q["Total rows"], 1))
claim("2.7\\%", round(100 * q["CPU% > 100"] / q["Total rows"], 1))
claim("(35 VMs)", steps.loc[steps.Step == 5, "Note"].iloc[0])
claim("32{,}131", steps.loc[steps.Step == 7, "Note"].iloc[0])
claim("321{,}115", int(split.loc["Training", "Rows"]))
claim("80{,}303", int(split.loc["Test", "Rows"]))
# ---- accuracy
claim("0.664", round(p4["LightGBM"]["MAE"], 3)); claim("0.941", round(p4["LightGBM"]["R2"], 3))
claim("MAE 0.924", round(p4["LSTM"]["MAE"], 3))
# ---- cost / energy (tuned, Phase 5)
claim("7.8\\,s", round(E("LightGBM")["train_time_sec"], 1)); claim("0.034\\,Wh", round(E("LightGBM")["energy_kwh"] * 1000, 3))
claim("195\\,s", round(E("Random Forest")["train_time_sec"])); claim("159\\,s", round(E("LSTM")["train_time_sec"]))
claim("28 and 22 times", (round(E("Random Forest")["energy_kwh"] / E("LightGBM")["energy_kwh"]), round(E("LSTM")["energy_kwh"] / E("LightGBM")["energy_kwh"])))
claim("612\\,s", round(p3["Random Forest"]["train_time_sec"])); claim("218\\,s", round(p3["LSTM"]["train_time_sec"]))
claim("about 260 and 90 times", (round(E("Random Forest", "default")["energy_kwh"] / E("LightGBM", "default")["energy_kwh"]),
                                 round(E("LSTM", "default")["energy_kwh"] / E("LightGBM", "default")["energy_kwh"])))
tuned5 = raw5[raw5.version == "tuned"]
claim("13.7--17.5\\,W", (round((tuned5.energy_kwh * 3.6e6 / tuned5.train_time_sec).min(), 1), round((tuned5.energy_kwh * 3.6e6 / tuned5.train_time_sec).max(), 1)))
claim("more than 800", round(E("Random Forest")["energy_kwh"] / E("Linear Regression")["energy_kwh"]))
# ---- tuning
claim("53.4 minutes", round(tun["Tuning time (min)"].sum(), 1))
claim("15.2\\,Wh", round(tue["Est. tuning energy (kWh)"].sum() * 1000, 1)); claim("5.6\\,g", round(tue["Est. tuning CO2 (g)"].sum(), 1))
claim("1.5\\% for LightGBM", round(tun.loc["LightGBM", "Δ MAE %"], 2)); claim("0.5\\% for XGBoost", round(tun.loc["XGBoost", "Δ MAE %"], 2))
claim("69\\% of the tuning time", round(100 * tun.loc["Random Forest", "Tuning time (min)"] / tun["Tuning time (min)"].sum()))
claim("RMSE improved by 17\\%", round(100 * (tun.loc["Decision Tree", "Tuned RMSE"] / tun.loc["Decision Tree", "Default RMSE"] - 1), 1))
claim("2.6 times faster", round(tun.loc["Random Forest", "Default train (s)"] / tun.loc["Random Forest", "Tuned train (s)"], 2))
# ---- stability
claim("$0.697\\pm0.003$", (round(stab.loc["Random Forest", "MAE_mean"], 3), round(stab.loc["Random Forest", "MAE_std"], 3)))
claim("$0.661\\pm0.003$", (round(stab.loc["LightGBM", "MAE_mean"], 3), round(stab.loc["LightGBM", "MAE_std"], 3)))
claim("$0.942\\pm0.061$", (round(stab.loc["LSTM", "MAE_mean"], 3), round(stab.loc["LSTM", "MAE_std"], 3)))
claim("(0.036 MAE)", round(stab.loc["Random Forest", "MAE_mean"] - stab.loc["LightGBM", "MAE_mean"], 4))
claim("about 15 times", round(stab.loc["LSTM", "MAE_cv_pct"] / stab.loc[["Random Forest", "LightGBM"], "MAE_cv_pct"].mean(), 1))
# ---- TOPSIS
claim("($C=0.968$)", round(rank.loc["XGBoost", "Closeness C"], 4)); claim("($0.965$)", round(rank.loc["LightGBM", "Closeness C"], 4))
ch = sweep[sweep.winner != sweep.winner.shift()][["cost_weight_share", "winner"]].values.tolist()
claim("less than 35\\%", ch); claim("between 35\\% and 75\\%", ch); claim("only above 80\\%", ch)
# ---- generalization
for m, txt in [("LSTM", "LSTM (MAE 0.696)"), ("Linear Regression", "Linear Regression (0.707)"), ("LightGBM", "LightGBM (1.083)"), ("Random Forest", "Random Forest (1.095)")]:
    claim(txt, round(gen.loc[m, "Unseen MAE"], 3))
gp = pd.read_parquet(C.DATA_PROCESSED / "generalization_predictions.parquet")
v = gp[gp.vm_id == 387]
claim("LightGBM MAE 28.7", round(np.abs(v.lightgbm_tuned - v.actual).mean(), 2)); claim("against 1.06 for the LSTM", round(np.abs(v.lstm_tuned - v.actual).mean(), 2))
# ---- ablation (Phase 11)
a, b = p11["train"]["LightGBM"], p11["train"]["LightGBM (VM-agnostic)"]
claim("MAE 0.972 to 0.681", (round(a["unseen"]["MAE"], 3), round(b["unseen"]["MAE"], 3)))
claim("falls by 30\\%", round(100 * (1 - b["unseen"]["MAE"] / a["unseen"]["MAE"]), 1))
claim("from 34.7 to 4.6", (round(a["unseen_per_vm"]["worst_vm_mae"], 2), round(b["unseen_per_vm"]["worst_vm_mae"], 2)))
claim("from 34.7 to 1.5", (round(a["unseen_per_vm"]["vm387_mae"], 2), round(b["unseen_per_vm"]["vm387_mae"], 2)))
for m, pct in [("Random Forest", "21"), ("Decision Tree", "33"), ("XGBoost", "12"), ("Linear Regression", "9")]:
    x, y = p11["train"][m], p11["train"][f"{m} (VM-agnostic)"]
    claim(f"{pct}\\%", round(100 * (1 - y["unseen"]["MAE"] / x["unseen"]["MAE"]), 1))
claim("worst VM 30.4 to 8.7", (round(p11["train"]["Random Forest"]["unseen_per_vm"]["worst_vm_mae"], 2), round(p11["train"]["Random Forest (VM-agnostic)"]["unseen_per_vm"]["worst_vm_mae"], 2)))
claim("6--10\\% higher error", [round(100 * (p11["train"][f"{m} (VM-agnostic)"]["test"]["MAE"] / p11["train"][m]["test"]["MAE"] - 1), 1) for m in ["Random Forest", "XGBoost", "LightGBM"]])
# ---- robust TOPSIS
claim("($C=0.967$)", round(rob.loc["LightGBM (VM-agnostic)", "Robust C"], 4))
rch = rsw[rsw.winner != rsw.winner.shift()][["unseen_weight", "winner"]].values.tolist()
claim("weight of 0.10", rch); claim("only above 0.60", rch)
# ---- significance
s1 = sig[(sig.Comparison == "XGBoost vs LightGBM")]
claim("27 of the 50 VMs", 50 - int(s1.iloc[0]["VMs where A better"])); claim("$p=0.005$", round(s1.iloc[0]["Paired t p"], 4))
claim("$p=0.08$", round(s1.iloc[0]["Wilcoxon p"], 3)); claim("71 of 100 VMs", 100 - int(s1.iloc[1]["VMs where A better"]))
claim("$p=0.71$", round(s1.iloc[1]["Paired t p"], 3))
claim("46 of 50 test VMs", int(sig[sig.Comparison == "LightGBM vs LSTM"].iloc[0]["VMs where A better"]))
# ---- fleet
claim("0.018\\,kg", round(fleet.loc["Linear Regression (VM-agnostic)", "Total CO2 (kg/yr)"], 4))
claim("0.64\\,kg", round(fleet.loc["LightGBM (VM-agnostic)", "Total CO2 (kg/yr)"], 3))
claim("17\\,kg for the LSTM", round(fleet.loc["LSTM", "Total CO2 (kg/yr)"], 2)); claim("25\\,kg for Random Forest", round(fleet.loc["Random Forest", "Total CO2 (kg/yr)"], 2))
claim("more than 97\\%", round(100 - fleet["Inference share (%)"].max(), 2))
claim("26--39 times", (round(fleet.loc["LSTM", "Total CO2 (kg/yr)"] / fleet.loc["LightGBM (VM-agnostic)", "Total CO2 (kg/yr)"], 1),
                       round(fleet.loc["Random Forest", "Total CO2 (kg/yr)"] / fleet.loc["LightGBM (VM-agnostic)", "Total CO2 (kg/yr)"], 1)))

for text, value, found in claims:
    print(f"{'in paper' if found else 'NOT FOUND'} | {text:<34} | computed: {value}")
