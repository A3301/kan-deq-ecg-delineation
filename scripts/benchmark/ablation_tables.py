"""Ablation summary (analysis plan) from results/cv/abl-*.json (LUDB 5-fold CV, validation mean F1 of the six
on/off points, edge-tolerant metric), the main-model CV re-evaluation, lead fusion and 500-Hz runs if present.
Output: results/tables/ablations.md/.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS  # noqa: E402

CV = RESULTS / "cv"
T = RESULTS / "tables"
re = json.loads((RESULTS / "cv_reeval.json").read_text())
rows = []


def add(group, label, tag, model, ref=None):
    vals, params = [], None
    for f in range(5):
        p = CV / f"{tag}_{model}_f{f}_s1.json"
        if p.exists():
            r = json.loads(p.read_text())
            vals.append(r["val_meanF1"])
            params = r["params"]
    if not vals:
        return
    row = {"ablation": group, "variant": label, "params": params, "n_folds": len(vals),
           "CV mean F1": f"{np.mean(vals):.2f}", "SD": f"{np.std(vals, ddof=1):.2f}" if len(vals) > 1 else ""}
    if ref is not None and len(vals) == 5:
        row["diff vs main"] = f"{np.mean(vals) - re[ref]['mean']:+.2f}"
    rows.append(row)


rows.append({"ablation": "KAN (main)", "variant": "U-Net-KAN, B-spline G=8 k=3, bottleneck+head", "params": 407828,
             "n_folds": 5, "CV mean F1": f"{re['unet_kan']['mean']:.2f}", "SD": f"{re['unet_kan']['sd']:.2f}"})
for g in [4, 8, 16]:
    for o in [1, 3]:
        if (g, o) != (8, 3):
            add("KAN grid x order", f"G={g}, k={o}", f"abl-g{g}o{o}", "unet_kan", "unet_kan")
add("KAN placement", "head only (param-matched to U-Net)", "abl-headonly", "unet_kan", "unet_kan")
add("KAN placement", "bottleneck only", "abl-bnonly", "unet_kan", "unet_kan")
add("KAN basis", "RBF (FastKAN) instead of B-spline", "abl-rbf", "unet_kan", "unet_kan")
rows.append({"ablation": "reference", "variant": "U-Net (no KAN)", "params": 308772, "n_folds": 5,
             "CV mean F1": f"{re['unet']['mean']:.2f}", "SD": f"{re['unet']['sd']:.2f}"})
rows.append({"ablation": "DEQ (main)", "variant": "implicit backward", "params": 103348, "n_folds": 5,
             "CV mean F1": f"{re['deq']['mean']:.2f}", "SD": f"{re['deq']['sd']:.2f}"})
add("DEQ backward", "phantom (1-step) gradient", "abl-phantom", "deq_phantom", "deq")
for m, lab in [("unet", "U-Net"), ("unet_kan", "U-Net-KAN")]:
    add("input rate", f"{lab} at 500 Hz", "fs500", m, m)
df = pd.DataFrame(rows).fillna("")
T.mkdir(parents=True, exist_ok=True)
df.to_csv(T / "ablations.csv", index=False)
txt = "### Ablations: LUDB 5-fold CV, validation mean F1 over 6 on/off points (edge-tolerant metric)\n\n" + \
      df.to_markdown(index=False) + "\n"
lf = RESULTS / "lead_fusion.json"
if lf.exists():
    d = json.loads(lf.read_text())
    lines = ["", "### Lead count: single-lead vs 12-lead probability fusion (LUDB internal test, ensembles)", "",
             "| model | P_on single / fused | P_off | QRS_on | QRS_off | T_off |", "|---|---|---|---|---|---|"]
    for m, v in d.items():
        cells = [f"{v['single_lead'][q]['F1']:.2f} / {v['fused_12_lead'][q]['F1']:.2f}"
                 for q in ["P_on", "P_off", "QRS_on", "QRS_off", "T_off"]]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    txt += "\n".join(lines) + "\n"
dq = RESULTS / "deq_analysis.json"
if dq.exists():
    a = json.loads(dq.read_text())["solver_ablation"]
    txt += "\n### DEQ solver tolerance x iteration cap (inference, fold-0 model, LUDB internal test)\n\n| tol | max_iter |" \
           " mean iterations | mean F1 (6 on/off) |\n|---|---|---|---|\n" + "\n".join(
               f"| {r['tol']:g} | {r['max_iter']} | {r['mean_iters']:.1f} | {r['meanF1_onoff']:.2f} |" for r in a) + "\n"
(T / "ablations.md").write_text(txt, encoding="utf-8")
print(txt)
