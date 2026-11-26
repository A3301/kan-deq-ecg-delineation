"""linear, GELU-MLP and KAN classification heads on the same U-Net encoder.

Models (5-fold ensembles, results/final): linear = U-Net (36 head parameters); KAN head = unet_kan@abl-headonly (404);
GELU MLP head = unet_kan@abl-mlphead (397, hidden 29). Each seed separately (seed 1, and seed 2 when present).
Endpoints: LUDB internal-test mean F1 over the six onset/offset points and P-onset F1 (subject paired bootstrap,
10 000); external context without tuning: QTDB P onset (each lead), BUT PDB P peak without overlap (each lead),
MIT-BIH P-wave P peak (each lead).
Output: results/tables/head_comparison.md, results/head_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
HEADS = {"linear": "unet", "GELU MLP": "unet_kan__abl-mlphead", "KAN": "unet_kan__abl-headonly"}
EXT = [("qtdb_each_lead", "P_on", "QTDB P onset"), ("butpdb_nooverlap_each_lead", "P_peak", "BUT PDB P peak (no overlap)"),
       ("pwave_each_lead", "P_peak", "MIT-BIH P-wave P peak")]


def name(base, seed):
    return base if seed == 1 else f"{base}__seed{seed}"


def mean6_diff(a, b, rng):
    ra, rb = stats.load_records(a)["ludb_test"]["subj"], stats.load_records(b)["ludb_test"]["subj"]
    keys = sorted(ra["P_on"])
    idx = rng.integers(0, len(keys), (stats.B, len(keys)))
    fa, fb, da, db = [], [], [], []
    for q in SIX:
        ta, na, pa, _ = stats.subj_arrays(ra[q], keys)
        tb, nb, pb, _ = stats.subj_arrays(rb[q], keys)
        fa.append(stats.f1(ta.sum(), na.sum(), pa.sum()))
        fb.append(stats.f1(tb.sum(), nb.sum(), pb.sum()))
        da.append(stats.f1(ta[idx].sum(1), na[idx].sum(1), pa[idx].sum(1)))
        db.append(stats.f1(tb[idx].sum(1), nb[idx].sum(1), pb[idx].sum(1)))
    d = np.mean(da, 0) - np.mean(db, 0)
    return float(np.mean(fa)), float(np.mean(fb)), float(np.mean(fa) - np.mean(fb)), np.percentile(d, [2.5, 97.5]).tolist()


def main():
    out = {"seeds": {}}
    md = ["### Classification heads on the same U-Net encoder (LUDB internal test; external sets without tuning)", ""]
    for seed in (1, 2):
        mods = {h: name(b, seed) for h, b in HEADS.items()}
        if not all((FINAL / f"{m}_records.pkl").exists() for m in mods.values()):
            md.append(f"Seed {seed}: not all three heads evaluated yet.")
            md.append("")
            continue
        res = {"models": {}, "pairs": []}
        md += [f"**Seed {seed}**", "",
               "| head | parameters (head) | LUDB mean of 6 | LUDB P onset | " + " | ".join(e[2] for e in EXT) + " |",
               "|---|---|---|---|" + "---|" * len(EXT)]
        for h, m in mods.items():
            j = json.loads((FINAL / f"{m}.json").read_text())
            ph = {"linear": 36, "GELU MLP": 397, "KAN": 404}[h]
            m6 = float(np.mean([j["ludb_test"][q]["F1"] for q in SIX]))
            row = {"params": j["params"], "head_params": ph, "mean6": m6, "P_on": j["ludb_test"]["P_on"]["F1"]}
            for part, q, lab in EXT:
                row[lab] = j[part][q]["F1"]
            res["models"][h] = row
            md.append(f"| {h} | {j['params']:,} ({ph}) | {m6:.2f} | {row['P_on']:.2f} | "
                      + " | ".join(f"{row[e[2]]:.2f}" for e in EXT) + " |")
        md += ["", "| comparison | LUDB mean of 6: Δ [95% CI] | LUDB P onset: Δ [95% CI] | " + " | ".join(e[2] + ": Δ [95% CI]" for e in EXT) + " |",
               "|---|---|---|" + "---|" * len(EXT)]
        for a, b in [("KAN", "linear"), ("KAN", "GELU MLP"), ("GELU MLP", "linear")]:
            # fresh rng(0) per comparison, as in make_tables.py, so shared comparisons reproduce the paper's CIs
            _, _, d6, ci6 = mean6_diff(mods[a], mods[b], np.random.default_rng(0))
            cp = stats.compare(mods[a], mods[b], "ludb_test", "P_on", np.random.default_rng(0))
            cells = [f"{d6:+.2f} [{ci6[0]:+.2f}, {ci6[1]:+.2f}]",
                     f"{cp['diff']:+.2f} [{cp['diff_CI95'][0]:+.2f}, {cp['diff_CI95'][1]:+.2f}]"]
            pair = {"A": a, "B": b, "mean6": [d6, ci6], "P_on": [cp["diff"], cp["diff_CI95"]]}
            for part, q, lab in EXT:
                ce = stats.compare(mods[a], mods[b], part, q, np.random.default_rng(0))
                cells.append(f"{ce['diff']:+.2f} [{ce['diff_CI95'][0]:+.2f}, {ce['diff_CI95'][1]:+.2f}]")
                pair[lab] = [ce["diff"], ce["diff_CI95"]]
            res["pairs"].append(pair)
            md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
        md.append("")
        out["seeds"][seed] = res
    (ROOT / "results" / "head_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "results" / "tables" / "head_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
