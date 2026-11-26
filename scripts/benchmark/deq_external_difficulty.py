"""does the DEQ solver effort flag high-error leads on independent data?

Fold-0 DEQ, one forward pass per 10-s lead (as for LUDB in deq_analysis.py), on Zhejiang and ISP (global labels applied
to every lead; six onset/offset points; +-150 ms matching inside the annotated span).
Per lead: Anderson iterations, mean predictive entropy, mean |boundary error| of matched points, missed + false points.
Everything that defines a decision is fixed on the LUDB internal test (results/deq_windows.csv) before looking at the
external data:
  high-error lead (primary)   : mean |error| >= LUDB 75th percentile of mean |error|
  high-error lead (composite) : primary OR missed+false > LUDB 75th percentile of missed+false (= 0, i.e. any)
  flag rule                   : score >= LUDB 75th percentile of that score (iterations, entropy)
Reported per database: AUROC of iterations, entropy and their rank sum (record-cluster bootstrap CI of differences),
record-level Spearman correlation, and sensitivity/precision of the fixed flag rules.
Output: results/deq_external_windows.csv, results/tables/deq_external_difficulty.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as ss
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import extra_data  # noqa: E402
from deq_analysis import load, run_window  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

POINTS = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
TOL = int(round(150 * data.FS / 1000))


def lead_rows(m, name):
    rows = []
    for s in extra_data.get(name):
        s0, s1 = s.span
        if s1 - s0 <= 0:
            continue
        p, it, res, ent = run_window(m, s.x)
        pred = mask_to_points(p.argmax(0), s.x)
        acc = Accumulator()
        for q in POINTS:
            if q not in s.points:
                continue
            ref = s.points[q][(s.points[q] >= s0) & (s.points[q] < s1)]
            acc.add(q, s.subject, ref, pred.get(q, np.zeros(0, int)), TOL, fp_windows=[(s0, s1)])
        errs = [e for v in acc.c.values() for e in v["err"]]
        rows.append({"db": name, "record": s.record, "lead": s.lead, "iters": it, "entropy": ent,
                     "mae_ms": float(np.mean(np.abs(errs))) if errs else np.nan,   # Accumulator errors are in ms
                     "n_missed_false": int(sum(v["fn"] + v["fp"] for v in acc.c.values()))})
    return rows


def auc_ci(d, y, rng, B=2000):
    recs = d.record.unique()
    g = {r: np.flatnonzero(d.record.values == r) for r in recs}
    a_it, a_en, a_bo = [], [], []
    for _ in range(B):
        idx = np.concatenate([g[r] for r in rng.choice(recs, len(recs))])
        yy, sub = y[idx], d.iloc[idx]
        if yy.min() == yy.max():
            continue
        both = ss.rankdata(sub.iters) + ss.rankdata(sub.entropy)
        e = roc_auc_score(yy, sub.entropy)
        a_it.append(roc_auc_score(yy, sub.iters) - e)
        a_bo.append(roc_auc_score(yy, both) - e)
    return np.percentile(a_it, [2.5, 97.5]), np.percentile(a_bo, [2.5, 97.5])


def main():
    torch.set_num_threads(2)
    lud = pd.read_csv(RESULTS / "deq_windows.csv")
    lud = lud[lud.db == "LUDB"]
    # LUDB stores mae in ms already; thresholds fixed here
    thr_err = float(lud.mae_ms.quantile(0.75))
    thr_mf = float(lud.n_missed_false.quantile(0.75))
    thr_it = float(lud.iters.quantile(0.75))
    thr_en = float(lud.entropy.quantile(0.75))
    f = RESULTS / "deq_external_windows.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        m = load()
        df = pd.DataFrame(lead_rows(m, "zj") + lead_rows(m, "isp"))
        df.to_csv(f, index=False)
    rng = np.random.default_rng(0)
    md = ["### DEQ solver effort on independent databases (fold-0 DEQ, one pass per 10-s lead)", "",
          f"Thresholds fixed on the LUDB internal test: high error = mean |error| ≥ {thr_err:.1f} ms; composite also "
          f"missed + false > {thr_mf:.0f}; flag = iterations ≥ {thr_it:.0f} or entropy ≥ {thr_en:.3f} (LUDB 75th percentiles).", "",
          "| database | leads (records) | high-error leads | AUROC iterations | AUROC entropy | AUROC both | iterations − entropy [95% CI] | both − entropy [95% CI] | record-level ρ (iterations, error) | flag by iterations: Se / PPV | flag by entropy: Se / PPV |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for db, d in df.groupby("db", sort=False):
        d = d.dropna(subset=["mae_ms"]).reset_index(drop=True)
        for lab, y in [("primary", (d.mae_ms >= thr_err).astype(int).values),
                       ("composite", ((d.mae_ms >= thr_err) | (d.n_missed_false > thr_mf)).astype(int).values),
                       ("worst quartile within database, descriptive", (d.mae_ms >= d.mae_ms.quantile(0.75)).astype(int).values)]:
            both = ss.rankdata(d.iters) + ss.rankdata(d.entropy)
            ai, ae, ab = roc_auc_score(y, d.iters), roc_auc_score(y, d.entropy), roc_auc_score(y, both)
            ci_it, ci_bo = auc_ci(d, y, rng)
            g = d.groupby("record").agg(iters=("iters", "mean"), mae=("mae_ms", "mean"))
            rho = ss.spearmanr(g.iters, g.mae).statistic
            fi, fe = d.iters.values >= thr_it, d.entropy.values >= thr_en
            se_i, pp_i = 100 * (fi & (y == 1)).sum() / max(y.sum(), 1), 100 * (fi & (y == 1)).sum() / max(fi.sum(), 1)
            se_e, pp_e = 100 * (fe & (y == 1)).sum() / max(y.sum(), 1), 100 * (fe & (y == 1)).sum() / max(fe.sum(), 1)
            name = {"zj": "Zhejiang", "isp": "ISP"}[db]
            md.append(f"| {name} ({lab}) | {len(d)} ({d.record.nunique()}) | {int(y.sum())} | {ai:.3f} | {ae:.3f} | {ab:.3f} | "
                      f"{ai - ae:+.3f} [{ci_it[0]:+.3f}, {ci_it[1]:+.3f}] | {ab - ae:+.3f} [{ci_bo[0]:+.3f}, {ci_bo[1]:+.3f}] | "
                      f"{rho:.2f} | {se_i:.1f} / {pp_i:.1f} | {se_e:.1f} / {pp_e:.1f} |")
    (RESULTS / "tables" / "deq_external_difficulty.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
