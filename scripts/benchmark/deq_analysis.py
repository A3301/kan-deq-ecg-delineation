"""Phase 4/5: DEQ convergence and iteration-vs-difficulty analysis (H2c) + solver ablation (plan section 9.4).

For every 10-s window (LUDB internal-test lead signals; 8.2-s windows of QTDB/BUT PDB/pwave) run the fold-0 DEQ with
batch size 1 and record the Anderson iteration count and final relative residual. Difficulty indicators:
  noise   : fraction of signal power in 15-40 Hz (muscle/HF noise proxy) of the filtered input
  entropy : mean predictive entropy of the softmax output
  error   : LUDB only, per-window mean |error| of matched boundaries and number of missed+false points
  group   : LUDB rhythm group / BUT PDB pathology tokens / database
Outputs: results/deq_analysis.json, results/deq_windows.csv, figures/deq/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.signal import welch  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from evaluate import LUDB_POINTS, ludb_eval  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import FIGURES, RESULTS, ROOT  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

FS = data.FS


def load(fold=0, model="deq"):
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{chosen[model]}_{model}_f{fold}_s1.pt",
                    weights_only=False)
    m = M.build(model, **ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def hf_frac(x):
    f, p = welch(x, fs=FS, nperseg=256)
    tot = p[(f >= 0.5) & (f <= 40)].sum()
    return float(p[(f >= 15) & (f <= 40)].sum() / max(tot, 1e-12))


@torch.no_grad()
def run_window(m, x):
    L = int(np.ceil(len(x) / 16) * 16)
    xp = np.zeros(L, np.float32)
    xp[:len(x)] = x
    out = m(torch.from_numpy(xp)[None, None])[0, :, :len(x)]
    p = torch.softmax(out, 0).numpy()
    ent = float((-(p * np.log(p + 1e-12)).sum(0)).mean())
    info = m.last_info
    return p, int(info["n_iter"]), float(info["rel_res"].max()), ent


def windows():
    sp = splits.load()
    rg = sp["rhythm_group"]
    for s in data.get("ludb"):
        if s.subject in set(sp["test"]):
            yield "LUDB", s, s.x, 0, rg[s.subject]
    for nm, ds in [("QTDB", data.get("qtdb")), ("BUTPDB", data.get("butpdb")), ("PWAVE", data.get("pwave"))]:
        for s in ds:
            s0, s1 = s.span
            step = 2048
            for st in range(s0, max(s0 + 1, s1 - step), step * (1 if nm != "PWAVE" else 4)):
                grp = s.meta.get("pathology", nm) if nm == "BUTPDB" else nm
                yield nm, s, s.x[st:st + step], st, grp


def main():
    torch.set_num_threads(6)
    m = load()
    rows = []
    tol = int(round(150 * FS / 1000))
    for db, s, x, st, grp in windows():
        if len(x) < 512:
            continue
        p, it, res, ent = run_window(m, x)
        row = {"db": db, "record": s.record, "lead": s.lead, "start": st, "group": grp, "iters": it,
               "rel_res": res, "entropy": ent, "hf_frac": hf_frac(x)}
        if db == "LUDB":
            pred = mask_to_points(p.argmax(0), x)
            acc = Accumulator()
            s0, s1 = s.span
            for q in LUDB_POINTS:
                if q.endswith("peak"):
                    continue
                ref = s.points[q][(s.points[q] >= s0) & (s.points[q] < s1)]
                acc.add(q, s.subject, ref, pred[q], tol, fp_windows=[(s0, s1)])
            errs = [e for v in acc.c.values() for e in v["err"]]
            row["mae_ms"] = float(np.mean(np.abs(errs))) if errs else np.nan
            row["n_missed_false"] = int(sum(v["fn"] + v["fp"] for v in acc.c.values()))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "deq_windows.csv", index=False)
    out = {"n_windows": len(df), "max_iter": m.max_iter, "tol": m.tol,
           "converged_frac": float((df.rel_res < m.tol).mean()),
           "iters_by_db": df.groupby("db").iters.describe().round(2).to_dict("index")}
    tests = {}
    for col in ["hf_frac", "entropy"]:
        r = spearmanr(df.iters, df[col])
        tests[f"all_iters_vs_{col}"] = {"rho": float(r.statistic), "p_two_sided": float(r.pvalue)}
    lud = df[df.db == "LUDB"]
    for col in ["mae_ms", "n_missed_false", "hf_frac", "entropy"]:
        r = spearmanr(lud.iters, lud[col], nan_policy="omit")
        tests[f"ludb_iters_vs_{col}"] = {"rho": float(r.statistic), "p_two_sided": float(r.pvalue)}
    a, b = lud[lud.group != "sinus"].iters, lud[lud.group == "sinus"].iters
    u = mannwhitneyu(a, b, alternative="greater")
    tests["ludb_pathological_vs_sinus_iters"] = {"median_path": float(a.median()), "median_sinus": float(b.median()),
                                                 "U": float(u.statistic), "p_one_sided": float(u.pvalue)}
    ext = df[df.db != "LUDB"]
    u = mannwhitneyu(ext.iters, lud.iters, alternative="greater")
    tests["external_vs_ludb_iters"] = {"median_ext": float(ext.iters.median()), "median_ludb": float(lud.iters.median()),
                                       "p_one_sided": float(u.pvalue)}
    out["tests"] = tests

    # Solver ablation on LUDB internal test (fold-0 model): tol x max_iter
    sp = splits.load()
    test = [s for s in data.get("ludb") if s.subject in set(sp["test"])]
    abl = []
    for tl in [1e-2, 1e-3, 1e-4]:
        for mi in [10, 30, 60]:
            m.tol, m.max_iter = tl, mi
            iters = []
            orig = m.core

            def core(u, _orig=orig):
                z = _orig(u)
                iters.append(m.last_info["n_iter"])
                return z
            m.core = core
            tab = ludb_eval([m], test).table()
            m.core = orig
            abl.append({"tol": tl, "max_iter": mi, "mean_iters": float(np.mean(iters)),
                        "meanF1_onoff": float(np.mean([tab[k]["F1"] for k in
                                                       ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]])),
                        "P_on_mae": tab["P_on"]["mae_ms"], "T_off_mae": tab["T_off"]["mae_ms"]})
            print(abl[-1], flush=True)
    out["solver_ablation"] = abl
    (RESULTS / "deq_analysis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    fd = FIGURES / "deq"
    fd.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    order = ["LUDB", "QTDB", "BUTPDB", "PWAVE"]
    ax[0].boxplot([df[df.db == d].iters for d in order], tick_labels=order)
    ax[0].set_ylabel("Anderson iterations")
    ax[0].set_title("DEQ iterations by database")
    ax[1].scatter(df.hf_frac, df.iters + np.random.default_rng(0).uniform(-.3, .3, len(df)), s=4, alpha=.4)
    ax[1].set_xlabel("HF power fraction (15-40 Hz)")
    ax[1].set_ylabel("iterations (jittered)")
    ax[1].set_title(f"rho={tests['all_iters_vs_hf_frac']['rho']:.2f}")
    ax[2].scatter(lud.mae_ms, lud.iters + np.random.default_rng(1).uniform(-.3, .3, len(lud)), s=4, alpha=.4)
    ax[2].set_xlabel("LUDB window mean |boundary error| (ms)")
    ax[2].set_title(f"rho={tests['ludb_iters_vs_mae_ms']['rho']:.2f}")
    fig.tight_layout()
    fig.savefig(fd / "deq_iterations_difficulty.png", dpi=150)
    plt.close(fig)
    print(json.dumps(out["tests"], indent=1))


if __name__ == "__main__":
    main()
