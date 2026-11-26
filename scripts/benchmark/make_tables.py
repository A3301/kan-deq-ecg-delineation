"""build result tables (Markdown + CSV) automatically from results/*.json (analysis plan).

Outputs results/tables/: cv_summary, ludb_test, qtdb, p_external, primary_family, h1, h2, efficiency, subgroups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS, ROOT  # noqa: E402

T = RESULTS / "tables"
T.mkdir(parents=True, exist_ok=True)
BASE = ["unet", "cnn_bilstm", "transformer"]
PROP = ["unet_kan", "convkan", "deq"]
CTRL = ["deq_unrolled", "explicit_stack"]
ALL = BASE + PROP + CTRL
ONOFF = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
NICE = {"unet": "U-Net", "cnn_bilstm": "CNN-BiLSTM", "transformer": "Transformer", "unet_kan": "U-Net-KAN",
        "convkan": "ConvKAN", "deq": "DEQ (implicit)", "deq_unrolled": "DEQ-unrolled", "explicit_stack":
        "Explicit stack", "wavelet": "Wavelet (NeuroKit2 dwt)"}


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(v, d=2):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write(df, name, title):
    df.to_csv(T / f"{name}.csv", index=False)
    (T / f"{name}.md").write_text(f"### {title}\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


def cv_summary():
    chosen = jl(ROOT / "configs" / "chosen_lr.json")
    rows = []
    for m in ALL:
        if m not in chosen:
            continue
        vals, walls, params = [], [], None
        for f in range(5):
            p = RESULTS / "cv" / f"lr{chosen[m]}_{m}_f{f}_s1.json"
            if p.exists():
                r = jl(p)
                vals.append(r["val_meanF1"])
                walls.append(r["wall_s"])
                params = r["params"]
        re = jl(RESULTS / "cv_reeval.json").get(m, {}) if (RESULTS / "cv_reeval.json").exists() else {}
        rows.append({"model": NICE[m], "lr": chosen[m], "params": params, "n_folds": len(vals),
                     "CV val mean F1 (edge-tolerant metric)": fmt(re.get("mean")),
                     "SD over folds": fmt(re.get("sd")),
                     "CV val mean F1 (at training time)": fmt(np.mean(vals)) if vals else "",
                     "train wall min/fold": fmt(np.mean(walls) / 60, 1) if walls else ""})
    df = pd.DataFrame(rows)
    write(df, "cv_summary", "LUDB 5-fold CV (validation folds), chosen learning rate")
    return df


def best_baseline():
    """Best DL baseline by LUDB CV mean F1, recomputed with the edge-tolerant metric (results/cv_reeval.json)."""
    re = jl(RESULTS / "cv_reeval.json")
    sc = {m: re[m]["mean"] for m in BASE if m in re}
    return max(sc, key=sc.get), sc


def point_table(part, points, name, title, models):
    rows = []
    for m in models:
        p = RESULTS / "final" / f"{m}.json"
        if not p.exists():
            continue
        r = jl(p)[part]
        for q in points:
            if q not in r:
                continue
            v = r[q]
            c = stats.ci(m, part, q)
            rows.append({"model": NICE[m], "point": q, "Se": fmt(v["Se"]), "PPV": fmt(v["PPV"]),
                         "F1 [95% CI]": f"{fmt(v['F1'])} [{fmt(c['F1'][0])}, {fmt(c['F1'][1])}]",
                         "mean±SD (ms)": f"{fmt(v['mean_ms'], 1)} ± {fmt(v['sd_ms'], 1)}",
                         "MAE (ms)": fmt(v["mae_ms"], 1), "TP/FN/FP": f"{v['TP']}/{v['FN']}/{v['FP']}"})
    df = pd.DataFrame(rows)
    write(df, name, title)
    return df


def wavelet_rows():
    p = RESULTS / "baselines" / "wavelet_neurokit_dwt.json"
    if not p.exists():
        return None
    w = jl(p)
    rows = []
    for part, key in [("LUDB internal test", "ludb_test"), ("QTDB (each lead)", "qtdb_each_lead"),
                      ("BUT PDB (each lead)", "butpdb_each_lead"), ("pwave (each lead)", "pwave_each_lead")]:
        for q, v in w[key]["table"].items():
            rows.append({"set": part, "point": q, "Se": fmt(v["Se"]), "PPV": fmt(v["PPV"]), "F1": fmt(v["F1"]),
                         "mean±SD (ms)": f"{fmt(v['mean_ms'], 1)} ± {fmt(v['sd_ms'], 1)}"})
    df = pd.DataFrame(rows)
    write(df, "wavelet", "Wavelet baseline (NeuroKit2 dwt; no training)")
    return df


def primary_family():
    bb, sc = best_baseline()
    res = []
    for m in PROP:
        if (RESULTS / "final" / f"{m}_records.pkl").exists():
            res.append(stats.compare(m, bb, "butpdb_each_lead", "P_peak"))
    adj = stats.holm([r["p_boot"] for r in res])
    for r, a in zip(res, adj):
        r["p_holm"] = a
    rows = [{"comparison": f"{NICE[r['A']]} vs {NICE[r['B']]}", "F1 A": fmt(r["F1_A"]), "F1 B": fmt(r["F1_B"]),
             "diff [95% CI]": f"{fmt(r['diff'])} [{fmt(r['diff_CI95'][0])}, {fmt(r['diff_CI95'][1])}]",
             "p (bootstrap)": fmt(r["p_boot"], 4), "p (Holm)": fmt(r["p_holm"], 4),
             "p (Wilcoxon, per-subject F1)": fmt(r["p_wilcoxon"], 4), "n subjects": r["n_subjects"]} for r in res]
    df = pd.DataFrame(rows)
    write(df, "primary_family", f"Primary endpoint: BUT PDB P_peak F1 (±150 ms, each lead). Best baseline = "
                                f"{NICE[bb]} (LUDB CV mean F1 {sc[bb]:.2f}; chosen before external data)")
    (T / "primary_family.json").write_text(json.dumps({"best_baseline": bb, "cv_scores": sc, "tests": res},
                                                      indent=1), encoding="utf-8")
    return df


def secondary_family():
    bb, _ = best_baseline()
    tests = []
    for part, q in [("pwave_each_lead", "P_peak"), ("qtdb_each_lead", "P_on"), ("qtdb_each_lead", "P_off"),
                    ("butpdb_nooverlap_each_lead", "P_peak")]:
        for m in PROP:
            if (RESULTS / "final" / f"{m}_records.pkl").exists():
                tests.append(stats.compare(m, bb, part, q))
    adj = stats.holm([r["p_boot"] for r in tests])
    rows = []
    for r, a in zip(tests, adj):
        rows.append({"set": r["part"], "point": r["point"], "comparison": f"{NICE[r['A']]} vs {NICE[r['B']]}",
                     "F1 A": fmt(r["F1_A"]), "F1 B": fmt(r["F1_B"]),
                     "diff [95% CI]": f"{fmt(r['diff'])} [{fmt(r['diff_CI95'][0])}, {fmt(r['diff_CI95'][1])}]",
                     "p (Holm, family)": fmt(a, 4)})
    write(pd.DataFrame(rows), "secondary_family", "Key secondary external P comparisons (exploratory, Holm)")


def h1_h2():
    bb, _ = best_baseline()
    rows = []
    cands = [("H1", "unet_kan", "unet"), ("H1", "convkan", "unet"), ("H1", "unet_kan", bb), ("H1", "convkan", bb),
             ("H2b", "deq", "deq_unrolled"), ("H2b", "deq", "explicit_stack")]
    for h, a, b in cands:
        pa, pb = RESULTS / "final" / f"{a}_records.pkl", RESULTS / "final" / f"{b}_records.pkl"
        if not (pa.exists() and pb.exists()):
            continue
        # mean F1 over 6 on/off points: bootstrap the mean of per-point pooled F1 on the same subject resample
        ra, rb = stats.load_records(a)["ludb_test"]["subj"], stats.load_records(b)["ludb_test"]["subj"]
        keys = sorted(ra["P_on"])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(keys), (stats.B, len(keys)))
        da, db_ = [], []
        fa, fb = [], []
        for q in ONOFF:
            ta, na, pa_, _ = stats.subj_arrays(ra[q], keys)
            tb, nb, pb_, _ = stats.subj_arrays(rb[q], keys)
            fa.append(stats.f1(ta.sum(), na.sum(), pa_.sum()))
            fb.append(stats.f1(tb.sum(), nb.sum(), pb_.sum()))
            da.append(stats.f1(ta[idx].sum(1), na[idx].sum(1), pa_[idx].sum(1)))
            db_.append(stats.f1(tb[idx].sum(1), nb[idx].sum(1), pb_[idx].sum(1)))
        d = np.mean(da, 0) - np.mean(db_, 0)
        lo, hi = np.percentile(d, [2.5, 97.5])
        par = {m: jl(RESULTS / "final" / f"{m}.json")["params"] for m in (a, b)}
        rows.append({"hypothesis": h, "A": NICE[a], "B": NICE[b], "params A": par[a], "params B": par[b],
                     "mean F1 A": fmt(np.mean(fa)), "mean F1 B": fmt(np.mean(fb)),
                     "diff [95% CI]": f"{fmt(np.mean(fa) - np.mean(fb))} [{fmt(lo)}, {fmt(hi)}]",
                     "non-inferior (lower CI > -1.0)": "yes" if lo > -1.0 else "no"})
    write(pd.DataFrame(rows), "h1_h2_noninferiority", "LUDB internal test, mean F1 over 6 on/off points "
                                                       "(ensemble of 5 fold models), non-inferiority margin -1.0 pp")


def subgroups():
    import data
    but = {s.record: s.meta["pathology"] for s in data.get("butpdb")}
    subj_of = {s.record: s.subject for s in data.get("butpdb")}
    groups = {"AV block (BI/BII/BIII)": ["BI", "BII", "BIII"], "Atrial ectopy (A/a)": ["A", "a"],
              "Ventricular ectopy (V/B/T/VT/VP/E/F)": ["V", "B", "T", "VT", "VP", "E", "F"],
              "AF/AFL": ["AFIB", "AFL", "FIB"], "Nodal/SVTA (NOD/J/SVTA)": ["NOD", "J", "SVTA"],
              "Paced/BBB/other (P/L/R/PREX/IVR/NA/VFL)": ["P", "L", "R", "PREX", "IVR", "NA", "VFL"]}
    rows = []
    for m in ["wavelet"] + BASE + PROP:
        if not (RESULTS / "final" / f"{m}_records.pkl").exists():
            continue
        subj = stats.load_records(m)["butpdb_each_lead"]["subj"]["P_peak"]
        for g, toks in groups.items():
            recs = [r for r, pth in but.items() if any(t in [x.strip() for x in str(pth).split(",")] for t in toks)]
            ss = sorted({subj_of[r] for r in recs})
            tp = sum(subj[s]["tp"] for s in ss if s in subj)
            fn = sum(subj[s]["fn"] for s in ss if s in subj)
            fp = sum(subj[s]["fp"] for s in ss if s in subj)
            rows.append({"model": NICE[m], "subgroup": g, "n records": len(recs),
                         "Se": fmt(100 * tp / max(tp + fn, 1)), "PPV": fmt(100 * tp / max(tp + fp, 1)),
                         "F1": fmt(200 * tp / max(2 * tp + fn + fp, 1))})
    write(pd.DataFrame(rows), "butpdb_subgroups",
          "BUT PDB P_peak by pathology subgroup (subject-pooled; a record can be in several groups; note: BUT PDB "
          "subjects are source records, so two segments of one source record pool together)")


CSE_2SD = {"P_on": 10.2, "P_off": 12.7, "QRS_on": 6.5, "QRS_off": 11.6, "T_off": 30.6}  # Di Marco 2011 Tab.6


def cse_table():
    rows = []
    for part, lab in [("ludb_test", "LUDB test"), ("qtdb_each_lead", "QTDB each lead"),
                      ("qtdb_best_lead", "QTDB best lead")]:
        for m in ["wavelet"] + ALL:
            if m == "wavelet":
                p = RESULTS / "baselines" / "wavelet_neurokit_dwt.json"
                if not p.exists() or part not in jl(p):
                    continue
                tab = jl(p)[part]["table"]
            else:
                p = RESULTS / "final" / f"{m}.json"
                if not p.exists():
                    continue
                tab = jl(p)[part]
            row = {"set": lab, "model": NICE[m]}
            for q, lim in CSE_2SD.items():
                sd = tab[q]["sd_ms"]
                row[f"{q} SD (2σCSE={lim})"] = f"{sd:.1f}{' ✓' if sd <= lim else ''}"
            rows.append(row)
    write(pd.DataFrame(rows), "cse_compliance", "CSE compliance: SD of boundary error vs 2σ_CSE tolerance "
                                                "(✓ = SD ≤ 2σ_CSE)")


def efficiency():
    p = RESULTS / "efficiency_trained.json"
    if not p.exists():
        return
    e = jl(p)
    rows = []
    for m in ALL:
        if m not in e:
            continue
        r = e[m]
        rows.append({"model": NICE[m], "params": r["params"], "GFLOPs / 10-s lead": fmt(r["gflops_10s"], 3),
                     "latency 1 thr (ms)": fmt(r["latency_ms_1thr"], 1), "latency 6 thr (ms)": fmt(r["latency_ms_6thr"], 1),
                     "train mem increase (MB)": fmt(r["train_mem_increase_mb"], 1),
                     "DEQ iters (this input)": r.get("deq_iters_last", "")})
    write(pd.DataFrame(rows), "efficiency", "Efficiency (fold-0 weights; 10-s LUDB lead II at 250 Hz; latency median "
                                            "of 50; training memory = peak working-set increase, batch 32x1024)")


def tol50():
    p = RESULTS / "p_tol50.json"
    if not p.exists():
        return
    d = jl(p)
    rows = []
    for m in ["wavelet"] + ALL:
        if m not in d:
            continue
        row = {"model": NICE[m]}
        for part, lab in [("butpdb_each_lead", "BUT PDB"), ("butpdb_nooverlap_each_lead", "BUT PDB no-overlap"),
                          ("pwave_each_lead", "pwave")]:
            if part not in d[m]:
                continue
            v = d[m][part]
            try:
                c = stats.ci(m, part, "P_peak", suffix="_tol50")["F1"]
                row[f"{lab} F1 [95% CI]"] = f"{fmt(v['F1'])} [{fmt(c[0])}, {fmt(c[1])}]"
            except Exception:  # noqa: BLE001
                row[f"{lab} F1 [95% CI]"] = fmt(v["F1"])
            row[f"{lab} Se/PPV"] = f"{fmt(v['Se'], 1)}/{fmt(v['PPV'], 1)}"
        rows.append(row)
    write(pd.DataFrame(rows).fillna(""), "p_tol50", "P_peak at the strict ±50 ms tolerance (secondary; each lead)")
    bb, _ = best_baseline()
    tests = [stats.compare(m, bb, "butpdb_each_lead", "P_peak", suffix="_tol50") for m in PROP]
    tests += [stats.compare("wavelet", bb, "butpdb_each_lead", "P_peak", suffix="_tol50")]
    adj = stats.holm([t["p_boot"] for t in tests])
    write(pd.DataFrame([{"comparison": f"{NICE[t['A']]} vs {NICE[t['B']]}", "F1 A": fmt(t["F1_A"]),
                         "F1 B": fmt(t["F1_B"]),
                         "diff [95% CI]": f"{fmt(t['diff'])} [{fmt(t['diff_CI95'][0])}, {fmt(t['diff_CI95'][1])}]",
                         "p (Holm)": fmt(a, 4)} for t, a in zip(tests, adj)]),
          "p_tol50_tests", "BUT PDB P_peak at ±50 ms vs best baseline (exploratory, Holm)")


def main():
    cv_summary()
    tol50()
    cse_table()
    efficiency()
    wavelet_rows()
    point_table("ludb_test", ["P_on", "P_peak", "P_off", "QRS_on", "QRS_peak", "QRS_off", "T_on", "T_peak", "T_off"],
                "ludb_test", "LUDB internal test (41 subjects x 12 leads), 5-fold ensemble, ±150 ms", ALL)
    point_table("qtdb_each_lead", ["P_on", "P_peak", "P_off", "QRS_on", "QRS_off", "T_peak", "T_off"], "qtdb_each",
                "QTDB q1c, each lead pooled (primary policy), ±150 ms", ALL)
    point_table("qtdb_best_lead", ["P_on", "P_peak", "P_off", "QRS_on", "QRS_off", "T_peak", "T_off"], "qtdb_best",
                "QTDB q1c, best lead per record and point (literature policy; optimistic), ±150 ms", ALL)
    for part, nm in [("butpdb_each_lead", "BUT PDB"), ("butpdb_nooverlap_each_lead", "BUT PDB without 9 segments "
                                                                                       "overlapping pwave"),
                     ("pwave_each_lead", "MIT-BIH P-wave"), ("butpdb_best_lead", "BUT PDB best lead"),
                     ("pwave_best_lead", "MIT-BIH P-wave best lead")]:
        point_table(part, ["P_peak"], f"p_{part}", f"{nm}: P_peak, ±150 ms", ALL)
    primary_family()
    secondary_family()
    h1_h2()
    subgroups()
    print("tables written to", T)


if __name__ == "__main__":
    main()
