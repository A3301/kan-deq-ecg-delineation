"""Phase 5 statistics (frozen plan, section 8).

- 95% CI for pooled Se/PPV/F1 and mean |error| by subject-level bootstrap (10 000 resamples).
- Paired comparison A vs B: subject-level paired bootstrap of the F1 difference (two-sided p), plus Wilcoxon
  signed-rank on per-subject F1. Holm correction within a family.
Subjects: LUDB subject, QTDB record, BUT PDB/pwave source MIT-BIH/LTAF/SVDB record (BUT segments from the same
source record are one subject).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
B = 10_000


def load_records(model, suffix=""):
    with open(FINAL / f"{model}{suffix}_records.pkl", "rb") as fh:
        return pickle.load(fh)


def subj_arrays(subj: dict, keys):
    tp = np.array([subj[k]["tp"] if k in subj else 0 for k in keys], float)
    fn = np.array([subj[k]["fn"] if k in subj else 0 for k in keys], float)
    fp = np.array([subj[k]["fp"] if k in subj else 0 for k in keys], float)
    ae_sum = np.array([np.abs(subj[k]["err"]).sum() if k in subj else 0 for k in keys], float)
    return tp, fn, fp, ae_sum


def f1(tp, fn, fp):
    d = 2 * tp + fn + fp
    return np.where(d > 0, 200 * tp / np.maximum(d, 1e-12), np.nan)


def ci(model, part, point, rng=None, suffix=""):
    rng = rng or np.random.default_rng(0)
    subj = load_records(model, suffix)[part]["subj"][point]
    keys = sorted(subj)
    tp, fn, fp, ae = subj_arrays(subj, keys)
    idx = rng.integers(0, len(keys), (B, len(keys)))
    T, N, P, A = tp[idx].sum(1), fn[idx].sum(1), fp[idx].sum(1), ae[idx].sum(1)
    out = {}
    for nm, v in [("Se", 100 * T / np.maximum(T + N, 1e-12)), ("PPV", 100 * T / np.maximum(T + P, 1e-12)),
                  ("F1", f1(T, N, P)), ("mae_ms", A / np.maximum(T, 1e-12))]:
        out[nm] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return out


def compare(model_a, model_b, part, point, rng=None, suffix=""):
    """Returns dict with F1_a, F1_b, diff, 95% CI, bootstrap p, Wilcoxon p (per-subject F1), n_subjects."""
    rng = rng or np.random.default_rng(0)
    sa = load_records(model_a, suffix)[part]["subj"][point]
    sb = load_records(model_b, suffix)[part]["subj"][point]
    keys = sorted(set(sa) | set(sb))
    ta, na, pa, _ = subj_arrays(sa, keys)
    tb, nb, pb, _ = subj_arrays(sb, keys)
    fa, fb = f1(ta.sum(), na.sum(), pa.sum()), f1(tb.sum(), nb.sum(), pb.sum())
    idx = rng.integers(0, len(keys), (B, len(keys)))
    d = f1(ta[idx].sum(1), na[idx].sum(1), pa[idx].sum(1)) - f1(tb[idx].sum(1), nb[idx].sum(1), pb[idx].sum(1))
    p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    ra, rb = f1(ta, na, pa), f1(tb, nb, pb)
    ok = ~np.isnan(ra) & ~np.isnan(rb)
    try:
        p_w = float(wilcoxon(ra[ok], rb[ok], zero_method="wilcox").pvalue) if ok.sum() > 5 and np.any(ra[ok] != rb[ok]) else 1.0
    except ValueError:
        p_w = 1.0
    return {"A": model_a, "B": model_b, "part": part, "point": point, "F1_A": float(fa), "F1_B": float(fb),
            "diff": float(fa - fb), "diff_CI95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "p_boot": p_boot, "p_wilcoxon": p_w, "n_subjects": len(keys)}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
