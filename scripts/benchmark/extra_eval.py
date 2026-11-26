"""evaluate ensembles (and the wavelet baseline) on the new external sets Zhejiang (zj) and ISP.

Global labels -> every lead is evaluated against the same reference and pooled ('each lead'); secondary: lead II only.
Points: P/QRS/T on/off (no peak labels). Evaluation region: first..last annotation. Matching rules as in the primary benchmark.
Usage: python src/extra_eval.py <model>[,<model>@<tag>,...|wavelet] [--threads 3]
Output: results/final_extra/<name>.json and <name>_records.pkl
"""
from __future__ import annotations

import argparse
import os
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import extra_data  # noqa: E402
from evaluate import TOL_MS, predict_proba  # noqa: E402
from final_eval import load_models, out_name  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

POINTS = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
OUT = RESULTS / "final_extra"


def predict_points(model, x):
    if model == "wavelet":
        from wavelet_baseline import delineate
        return delineate(x)
    return mask_to_points(predict_proba(model, x).argmax(0), x)


def evaluate(models, samples, tol_ms=TOL_MS, tol40=True):
    """Returns (each-lead, lead-II, each-lead at +-40 ms) accumulators; +-40 ms matches Lim et al. 2026."""
    tol = int(round(tol_ms * data.FS / 1000))
    t40 = int(round(40 * data.FS / 1000))
    each, lead2, each40 = Accumulator(), Accumulator(), Accumulator()
    if models == "wavelet" or type(models[0]).__name__ == "DEQSeg":
        # DEQ: the Anderson stopping rule acts on the whole batch, so the output depends on batch composition;
        # evaluate one lead at a time, exactly as in every other evaluation of this study
        preds = [predict_points(models, s.x) for s in samples]
    else:                                   # batched inference, identical output (src/batch_infer.py)
        from batch_infer import predict_many
        preds = []
        for r in range(0, len(samples), 12):
            chunk = samples[r:r + 12]
            preds += [mask_to_points(pr.argmax(0), s.x) for pr, s in zip(predict_many(models, [s.x for s in chunk]), chunk)]
    for s, pred in zip(samples, preds):
        s0, s1 = s.span
        if s1 - s0 <= 0:
            continue
        for q in POINTS:
            if q not in s.points:
                continue
            ref = s.points[q][(s.points[q] >= s0) & (s.points[q] < s1)]
            det = pred.get(q, np.zeros(0, int))
            each.add(q, s.subject, ref, det, tol, fp_windows=[(s0, s1)])
            each40.add(q, s.subject, ref, det, t40, fp_windows=[(s0, s1)])
            if s.lead == "ii":
                lead2.add(q, s.subject, ref, det, tol, fp_windows=[(s0, s1)])
    return each, lead2, each40


def pack(acc):
    return {"table": acc.table(), "subj": {p: v["subj"] for p, v in acc.c.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--sets", default="zj,isp")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {nm: extra_data.get(nm) for nm in a.sets.split(",")}
    for m in a.models.split(","):
        if (OUT / f"{out_name(m)}.json").exists():
            continue
        claim = OUT / f"{out_name(m)}.claim"
        try:                                   # several instances may share one model list
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        t0 = time.perf_counter()
        models = "wavelet" if m == "wavelet" else load_models(m)[0]
        res, recs = {"model": m}, {}
        for nm, smp in sets.items():
            each, lead2, each40 = evaluate(models, smp)
            res[f"{nm}_each_lead"], res[f"{nm}_lead_ii"] = each.table(), lead2.table()
            res[f"{nm}_each_lead_tol40"] = each40.table()
            recs[f"{nm}_each_lead"], recs[f"{nm}_lead_ii"] = pack(each), pack(lead2)
            recs[f"{nm}_each_lead_tol40"] = pack(each40)
        res["wall_s"] = time.perf_counter() - t0
        name = out_name(m)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        with open(OUT / f"{name}_records.pkl", "wb") as fh:
            pickle.dump(recs, fh)
        msg = " | ".join(f"{nm} P_on {res[f'{nm}_each_lead']['P_on']['F1']:.2f} "
                         f"mean6 {np.mean([res[f'{nm}_each_lead'][q]['F1'] for q in POINTS]):.2f}" for nm in sets)
        print(m, msg, f"| {res['wall_s']:.0f}s", flush=True)
        claim.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
