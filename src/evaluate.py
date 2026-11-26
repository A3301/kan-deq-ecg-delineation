"""Inference (sliding window) and evaluation on LUDB / QTDB / BUT PDB / pwave with frozen rules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import Accumulator, in_windows  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

import os  # noqa: E402

FS = int(os.environ.get("ECG_FS", 250))
TOL_MS = 150                      # primary detection tolerance (frozen; see docs/analiz_plani.md)
WIN, HOP = 2048 * FS // 250, 1024 * FS // 250
LUDB_POINTS = ["P_on", "P_peak", "P_off", "QRS_on", "QRS_peak", "QRS_off", "T_on", "T_peak", "T_off"]
QTDB_POINTS = ["P_on", "P_peak", "P_off", "QRS_on", "QRS_off", "T_peak", "T_off"]


@torch.no_grad()
def predict_proba(models, x: np.ndarray) -> np.ndarray:
    """models: list of nn.Module (ensemble, probabilities averaged). x: 1-D float32. Returns [4, n]."""
    n = len(x)
    L = max(16, int(np.ceil(min(n, WIN) / 16) * 16))
    if n <= WIN:
        xp = np.zeros(L, np.float32)
        xp[:n] = x
        t = torch.from_numpy(xp)[None, None]
        p = sum(torch.softmax(m(t), 1)[0] for m in models) / len(models)
        return p[:, :n].numpy()
    acc = np.zeros((4, n), np.float32)
    w = np.zeros(n, np.float32)
    taper = np.hanning(WIN).astype(np.float32) + 1e-3
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    batch = 16
    for b in range(0, len(starts), batch):
        ss = starts[b:b + batch]
        t = torch.from_numpy(np.stack([x[s:s + WIN] for s in ss]))[:, None]
        p = (sum(torch.softmax(m(t), 1) for m in models) / len(models)).numpy()
        for j, s in enumerate(ss):
            acc[:, s:s + WIN] += p[j] * taper
            w[s:s + WIN] += taper
    return acc / w


def ludb_eval(models, samples, tol_ms=TOL_MS) -> Accumulator:
    acc = Accumulator()
    tol = int(round(tol_ms * FS / 1000))
    for s in samples:
        prob = predict_proba(models, s.x)
        pred = mask_to_points(prob.argmax(0), s.x)
        s0, s1 = s.span
        for p in LUDB_POINTS:
            ref = s.points[p]
            ref = ref[(ref >= s0) & (ref < s1)]
            acc.add(p, s.subject, ref, pred[p], tol, fp_windows=[(s0, s1)])
    return acc


def qtdb_beat_windows(s) -> list[tuple[int, int]]:
    """Windows around each annotated beat: from (P_on or QRS_on - 250 ms) to (T_off or QRS_off + 450 ms)."""
    wins = []
    qrs = s.points["QRS_peak"]
    for q in qrs:
        lo = q - int(0.35 * FS)
        hi = q + int(0.55 * FS)
        wins.append((lo, hi))
    return wins


def qtdb_eval(models, samples, tol_ms=TOL_MS, lead_policy="best") -> dict:
    """lead_policy: 'each' = every lead separately; 'best' = per record and point, lead with lower mean |error|
    (standard practice in the QTDB literature, e.g. Martinez 2004); reported both."""
    tol = int(round(tol_ms * FS / 1000))
    per_lead = {}
    for s in samples:
        # inference restricted to the annotated region +- one window of context (evaluation never looks outside)
        c0 = max(0, s.span[0] - WIN)
        c1 = min(len(s.x), s.span[1] + WIN)
        xs = s.x[c0:c1]
        prob = predict_proba(models, xs)
        pred = {k: v + c0 for k, v in mask_to_points(prob.argmax(0), xs).items()}
        wins = qtdb_beat_windows(s)
        per_lead[(s.record, s.lead)] = (s, pred, wins)
    each = Accumulator()
    best = Accumulator()
    recs = sorted({k[0] for k in per_lead})
    for r in recs:
        leads = [v for k, v in per_lead.items() if k[0] == r]
        for p in QTDB_POINTS:
            cands = []
            for s, pred, wins in leads:
                ref = s.points[p]
                det = pred[p]
                each.add(p, s.subject, ref, det, tol, fp_windows=wins)
                tmp = Accumulator()
                tmp.add(p, s.subject, ref, det, tol, fp_windows=wins)
                e = np.abs(tmp.c[p]["err"]).mean() if tmp.c[p]["err"] else np.inf
                cands.append((e, ref, det, s.subject, wins))
            if cands and len(cands[0][1]):
                e, ref, det, subj, wins = min(cands, key=lambda c: c[0])
                best.add(p, subj, ref, det, tol, fp_windows=wins)
    return {"each_lead": each, "best_lead": best}


def p_only_eval(models, samples, tol_ms=TOL_MS, exclude_records=()) -> dict:
    """BUT PDB / pwave: P_peak only. Per lead and 'any lead' (a reference P counts as found if either lead finds
    it; predictions merged across leads by keeping the lead with more matches -> reported as best_lead)."""
    tol = int(round(tol_ms * FS / 1000))
    each = Accumulator()
    best = Accumulator()
    by_rec = {}
    for s in samples:
        if s.record in exclude_records:
            continue
        prob = predict_proba(models, s.x)
        pred = mask_to_points(prob.argmax(0), s.x)["P_peak"]
        s0, s1 = s.span
        det = pred
        ref = s.points["P_peak"]
        each.add("P_peak", s.subject, ref, det, tol, fp_windows=[(s0, s1)])
        by_rec.setdefault(s.record, []).append((s, ref, det))
    for r, lst in by_rec.items():
        scored = []
        for s, ref, det in lst:
            tmp = Accumulator()
            tmp.add("P_peak", s.subject, ref, det, tol, fp_windows=[s.span])
            t = tmp.table()["P_peak"]
            scored.append((t["F1"] if not np.isnan(t["F1"]) else -1, s, ref, det))
        _, s, ref, det = max(scored, key=lambda c: c[0])
        best.add("P_peak", s.subject, ref, det, tol, fp_windows=[s.span])
    return {"each_lead": each, "best_lead": best}
