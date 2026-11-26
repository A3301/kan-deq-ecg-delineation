"""Fiducial-point matching and metrics (Se, PPV, F1, mean error +- SD in ms).

Matching: for one fiducial type, reference points R and predicted points D inside the evaluation span.
Greedy one-to-one matching by increasing |d - r| with |d - r| <= tol. Matched pairs are TP (error = d - r),
unmatched references are FN, unmatched predictions are FP.

Evaluation-span rules per database (frozen in docs/analiz_plani.md):
- LUDB: [first annotation, last annotation] of the lead (edges are unannotated).
- QTDB: only beats annotated in q1c are references. A prediction is counted as FP only if it falls within
  the annotated-beat windows (from 200 ms before the annotated QRS peak... see beat_windows()).
- BUT PDB / pwave: P_peak only; span = [first P - 0.5 s, last P + 0.5 s].
"""
from __future__ import annotations

import os

import numpy as np

FS = int(os.environ.get("ECG_FS", 250))


def match(ref: np.ndarray, det: np.ndarray, tol: int):
    ref = np.asarray(ref, dtype=int)
    det = np.asarray(det, dtype=int)
    if len(ref) == 0 or len(det) == 0:
        return np.zeros(0, int), np.zeros(0, int), len(ref), len(det)
    d = np.abs(det[None, :] - ref[:, None])
    ii, jj = np.nonzero(d <= tol)
    order = np.argsort(d[ii, jj], kind="stable")
    used_r, used_d, pairs = set(), set(), []
    for k in order:
        i, j = ii[k], jj[k]
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        pairs.append((i, j))
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), len(ref), len(det)
    pr = np.array([p[0] for p in pairs])
    pd_ = np.array([p[1] for p in pairs])
    return ref[pr], det[pd_], len(ref) - len(pairs), len(det) - len(pairs)


def in_windows(p: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    p = np.asarray(p, dtype=int)
    if not windows:
        return np.zeros(len(p), bool)
    keep = np.zeros(len(p), bool)
    for s, e in windows:
        keep |= (p >= s) & (p < e)
    return keep


class Accumulator:
    """Accumulates TP/FN/FP and signed errors per fiducial point; also per-subject for paired statistics."""

    def __init__(self):
        self.c = {}

    def add(self, point: str, subject: str, ref, det, tol: int, fp_windows=None):
        """ref: references inside the evaluation region. det: ALL detections of the record (or a superset of the
        region). Matching uses every detection; an unmatched detection is a FP only if it lies inside fp_windows
        (list of [s, e)); if fp_windows is None every unmatched detection is a FP.
        (Evaluation rule dated 2026-11-23: avoids discarding correct detections that fall a few samples outside the
        region boundary, e.g. the first QRS onset of a LUDB lead.)"""
        det = np.asarray(det, dtype=int)
        r, d, fn, fp = match(ref, det, tol)
        if fp_windows is not None:
            un = np.setdiff1d(det, d, assume_unique=False)
            fp = int(in_windows(un, fp_windows).sum())
        e = (d - r) * 1000.0 / FS
        rec = self.c.setdefault(point, {"tp": 0, "fn": 0, "fp": 0, "err": [], "subj": {}})
        rec["tp"] += len(r)
        rec["fn"] += fn
        rec["fp"] += fp
        rec["err"].extend(e.tolist())
        s = rec["subj"].setdefault(subject, {"tp": 0, "fn": 0, "fp": 0, "err": []})
        s["tp"] += len(r)
        s["fn"] += fn
        s["fp"] += fp
        s["err"].extend(e.tolist())

    @staticmethod
    def summarize(rec) -> dict:
        tp, fn, fp = rec["tp"], rec["fn"], rec["fp"]
        se = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * se * ppv / (se + ppv) if se + ppv and not np.isnan(se + ppv) else np.nan
        err = np.asarray(rec["err"])
        return {"TP": tp, "FN": fn, "FP": fp, "Se": 100 * se, "PPV": 100 * ppv, "F1": 100 * f1,
                "mean_ms": float(err.mean()) if len(err) else np.nan,
                "sd_ms": float(err.std(ddof=1)) if len(err) > 1 else np.nan,
                "mae_ms": float(np.abs(err).mean()) if len(err) else np.nan}

    def table(self) -> dict:
        return {k: self.summarize(v) for k, v in self.c.items()}
