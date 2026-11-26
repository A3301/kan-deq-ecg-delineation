"""Mask -> fiducial points. Identical for every model (frozen in docs/analiz_plani.md).

Rules (FS = 250 Hz):
1. Take argmax class per sample (0 bg, 1 P, 2 QRS, 3 T).
2. Find contiguous runs per class. Merge runs of the same class separated by a gap < 20 ms (5 samples).
3. Drop runs shorter than the minimum duration: P 40 ms, QRS 40 ms, T 60 ms.
4. Onset = first sample, offset = last sample of the run.
5. Peak: P and T = sample of maximum absolute deviation of the input signal from the straight line joining the
   run's onset and offset values (robust to baseline offset; works for inverted waves).
   QRS = sample of maximum (most positive) signal value, i.e. the R peak, matching the LUDB annotation
   convention (chosen on ground-truth masks only: mean |error| 6.2 ms vs 13.8 ms for the deviation rule;
   results/oracle_check_ludb.json, docs/faz_2_rapor.md).
No ordering (P->QRS->T) constraint is imposed, because AV block / AF / ectopy break that order; this is a
deliberate choice for the pathological-P question (H3).
"""
from __future__ import annotations

import numpy as np

import os  # noqa: E402

FS = int(os.environ.get("ECG_FS", 250))
MERGE_GAP = int(round(0.020 * FS))                                              # 5 samples at 250 Hz
MIN_LEN = {1: int(round(0.040 * FS)), 2: int(round(0.040 * FS)), 3: int(round(0.060 * FS))}  # 10/10/15
NAMES = {1: "P", 2: "QRS", 3: "T"}


def runs(mask: np.ndarray, cls: int) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask == cls, [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def merge(rs, gap=MERGE_GAP):
    out = []
    for s, e in rs:
        if out and s - out[-1][1] - 1 < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def peak_of(x: np.ndarray, s: int, e: int) -> int:
    seg = x[s:e + 1]
    if len(seg) < 3:
        return s + len(seg) // 2
    base = np.linspace(seg[0], seg[-1], len(seg))
    return s + int(np.argmax(np.abs(seg - base)))


def mask_to_points(mask: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    pts = {}
    for c, nm in NAMES.items():
        rs = [r for r in merge(runs(mask, c)) if r[1] - r[0] + 1 >= MIN_LEN[c]]
        pts[f"{nm}_on"] = np.array([s for s, _ in rs], dtype=int)
        pts[f"{nm}_off"] = np.array([e for _, e in rs], dtype=int)
        if c == 2:
            pts[f"{nm}_peak"] = np.array([s + int(np.argmax(x[s:e + 1])) for s, e in rs], dtype=int)
        else:
            pts[f"{nm}_peak"] = np.array([peak_of(x, s, e) for s, e in rs], dtype=int)
    return pts
