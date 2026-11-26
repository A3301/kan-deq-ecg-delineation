"""Faz 3 baseline 1: Martinez-type wavelet delineator (NeuroKit2 ecg_delineate, method='dwt'). No training.

Run on LUDB internal test + all CV subjects (no tuning), QTDB (each lead), BUT PDB, pwave.
Evaluation rules identical to src/evaluate.py (incl. the evaluation rule of 2026-11-23: matching with all detections,
FP only inside the evaluation region). QTDB delineation restricted to the annotated region +- 8.2 s like the DL models.
Outputs results/baselines/wavelet_neurokit_dwt.json and results/final/wavelet_records.pkl (per-subject counts).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import splits  # noqa: E402
from evaluate import LUDB_POINTS, QTDB_POINTS, TOL_MS, WIN, qtdb_beat_windows  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402

warnings.filterwarnings("ignore")
FS = data.FS
MAP = {"P_on": "ECG_P_Onsets", "P_peak": "ECG_P_Peaks", "P_off": "ECG_P_Offsets", "QRS_on": "ECG_R_Onsets",
       "QRS_off": "ECG_R_Offsets", "T_on": "ECG_T_Onsets", "T_peak": "ECG_T_Peaks", "T_off": "ECG_T_Offsets"}
OVERLAP_BUT = {"01", "02", "09", "13", "16", "27", "33", "36", "38"}


def delineate(x: np.ndarray) -> dict:
    out = {k: np.zeros(0, int) for k in LUDB_POINTS}
    try:
        _, info = nk.ecg_peaks(x, sampling_rate=FS, method="neurokit")
        r = np.asarray(info["ECG_R_Peaks"], int)
        out["QRS_peak"] = r
        if len(r) < 2:
            return out
        _, waves = nk.ecg_delineate(x, r, sampling_rate=FS, method="dwt")
        for k, col in MAP.items():
            v = np.asarray(waves.get(col, []), float)
            out[k] = np.sort(v[~np.isnan(v)].astype(int))
    except Exception as e:  # noqa: BLE001
        out["_error"] = str(e)
    return out


def pack(acc):
    return {"table": acc.table(), "subj": {p: v["subj"] for p, v in acc.c.items()}}


def main():
    tol = int(round(TOL_MS * FS / 1000))
    out_dir = RESULTS / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final").mkdir(parents=True, exist_ok=True)
    sp = splits.load()
    res, recs = {}, {}
    t0 = time.perf_counter()
    ludb = data.get("ludb")
    for part, subj in [("ludb_test", set(sp["test"])), ("ludb_cv", {k for f in sp["folds"] for k in f})]:
        acc, n_err = Accumulator(), 0
        for s in ludb:
            if s.subject not in subj:
                continue
            pred = delineate(s.x)
            n_err += "_error" in pred
            s0, s1 = s.span
            for p in LUDB_POINTS:
                ref = s.points[p][(s.points[p] >= s0) & (s.points[p] < s1)]
                acc.add(p, s.subject, ref, pred[p], tol, fp_windows=[(s0, s1)])
        res[part] = {"table": acc.table(), "n_errors": n_err}
        recs[part] = pack(acc)
        print(part, json.dumps({k: round(v["F1"], 2) for k, v in res[part]["table"].items()}), flush=True)
    acc = Accumulator()
    for s in data.get("qtdb"):
        c0, c1 = max(0, s.span[0] - WIN), min(len(s.x), s.span[1] + WIN)
        pred = {k: v + c0 for k, v in delineate(s.x[c0:c1]).items() if not k.startswith("_")}
        wins = qtdb_beat_windows(s)
        for p in QTDB_POINTS:
            acc.add(p, s.subject, s.points[p], pred[p], tol, fp_windows=wins)
    res["qtdb_each_lead"] = {"table": acc.table()}
    recs["qtdb_each_lead"] = pack(acc)
    print("qtdb", json.dumps({k: round(v["F1"], 2) for k, v in res["qtdb_each_lead"]["table"].items()}), flush=True)
    for nm in ["butpdb", "pwave"]:
        acc, acc_no = Accumulator(), Accumulator()
        for s in data.get(nm):
            pred = delineate(s.x)["P_peak"]
            acc.add("P_peak", s.subject, s.points["P_peak"], pred, tol, fp_windows=[s.span])
            if nm == "butpdb" and s.record not in OVERLAP_BUT:
                acc_no.add("P_peak", s.subject, s.points["P_peak"], pred, tol, fp_windows=[s.span])
        res[f"{nm}_each_lead"] = {"table": acc.table()}
        recs[f"{nm}_each_lead"] = pack(acc)
        if nm == "butpdb":
            res["butpdb_nooverlap_each_lead"] = {"table": acc_no.table()}
            recs["butpdb_nooverlap_each_lead"] = pack(acc_no)
        print(nm, json.dumps({k: round(v["F1"], 2) for k, v in res[f"{nm}_each_lead"]["table"].items()}), flush=True)
    res["wall_s"] = time.perf_counter() - t0
    res["neurokit2_version"] = nk.__version__
    (out_dir / "wavelet_neurokit_dwt.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    with open(RESULTS / "final" / "wavelet_records.pkl", "wb") as fh:
        pickle.dump(recs, fh)


if __name__ == "__main__":
    main()
