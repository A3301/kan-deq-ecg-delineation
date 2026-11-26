"""Signal preparation and label construction for all four datasets (Phase 2).

Design (frozen in docs/analiz_plani.md):
- Common sampling rate FS = 250 Hz (polyphase resampling, scipy.signal.resample_poly).
- Filter: zero-phase Butterworth band-pass 0.5-40 Hz (order 2 per direction, SOS, sosfiltfilt).
- Normalisation: per lead-signal robust standardisation (median / (IQR/1.349)). Uses no statistics from other
  records, so no train/test leakage; global statistics would not transfer across databases with different units.
- Single-lead samples: every lead of every record is one sample; split is by subject (LUDB record = subject).
- Segmentation target: 0 = background, 1 = P, 2 = QRS, 3 = T; -1 = ignore (outside the annotated span of the lead,
  or inside an incomplete wave annotation).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotations import parse_waves  # noqa: E402
from paths import BUTPDB, LUDB, LUDB_DATA, LUDB_LEADS, PROCESSED, PWAVE, QTDB, records  # noqa: E402

import os  # noqa: E402

FS = int(os.environ.get("ECG_FS", 250))   # 250 Hz main analysis; 500 Hz only for ablation 9.7
BAND = (0.5, 40.0)
CLASS = {"P": 1, "QRS": 2, "T": 3}
POINTS = ["P_on", "P_peak", "P_off", "QRS_on", "QRS_peak", "QRS_off", "T_on", "T_peak", "T_off"]


@dataclass
class LeadSample:
    db: str
    record: str
    subject: str
    lead: str
    x: np.ndarray                       # float32, filtered + normalised, at FS
    y: np.ndarray | None                # int8 mask (LUDB) or None
    points: dict = field(default_factory=dict)   # name -> int array of sample indices at FS
    span: tuple[int, int] = (0, 0)      # evaluation span [start, end) at FS
    meta: dict = field(default_factory=dict)


_SOS = butter(2, BAND, btype="bandpass", fs=FS, output="sos")


def resample(x: np.ndarray, fs_in: float) -> np.ndarray:
    if fs_in == FS:
        return x
    fr = Fraction(FS, int(round(fs_in))).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0)


def to_fs(idx, fs_in: float) -> np.ndarray:
    return np.round(np.asarray(idx, dtype=float) * FS / fs_in).astype(int)


def clean(x: np.ndarray) -> np.ndarray:
    """x: 1-D signal already at FS. Band-pass then robust standardisation."""
    x = np.nan_to_num(x.astype(np.float64))
    x = sosfiltfilt(_SOS, x)
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.349
    if scale < 1e-8:
        scale = np.std(x) + 1e-8
    return ((x - med) / scale).astype(np.float32)


def waves_to_points(waves, fs_in: float) -> dict:
    pts = {p: [] for p in POINTS}
    for w in waves:
        if w.kind not in CLASS:
            continue
        if w.on is not None:
            pts[f"{w.kind}_on"].append(w.on)
        pts[f"{w.kind}_peak"].append(w.peak)
        if w.off is not None:
            pts[f"{w.kind}_off"].append(w.off)
    return {k: np.sort(to_fs(v, fs_in)) for k, v in pts.items()}


# ----------------------------------------------------------------------------- LUDB
def load_ludb() -> list[LeadSample]:
    out = []
    for r in [s.split("/")[-1] for s in records(LUDB)]:
        sig, f = wfdb.rdsamp(str(LUDB_DATA / r))
        fs_in = f["fs"]
        comments = f["comments"]
        rhythm = next((c.split(":", 1)[1].strip().rstrip(".") for c in comments if c.startswith("Rhythm")), "")
        sig = resample(sig, fs_in)
        n = sig.shape[0]
        for li, lead in enumerate(LUDB_LEADS):
            a = wfdb.rdann(str(LUDB_DATA / r), lead)
            waves = parse_waves(a.sample, list(a.symbol))
            y = np.full(n, -1, dtype=np.int8)
            if len(a.sample):
                s0 = int(to_fs([a.sample.min()], fs_in)[0])
                s1 = int(to_fs([a.sample.max()], fs_in)[0]) + 1
                y[s0:s1] = 0
            else:
                s0 = s1 = 0
            for w in waves:
                if w.kind not in CLASS:
                    continue
                if w.on is None or w.off is None:  # incomplete annotation -> ignore its neighbourhood
                    c = int(to_fs([w.peak], fs_in)[0])
                    h = int(0.1 * FS)
                    y[max(c - h, 0):c + h] = -1
                    continue
                on, off = to_fs([w.on, w.off], fs_in)
                y[on:off + 1] = CLASS[w.kind]
            out.append(LeadSample("LUDB", r, r, lead, clean(sig[:, li]), y, waves_to_points(waves, fs_in),
                                  (s0, s1), {"rhythm": rhythm}))
    return out


# ----------------------------------------------------------------------------- QTDB
def load_qtdb(annotator: str = "q1c") -> list[LeadSample]:
    out = []
    for r in records(QTDB):
        sig, f = wfdb.rdsamp(str(QTDB / r))
        fs_in = f["fs"]
        a = wfdb.rdann(str(QTDB / r), annotator)
        waves = parse_waves(a.sample, list(a.symbol))
        pts = waves_to_points(waves, fs_in)
        # Evaluation span: from 200 ms before the first to 400 ms after the last annotation.
        s0 = max(int(to_fs([a.sample.min()], fs_in)[0]) - 50, 0)
        s1 = int(to_fs([a.sample.max()], fs_in)[0]) + 100
        sig = resample(sig, fs_in)
        for li, lead in enumerate(f["sig_name"]):
            out.append(LeadSample("QTDB", r, r, lead, clean(sig[:, li]), None, pts, (s0, s1),
                                  {"annotator": annotator, "beats": [w.peak for w in waves if w.kind == "QRS"]}))
    return out


# ----------------------------------------------------------------------------- BUT PDB / pwave
def _p_only(db_name, db_dir, recs, pext, qext=None, meta_fn=None) -> list[LeadSample]:
    out = []
    for r in recs:
        sig, f = wfdb.rdsamp(str(db_dir / r))
        fs_in = f["fs"]
        p = wfdb.rdann(str(db_dir / r), pext)
        pts = {"P_peak": np.sort(to_fs(p.sample, fs_in))}
        if qext:
            q = wfdb.rdann(str(db_dir / r), qext)
            pts["QRS_ref"] = np.sort(to_fs(q.sample, fs_in))
            pts["QRS_ref_symbols"] = np.array(q.symbol)[np.argsort(q.sample)]
        sig = resample(sig, fs_in)
        n = sig.shape[0]
        s0 = max(int(pts["P_peak"].min()) - FS // 2, 0) if len(pts["P_peak"]) else 0
        s1 = min(int(pts["P_peak"].max()) + FS // 2, n) if len(pts["P_peak"]) else n
        meta = meta_fn(r) if meta_fn else {}
        for li in range(sig.shape[1]):
            lead = f["sig_name"][li] or f"ch{li}"
            out.append(LeadSample(db_name, r, meta.get("subject", r), lead, clean(sig[:, li]), None, pts, (s0, s1),
                                  {**meta, "fs_orig": fs_in}))
    return out


def load_butpdb() -> list[LeadSample]:
    import pandas as pd
    from eda import parse_butpdb_readme  # README table: pathology, source db/record
    meta = parse_butpdb_readme().set_index("record")

    def m(r):
        row = meta.loc[r]
        return {"pathology": row.pathology, "source_db": row.source_db, "source_record": row.source_record,
                "seg_start": int(row.seg_start), "seg_end": int(row.seg_end),
                "subject": f"{row.source_db}:{row.source_record}"}
    recs = [f"{int(r):02d}" for r in records(BUTPDB)]
    _ = pd  # keep import local
    return _p_only("BUTPDB", BUTPDB, recs, "pwave", "qrs", m)


def load_pwave() -> list[LeadSample]:
    def m(r):
        return {"subject": f"MIT-BIH Arrhythmia Database:{r}",
                "header": " ".join(wfdb.rdheader(str(PWAVE / r)).comments)}
    return _p_only("PWAVE", PWAVE, records(PWAVE), "pwave", None, m)


LOADERS = {"ludb": load_ludb, "qtdb": load_qtdb, "butpdb": load_butpdb, "pwave": load_pwave}


def get(name: str, rebuild: bool = False) -> list[LeadSample]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}_fs{FS}.pkl"
    if path.exists() and not rebuild:
        class _U(pickle.Unpickler):  # files written when data.py ran as __main__
            def find_class(self, module, nm):
                if nm == "LeadSample":
                    return LeadSample
                return super().find_class(module, nm)
        with open(path, "rb") as fh:
            return _U(fh).load()
    data = LOADERS[name]()
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return data


if __name__ == "__main__":
    for nm in LOADERS:
        d = get(nm, rebuild=True)
        print(nm, len(d), "lead-samples; example len", len(d[0].x), "span", d[0].span,
              {k: len(v) for k, v in d[0].points.items() if hasattr(v, "__len__")})
