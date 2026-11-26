# Post hoc: is the 15-40 Hz power fraction (H2c "noise" proxy) related to QRS width on the 41 LUDB test records?
import sys, types, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT / "src")); sys.modules.setdefault("wfdb", types.ModuleType("wfdb"))
import data, splits
w = pd.read_csv(ROOT / "results" / "deq_windows.csv"); w = w[w.db == "LUDB"]
rec = w.groupby("record").agg(iters=("iters", "mean"), hf=("hf_frac", "mean"), mae=("mae_ms", "mean")).reset_index()
test = set(splits.load()["test"]); rows = []
for s in data.get("ludb"):
    if s.subject not in test: continue
    on, off = s.points["QRS_on"], s.points["QRS_off"]
    d = []
    for a in on:
        b = off[off > a]
        if len(b) and (b[0] - a) * 1000 / data.FS < 250: d.append((b[0] - a) * 1000 / data.FS)
    # QRS share of the 15-40 Hz power: fraction of HF power inside QRS spans
    rows.append((int(s.record), s.lead, np.mean(d) if d else np.nan, np.ptp(s.x)))
q = pd.DataFrame(rows, columns=["record", "lead", "qrs_ms", "ptp"]).groupby("record").mean(numeric_only=True).reset_index()
rec["record"] = rec.record.astype(int); m = rec.merge(q, on="record")
print("n records", len(m))
for a, b in [("hf", "qrs_ms"), ("iters", "qrs_ms"), ("iters", "hf"), ("hf", "mae"), ("iters", "mae"), ("mae", "qrs_ms")]:
    r, p = spearmanr(m[a], m[b]); print(f"spearman {a} vs {b}: rho={r:.3f} p={p:.4f}")
# energy check on one representative: fraction of 15-40 Hz band power that falls inside QRS spans (all test leads)
from scipy.signal import butter, sosfiltfilt
sos = butter(4, [15, 40], btype="band", fs=data.FS, output="sos"); fr = []
for s in data.get("ludb"):
    if s.subject not in test: continue
    y = sosfiltfilt(sos, s.x); s0, s1 = s.span; mask = np.zeros(len(y), bool)
    lab = s.y == 2; e = (y ** 2)[s0:s1]; fr.append((e[lab[s0:s1]].sum() / e.sum(), lab[s0:s1].mean()))
fr = np.array(fr); print("median share of 15-40 Hz energy inside reference QRS spans: %.3f (QRS occupies %.3f of samples)" % tuple(np.median(fr, 0)))
