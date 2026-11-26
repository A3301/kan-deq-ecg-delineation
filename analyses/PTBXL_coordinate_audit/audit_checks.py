"""Sanity checks restricted to calibration IDs; no predicted study landmarks."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from coordinate_audit import make_records,coverage,ROOT,OUT
import wfdb

v=make_records("calibration")
spacings=[];energy=[]
for rid,patient,stratum,eps,anchors in v:
    rr=np.median(np.diff(anchors[1])) if len(anchors[1])>=4 else np.nan
    for code,pairs in eps.items():
        if len(pairs)>=4:
            s=np.median(np.diff(pairs.mean(axis=1)))
            spacings.append((rid,stratum,code,s,rr))
    sig,info=wfdb.rdsamp(str(ROOT/"source"/"ptbxl"/f"{rid:05d}_hr"))
    assert info["fs"]==500 and sig.shape==(5000,12)
    d=np.diff(sig,axis=0)
    positions=(anchors[1]/2).astype(int)
    # Native lead-II ECG and multisignal slope, no HF label used. Fixed window around
    # each ECGDeli R peak versus a same-length window 160 ms earlier.
    positions=positions[(positions>=101)&(positions<len(d)-21)]
    if len(positions):
        window=np.arange(-20,21)
        pos=np.median([np.max(np.abs(d[k+window,1])) for k in positions])
        base=np.median([np.max(np.abs(d[k-80+window,1])) for k in positions])
        energy.append((rid,patient,len(positions),pos,base,bool(pos>base)))
t=pd.DataFrame(spacings,columns=["record","stratum","code","mask_spacing_in_cells","automated_median_RR_ms"])
t.to_csv(OUT/"calibration_spacing.csv",index=False)
rows={}
for k,g in t.dropna().groupby("code"):
    rows[int(k)]=dict(n=len(g),spearman=float(spearmanr(g.mask_spacing_in_cells,g.automated_median_RR_ms).statistic))
eg=pd.DataFrame(energy,columns=["record","patient","n_R_peaks","median_peak_window_slope","median_control_window_slope","peak_window_greater"])
eg.to_csv(OUT/"waveform_anchor_check.csv",index=False)
# Exercise each of the nine matrix cells and the fixed time-axis conversion on a 
# deterministic synthetic set with narrow source labels and known absolute positions.
truth={1:np.array([[40,43],[285,289],[493,497],[702,706]],float),
       2:np.array([[14,17],[252,255],[460,464]],float),
       3:np.array([[106,109],[359,362],[563,567]],float)}
perm=[1,0,2];peaks=[None]*3
for code,j in enumerate(perm,1):peaks[j]=2400+truth[code].mean(1)*4
z=coverage((0,0,"synthetic",truth,peaks),250,np.array([2300,2400,2500]))
assert np.all(z[1,[0,1,2],perm]==1)
assert np.all(z[1].sum(axis=1)==1)
assert np.all(z[[0,2]][:,[0,1,2],perm]==0)
checks=dict(calibration_records=len(v),code_repeat_spacing_vs_known_RR=rows,
        raw_records_with_peak_window_steeper=int(eg.peak_window_greater.sum()),raw_records_checked=len(eg),raw_R_peaks_checked=int(eg.n_R_peaks.sum()),
        synthetic_fixture_known_class_grid_offset_recovered=True,
        no_RDB_models_loaded_or_scored=True)
(OUT/"sanity_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf8")
print(checks)
