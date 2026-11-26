"""Calibrate mask-state meanings and test whether view clocks are identifiable without either model."""
from pathlib import Path
from itertools import permutations
import json
import numpy as np
import pandas as pd
if hasattr(pd.options.future,"infer_string"):pd.options.future.infer_string=False
import wfdb

ROOT=Path(__file__).resolve().parent
W=ROOT/"source"
OUT=ROOT/"audit"
OUT.mkdir(exist_ok=True)
WAVES=["p-wave peak","R peak","t-wave peak"]
RATES=[120,125,150,200,240,250,256,300,400,500]
MAPS=list(permutations(range(3)))
def episodes(mask):
    runs=[]
    starts=np.flatnonzero(np.r_[True,np.diff(mask)!=0])
    for i,j in zip(starts,np.r_[starts[1:],len(mask)]):
        if i>0 and j<len(mask) and mask[i]>0:runs.append((int(mask[i]),i,j-1))
    return {k:np.array([(s,e) for c,s,e in runs if c==k],dtype=float).reshape(-1,2) for k in [1,2,3]}

def annot(rid):
    name=W/"ecgdeli"/f"{rid:05d}_points_global"
    if not name.with_suffix(".atr").exists(): return None
    a=wfdb.rdann(str(name),"atr")
    assert a.fs==500
    return [a.sample[np.asarray(a.aux_note)==key]*2 for key in WAVES]

def make_records(split):
    roster=pd.read_csv(ROOT/"roster.csv");masks={}
    for gr in ["AF","others"]:
        m=np.load(W/(gr+".npy"),mmap_mode="r",allow_pickle=False)
        masks[gr]=m.reshape(-1,12,1200)
    out=[]
    for r in roster[roster.split.eq(split)].itertuples():
        ann=annot(r.ecg_id)
        if ann is None:continue
        out.append((r.ecg_id,r.patient_id,r.stratum,episodes(masks[r.stratum][r.array_position,1]),ann))
    return out

def coverage(r,rate,offsets):
    result=np.zeros((len(offsets),3,3))
    for idx,seg in r[3].items():
        if not len(seg):
            result[:,idx-1,:]=np.nan
            continue
        periods=seg[None,:,:]*1000/rate+offsets[:,None,None]
        for t,peaks in enumerate(r[4]):
            if not len(peaks):continue
            slots=np.searchsorted(peaks,periods[:,:,0],side="left")
            valid=slots<len(peaks)
            result[:,idx-1,t]=np.mean(valid & (peaks[np.minimum(slots,len(peaks)-1)]<=periods[:,:,1]),axis=1)
    return result

def profile(records):
    rows=[]
    for rate in RATES:
        last=10000-1200*1000/rate
        offs=np.unique(np.r_[np.arange(0,last+.0001,10),last/2,last])
        cube=np.stack([coverage(r,rate,offs) for r in records])
        for mapping in MAPS:
            tag="/".join(WAVES[j] for j in mapping)
            by_rec=np.nanmean(np.stack([cube[:,:,k,k2] for k,k2 in enumerate(mapping)],axis=2),axis=2)
            scores=np.nanmean(by_rec,axis=0)
            for offset,score in zip(offs,scores):
                rows.append((rate,offset,tag,len(records),score))
    return pd.DataFrame(rows,columns=["hz","offset_ms","states_1_2_3","records","equal_record_peak_containment"])

if __name__=="__main__":
    recs=make_records("calibration")
    official=json.loads((W/"official_retrieval.json").read_text())
    available=[x for x in official if x[0].endswith("_points_global.atr") and not x[1].startswith("not in")]
    print("available archive landmarks",len(available),"Calibration with marks",len(recs),"of",len(pd.read_csv(ROOT/"roster.csv").query("split=='calibration'")),flush=True)
    prof=profile(recs)
    prof.to_csv(OUT/"calibration_grid.csv.gz",index=False)
    summary=prof.sort_values("equal_record_peak_containment",ascending=False)
    summary.groupby("hz",sort=False).head(1).to_csv(OUT/"best_by_rate.csv",index=False)
    summary.groupby("states_1_2_3",sort=False).head(1).to_csv(OUT/"best_by_mapping.csv",index=False)
    print(summary.head(12).to_string(index=False),flush=True)
