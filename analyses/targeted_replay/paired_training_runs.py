"""Paired, record-level sensitivity to the two observed training runs.

Run with python paired_training_runs.py COUNT_CSV OUTPUT_CSV. The four models
must share the exact same records and reference totals within each endpoint.
Resample a record once across both models, both runs and all selected landmarks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
ENDPOINTS = [
    ("LUDB internal", "ludb_test", SIX), ("LUDB P onset", "ludb_test", ["P_on"]),
    ("QTDB P onset", "qtdb_each_lead", ["P_on"]),
    ("BUT P peak, restricted", "butpdb_nooverlap_each_lead", ["P_peak"]),
    ("MIT-BIH P peak", "pwave_each_lead", ["P_peak"]),
]
def replay(data):
    results = []
    for label, domain, landmarks in ENDPOINTS:
        part = data[data.dataset.eq(domain) & data.landmark.isin(landmarks)]
        keys = sorted(part.record.unique())
        ref = pd.MultiIndex.from_product([keys, landmarks], names=["record","landmark"])
        models = []
        targets = []
        for seed in (1, 2):
            for head in ("KAN", "GELU"):
                rows = part[part.seed.eq(seed) & part["head"].eq(head)].set_index(["record","landmark"])
                if rows.index.has_duplicates or set(ref)!=set(rows.index):
                    raise ValueError(f"Unpaired {domain}/{head}/{seed}")
                a = rows.loc[ref,["tp","fn","fp"]].to_numpy().reshape(len(keys),len(landmarks),3)
                if (a < 0).any(): raise ValueError("Negative counts")
                targets.append(a[:,:,0]+a[:,:,1])
                models.append(a)
        if not all(np.array_equal(targets[0],t) for t in targets[1:]):
            raise ValueError("Reference events differ between runs")
        counts = np.stack(models,axis=1)
        rng = np.random.default_rng(20261125)
        weights = rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=10000)
        weights = np.vstack([np.ones(len(keys),dtype=int),weights])
        aggregated = np.einsum("br,rmkc->bmkc",weights,counts,optimize=True)
        tp,fn,fp = np.moveaxis(aggregated,-1,0)
        denom=2*tp+fn+fp
        f1=np.divide(200.*tp,denom,out=np.full_like(tp,np.nan,dtype=float),where=denom!=0).mean(axis=-1)
        d1,d2=f1[:,0]-f1[:,1],f1[:,2]-f1[:,3]
        for metric,vals in [("run1",d1),("run2",d2),("mean_of_runs",(d1+d2)/2),("change_in_head_contrast",d2-d1)]:
            boot=vals[1:]
            if not np.isfinite(boot).all(): raise ValueError("Undefined bootstrap replicate")
            lo,hi=np.quantile(boot,[.025,.975])
            results.append(dict(endpoint=label,clusters=len(keys),metric=metric,estimate=vals[0],lower=lo,upper=hi))
    return pd.DataFrame(results)
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("counts",type=Path); ap.add_argument("output",type=Path)
    args=ap.parse_args()
    results=replay(pd.read_csv(args.counts,dtype={"record":str}))
    results.to_csv(args.output,index=False,float_format="%.8f")
    print(results.round(3).to_string(index=False))
