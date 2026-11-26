"""Independent nearest-pair replay from the published native P annotations and saved peaks."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def recount(r,d,tol,start,end):
    pairs=[]
    for i,t in enumerate(r):
        a=np.searchsorted(d,t-tol,side="left")
        b=np.searchsorted(d,t+tol,side="right")
        pairs.extend((abs(int(t)-int(d[j])),i,j) for j in range(a,b))
    rr,dd=set(),set()
    for _,i,j in sorted(pairs):
        if i not in rr and j not in dd:
            rr.add(i)
            dd.add(j)
    return len(rr),len(r)-len(rr),sum(start<=p<end and j not in dd for j,p in enumerate(d))

def main():
    out=ROOT/"results"
    refs=pd.read_csv(ROOT/"source"/"references.csv",dtype={"segment":str},keep_default_na=False).set_index("segment")
    seen={}
    with (out/"events.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            ix=(r["model"],r["seed"],r["segment"],r["lead"],r["coordinate"],int(r["tol_ms"]))
            assert ix not in seen
            seen[ix]=(int(r["tp"]),int(r["fn"]),int(r["fp"]))
    counts=Counter()
    with (out/"predictions.csv").open(newline="") as f:
        for prediction in csv.DictReader(f):
            rid=prediction["segment"]
            reference=refs.loc[rid]
            native=np.array([int(x) for x in reference.native_ref.split()],dtype=int)
            peaks=np.array([int(x) for x in prediction["peaks_250Hz"].split()],dtype=int)
            fs=int(reference.native_hz)
            assert reference.source==prediction["source"]
            for method in ["native_ms","historical_grid"]:
                for tol in [150,50]:
                    if method=="native_ms":
                        r=native*1000
                        d=peaks*4*fs
                        epsilon=tol*fs
                        fringe=500*fs
                        stop=int(reference.samples_250Hz)*4*fs
                    else:
                        r=np.array([round(250*t/fs) for t in native],dtype=int)
                        d=peaks
                        epsilon=round(tol/4)
                        fringe=125
                        stop=reference.samples_250Hz
                    lo=max(r[0]-fringe,0) if len(r) else 0
                    hi=min(r[-1]+fringe,stop) if len(r) else stop
                    ix=(prediction["model"],prediction["seed"],rid,prediction["lead"],method,tol)
                    assert seen.pop(ix)==recount(r,d,epsilon,lo,hi),ix
                    counts[(method,tol)]+=1
    assert set(counts.values())=={1200} and not seen
    print("Independently verified all 4,800 score cells from the saved 1,200 lead prediction lists.")

if __name__=="__main__":
    main()
