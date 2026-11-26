"""Independent audit of 48 raw reference and predicted masks, plus published source checks."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import itertools
from zipfile import ZipFile
import numpy as np
import pandas as pd


POINTS = ["P_on","P_off","QRS_on","QRS_off","T_on","T_off"]


def raw_reference(z,rid):
    references = [set() for p in POINTS]
    content = z.read(f"ann_txt/{rid}.ii.txt").decode()
    for row in csv.DictReader(io.StringIO(content)):
        cls,start,end = int(row["TYPE"]),float(row["START"]),float(row["END"])
        if not (0 <= start < end < 5000) or cls not in [0,1,2]:
            continue
        if start>0: references[2*cls].add(start*2)
        if end<4999: references[2*cls+1].add(end*2)
    return [sorted(p) for p in references]


def raw_prediction(mask):
    found = {1: [],2: [],3: []}
    index = 0
    for value, group in itertools.groupby(mask.tolist()):
        n = sum(1 for _ in group)
        if value:
            waves = found[value]
            if waves and index-waves[-1][1]-1<5:
                waves[-1]=(waves[-1][0], index+n-1)
            else:
                waves.append((index,index+n-1))
        index += n
    out = []
    for cls, minimum in enumerate([10,10,15],start=1):
        waves=[(s,e) for s,e in found[cls] if e-s+1>=minimum]
        out.extend([[s*4 for s,e in waves if s>0],[e*4 for s,e in waves if e<2499]])
    return out


def measure(reference, detections, tolerance):
    pairs = [(abs(x-y),i,j) for i,x in enumerate(reference) for j,y in enumerate(detections) if abs(x-y)<=tolerance]
    rows, cols = set(), set()
    for d,i,j in sorted(pairs):
        if i not in rows and j not in cols:
            rows.add(i)
            cols.add(j)
    return len(rows),len(detections)-len(rows),len(reference)-len(rows)


def verify(args):
    ids=json.loads((args.out/"signal_order.json").read_text())
    meta=pd.read_csv(args.out/"cohort.csv").query("status=='included'").set_index("record")
    # Six records/rhythm, fixed by ID hash independent of predictions.
    sample = [rid for rhythm,sub in meta.groupby("rhythm") for rid in sorted(sub.index,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:6]]
    scores1=pd.read_csv(args.out/"record_scores.csv")
    scores2=pd.read_csv(args.out/"stage1_record_scores.csv")
    scores=pd.concat([scores1,scores2]).set_index(["record","arm","seed","tolerance_ms","endpoint"])
    masks = {(arm,seed): np.load(args.out/f"masks_{arm}_{seed}.npy",mmap_mode="r") for arm in ["KAN","GELU","UNetKAN","UNet"] for seed in [1,2]}
    rows_compared=0
    with ZipFile(args.rdb) as z:
        same_signal = z.read("dat_csv/SI0109.csv")==z.read("dat_csv/SI0211.csv")
        same_labels = z.read("ann_txt/SI0109.ii.txt")==z.read("ann_txt/SI0211.ii.txt")
        assert same_signal and not same_labels
        for rid in sample:
            ref = raw_reference(z,rid)
            bounds = list(itertools.chain.from_iterable(ref))
            low,high=max(0,min(bounds)-150),min(9998,max(bounds)+150)
            idx=ids.index(rid)
            for (arm,seed), arr in masks.items():
                det = raw_prediction(arr[idx])
                for point,x,y in zip(POINTS,ref,det,strict=True):
                    filtered=[v for v in y if low<=v<=high]
                    for tol in (40,150):
                        counts=measure(x,filtered,tol)
                        actual=tuple(scores.loc[(rid,arm,seed,tol,point),["tp","fp","fn"]].astype(int))
                        assert counts==actual,(rid,arm,seed,tol,point,counts,actual)
                        rows_compared+=1
    summary = dict(checked_records=sorted(sample),checked_count_rows=rows_compared,disagreements=0,
        si0109_si0211_waveform_files_identical=same_signal,si0109_si0211_lead_II_labels_identical=same_labels)
    (args.out/"independent_count_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(rows_compared,"independent TP/FP/FN checks passed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rdb",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    verify(p.parse_args())
