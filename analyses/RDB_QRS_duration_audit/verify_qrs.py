"""Reconstruct QRS matches independently on 48 ECGs and all eight model/seed outputs."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd


ARMS=["KAN","GELU","DEQ","Twin8"]


def calculate(raw,mask,left,right):
    # Alternate implementation uses grouping of the full class-label sequence.
    true={};seen=set()
    for i,row in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
        wave=int(row["TYPE"])
        s,e=(2*float(row[c]) for c in ("START","END"))
        if wave==1 and 0<s<e<9998 and left <= (s+e)/2 <= right and (s,e) not in seen:
            true[i]=(s,e);seen.add((s,e))
    segments=[]
    pos=0
    for v,g in itertools.groupby(mask):
        L=sum(1 for _ in g)
        if v==2:
            if segments and pos-(segments[-1][1]+1)<5:
                segments[-1]=(segments[-1][0],pos+L-1)
            else:
                segments.append((pos,pos+L-1))
        pos+=L
    pred=[(s*4,e*4) for s,e in segments if s>0 and e<2499 and e-s>=9 and left<=(s+e)*2<=right]
    ids=list(true)
    if not ids or not pred: return {},len(true),len(pred)
    t=np.array(list(true.values()));p=np.array(pred)
    distances=np.abs(t.mean(1)[:,None]-p.mean(1)[None,:])
    intersect=np.minimum(t[:,1,None],p[None,:,1])>np.maximum(t[:,0,None],p[None,:,0])
    ix,ixp=np.nonzero((distances<=150)&intersect)
    usedt=set();usedp=set();pairs={}
    for k in np.argsort(distances[ix,ixp],kind="stable"):
        a,b=ix[k],ixp[k]
        if a in usedt or b in usedp: continue
        usedt.add(a);usedp.add(b)
        e=(pred[b][1]-pred[b][0])-(true[ids[a]][1]-true[ids[a]][0])
        pairs[ids[a]]=(true[ids[a]],pred[b],e)
    return pairs,len(true),len(pred)


def audit(args):
    ev=pd.read_csv(args.out/"matched_qrs_events.csv.gz")
    rec=pd.read_csv(args.out/"record_duration.csv").set_index(["record","arm","seed"])
    cover=pd.read_csv(args.out/"record_coverage.csv").set_index(["record","arm","seed"])
    assert not ev.duplicated(["record","arm","seed","reference_row"]).any()
    diff=ev.predicted_off_ms-ev.predicted_on_ms-ev.reference_off_ms+ev.reference_on_ms
    assert np.allclose(ev.duration_error_ms,diff,atol=1e-8,rtol=0)
    common=ev[ev.four_way_common==1].copy()
    common["abs_error"]=common.duration_error_ms.abs()
    sums=common.groupby(["record","arm","seed"]).agg(common=("duration_error_ms","size"),
                 abs_sum_ms=("abs_error","sum"),signed_sum_ms=("duration_error_ms","sum")).reindex(rec.index,fill_value=0)
    assert np.allclose(sums.to_numpy(),rec[list(sums.columns)].to_numpy(),atol=1e-7,rtol=0)
    ids=json.loads((args.heads/"signal_order.json").read_text())
    meta=pd.read_csv(args.heads/"cohort.csv").query("status=='included'").set_index("record")
    sample=[r for _,s in meta.groupby("rhythm") for r in sorted(s.index,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:6]]
    masks={(arm,seed):np.load((args.heads if arm in ARMS[:2] else args.equilibrium)/f"masks_{arm}_{seed}.npy",mmap_mode="r")
           for arm in ARMS for seed in (1,2)}
    sample_df=ev[ev.record.isin(sample)]
    groups={k:v.set_index("reference_row") for k,v in sample_df.groupby(["record","arm","seed"])}
    beats=0
    with ZipFile(args.rdb) as z:
        for rid in sample:
            i=ids.index(rid);lo=meta.loc[rid,"score_start_ms"];hi=meta.loc[rid,"score_end_ms"]
            for key,arr in masks.items():
                k=(rid,*key)
                expected,nref,npred=calculate(z.read(f"ann_txt/{rid}.ii.txt"),arr[i],lo,hi)
                c=cover.loc[k]
                assert nref==c.references and npred==c.predictions and len(expected)==c.matches
                actual=groups.get(k,pd.DataFrame())
                assert set(expected)==set(actual.index)
                for ri,(true,pred,error) in expected.items():
                    r=actual.loc[ri]
                    assert np.allclose([r.reference_on_ms,r.reference_off_ms,r.predicted_on_ms,r.predicted_off_ms,r.duration_error_ms],[*true,*pred,error],atol=1e-8,rtol=0)
                    beats+=1
    summary=dict(sampled_records=len(sample),audited_record_model_seed_cells=len(sample)*8,
                 independently_reconstructed_matches=beats,individually_checked_event_rows=len(ev),
                 verified_event_counts_vs_record_aggregates=len(rec),mismatches=0,sample_ids=sample)
    (args.out/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ["rdb","heads","equilibrium","out"]: p.add_argument("--"+field,type=Path,required=True)
    audit(p.parse_args())
