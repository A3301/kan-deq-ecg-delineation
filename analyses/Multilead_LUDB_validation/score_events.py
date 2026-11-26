"""Reconstruct lead-specific and time-coherent QRS intervals; preserve all denominators."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from frozen_model import merge,runs

HERE=Path(__file__).resolve().parent
LEADS=["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
POINTS=("P_on","P_off","QRS_on","QRS_off","T_on","T_off")
ARMS=[(a,s) for a in ("KAN","GELU") for s in (1,2,3)]

def extract(mask,kind):
    return [(4*a,4*b) for a,b in merge(runs(mask,kind)) if b-a+1 >= (15 if kind==3 else 10)]

def associations(ref,det,tol=150):
    """Greedy positive overlap and nearest midpoint; no edge-/width-error threshold."""
    cand=[]
    for i,(a,b) in enumerate(ref):
        for j,(c,d) in enumerate(det):
            dist=abs((a+b)-(c+d))/2
            if dist<=tol and min(b,d)>max(a,c):cand.append((dist,i,j))
    matches={};taken=set()
    for _,i,j in sorted(cand):
        if i not in matches and j not in taken:matches[i]=j;taken.add(j)
    return matches

def fuse(leads):
    groups=[]
    ordered=sorted(((a+b)/2,i,a,b) for i,li in enumerate(leads) for a,b in li)
    for center,i,a,b in ordered:
        candidates=[]
        for k,g in enumerate(groups):
            if center-g["centers"][0]>120 or i in g["leads"]:continue
            candidates.append((abs(center-np.median(g["centers"])),k))
        if candidates:
            group=groups[min(candidates)[1]]
        else:
            group={"leads":[],"centers":[],"on":[],"off":[]};groups.append(group)
        group["leads"].append(i);group["centers"].append(center);group["on"].append(a);group["off"].append(b)
    return groups

def in_span(pair,span):
    return span is not None and 2*span[0]<=pair[0]+pair[1]<=2*span[1]

def global_pairs(groups,threshold,span):
    return [(min(g["on"]),max(g["off"])) for g in groups if len(g["leads"])>=threshold and in_span((min(g["on"]),max(g["off"])),span)]

def mark_counts(ref,det,span,tol):
    pairs=sorted((abs(a-b),i,j) for i,a in enumerate(ref) for j,b in enumerate(det) if abs(a-b)<=tol)
    found=set();used=set()
    for _,i,j in pairs:
        if i not in found and j not in used:found.add(i);used.add(j)
    fp=sum(span is not None and span[0]<=b<=span[1] and j not in used for j,b in enumerate(det))
    return len(found),len(ref)-len(found),fp

def main(work):
    out=work/"outputs"; subjects=json.loads((out/"reference.json").read_text())
    masks={arm:np.load(out/f"{arm[0]}_{arm[1]}_masks.npz")["masks"] for arm in ARMS}
    qrows=[]; brows=[]; group_rows=[]
    evtfile=gzip.open(out/"paired_events.csv.gz","wt",encoding="utf8",newline="")
    events=csv.writer(evtfile);events.writerow(["record","scope","arm","seed","reference_index","ref_on_ms","ref_off_ms","pred_index","pred_on_ms","pred_off_ms"])
    for pos,rec in enumerate(subjects):
        rid=rec["record"];sub=rec["subset"]
        spans=[rec["leads"][lead]["span"] for lead in LEADS]
        span=None if any(x is None for x in spans) else [max(x[0] for x in spans),min(x[-1] for x in spans)]
        if span is not None and span[0]>span[1]:span=None
        reference={lead:[tuple(r) for r in rec["leads"][lead]["qrs"] if in_span(r,spans[j])] for j,lead in enumerate(LEADS)}
        rawref=[list(map(tuple,rec["leads"][lead]["qrs"])) for lead in LEADS]
        rg=fuse(rawref)
        def add_group_diagnostics(groups,model,seed):
            valid=[g for g in groups if in_span((min(g["on"]),max(g["off"])),span)]
            sizes=Counter(len(g["leads"]) for g in valid)
            group_rows.append(dict(record=rid,model=model,seed=seed,span_ms=None if span is None else span[1]-span[0],
                                   groups_in_span=len(valid),n_full=sizes[12],n_8_11=sum(sizes[n] for n in range(8,12)),
                                   n_1_7=sum(sizes[n] for n in range(1,8)),max_accepted_center_span=max((max(g["centers"])-min(g["centers"]) for g in valid),default=0)))
        add_group_diagnostics(rg,"reference",0)
        for t in [8,12]:reference[f"global{t}"]=global_pairs(rg,t,span)
        per_arm={}
        for arm in ARMS:
            kind,seed=arm; record=masks[arm][pos]
            qlists=[[(a,b) for a,b in extract(mask,2) if 0<a<b<9996] for mask in record]
            predictions={lead:[v for v in qlists[j] if in_span(v,spans[j])] for j,lead in enumerate(LEADS)}
            cg=fuse(qlists);add_group_diagnostics(cg,kind,seed)
            for t in [8,12]:predictions[f"global{t}"]=global_pairs(cg,t,span)
            out_matches={scope:associations(ref,predictions[scope]) for scope,ref in reference.items()}
            per_arm[arm]=(predictions,out_matches)
            ref_ii=rec["leads"]["ii"]
            for wi,c in enumerate(["P","QRS","T"],1):
                intervals=extract(record[1],wi)
                for side,k in enumerate(["on","off"]):
                    q=c+"_"+k;dp=[a[side] for a in intervals]
                    for tol in [150,40]:
                        tp,fn,fp=mark_counts(ref_ii["points"][q],dp,ref_ii["span"],tol)
                        brows.append(dict(record=rid,subset=sub,head=kind,seed=seed,landmark=q,tol_ms=tol,tp=tp,fn=fn,fp=fp))
        for scope,ref in reference.items():
            common=set(range(len(ref)))
            for _,matches in per_arm.values():common.intersection_update(matches[scope])
            for arm,(preds,gmatch) in per_arm.items():
                pred=preds[scope];m=gmatch[scope]
                errs={i:pred[j][1]-pred[j][0]-(ref[i][1]-ref[i][0]) for i,j in m.items()}
                for i,(a,b) in enumerate(ref):
                    j=m.get(i)
                    events.writerow([rid,scope,*arm,i,a,b,j,*(pred[j] if j is not None else ["",""])])
                used=set(m.values())
                for j,(a,b) in enumerate(pred):
                    if j not in used:events.writerow([rid,scope,*arm,"","","",j,a,b])
                qrows.append(dict(record=rid,subset=sub,scope=scope,head=arm[0],seed=arm[1],reference=len(ref),predicted=len(pred),matched=len(m),missed=len(ref)-len(m),extras=len(pred)-len(m),correct10=sum(abs(v)<=10 for v in errs.values()),correct20=sum(abs(v)<=20 for v in errs.values()),correct30=sum(abs(v)<=30 for v in errs.values()),sum_error=sum(errs.values()),sum_abs=sum(abs(v) for v in errs.values()),common=len(common),common_sum_error=sum(errs[i] for i in common),common_sum_abs=sum(abs(errs[i]) for i in common)))
    evtfile.close()
    pd.DataFrame(qrows).to_csv(out/"qrs_counts.csv",index=False);pd.DataFrame(brows).to_csv(out/"boundary_counts.csv",index=False)
    pd.DataFrame(group_rows).to_csv(out/"fusion_audit.csv",index=False)
    print("Scored",len(qrows),"QRS count rows and",len(brows),"boundary count rows",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--work",type=Path,default=HERE)
    main(ap.parse_args().work)
