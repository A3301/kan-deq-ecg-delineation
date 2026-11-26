"""Re-use the locked parent protocol to assess six-lead widths in complementary sets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROTOCOL = "e5752788c4827d12e6f50bed563a2668fa4b611e74e7eaeE5459e698cb8001aa".lower()
PARENT_ARCHIVE = "9e4ed873e0f7601bee2a8991dfa6d814ffd529f4d8262d76e0069afadbcb6b65"
DIRECTIONS = {"limb_group_chest_target":(range(6),range(6,12)), "chest_group_limb_target":(range(6,12),range(6))}
ARMS = [(h,s) for h in ("KAN","GELU") for s in (1,2,3)]
METRICS = ["bias_ms","allref20_pct","mae_ms","detected_pct"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def geometry(group,indices):
    rows=[i for i,lead in enumerate(group["leads"]) if lead in indices]
    assert len(rows)==len(indices)
    return min(group["on"][i] for i in rows),max(group["off"][i] for i in rows)


def create_eligibility(headers,table):
    rows=[]
    for record in range(1,201):
        text=(headers/f"{record}.hea").read_text()
        match=re.search(r"^#<age>:\s*([0-9]+)",text,re.M)
        rows.append((record,bool(match and int(match.group(1))>=18)))
    pd.DataFrame(rows,columns=["record","documented_adult"]).to_csv(table,index=False)


def score(parent,out,age_file):
    sys.path.insert(0,str(parent.resolve()))
    from score_events import LEADS, associations, extract, fuse, in_span
    frozen=parent/"outputs"
    manifest=json.loads((frozen/"input_manifest.json").read_text())
    assert digest(parent/"protocol.txt")==manifest["protocol"]
    masks={}
    for arm in ARMS:
        key=f"{arm[0]}_{arm[1]}"
        p=frozen/f"{key}_masks.npz"
        assert manifest["masks"][key]==digest(p)
        masks[arm]=np.load(p)["masks"]
    subjects=json.loads((frozen/"reference.json").read_text())
    eligible=pd.read_csv(age_file)
    adults=set(eligible.loc[eligible.documented_adult,"record"])
    assert len(adults)==196
    old=pd.read_csv(frozen/"paired_events.csv.gz").query("scope=='global12' and arm=='KAN' and seed==1 and reference_index==reference_index")
    old=old[old.record.isin(adults)].sort_values(["record","reference_index"])
    expected={int(k):list(zip(g.ref_on_ms.astype(int),g.ref_off_ms.astype(int))) for k,g in old.groupby("record")}
    refs=[];events=[];totals=[]
    for pos,rec in enumerate(subjects):
        rid=int(rec["record"])
        if rid not in adults:continue
        data=rec["leads"]
        span=[max(data[l]["span"][0] for l in LEADS),min(data[l]["span"][1] for l in LEADS)]
        full=[g for g in fuse([data[x]["qrs"] for x in LEADS]) if len(g["leads"])==12 and in_span(geometry(g,range(12)),span)]
        assert [geometry(g,range(12)) for g in full]==expected.get(rid,[]),rid
        for i,g in enumerate(full):
            limb=geometry(g,range(6));chest=geometry(g,range(6,12))
            refs.append((rid,i,*limb,*chest))
        for head,seed in ARMS:
            intervals=[[(a,b) for a,b in extract(mask,2) if 0<a<b<9996] for mask in masks[head,seed][pos]]
            for label,(group_ids,target_ids) in DIRECTIONS.items():
                source=[geometry(g,group_ids) for g in full]
                target=[geometry(g,target_ids) for g in full]
                # Reference eligibility came from the original full-12 interval. Do not
                # remove a gold beat because its target-only midpoint moves across that edge.
                candidates=fuse([intervals[j] for j in target_ids])
                pr=[geometry(g,range(6)) for g in candidates if len(g["leads"])==6 and in_span(geometry(g,range(6)),span)]
                ix=associations(target,pr)
                totals.append((rid,label,head,seed,len(target),len(pr),len(ix),len(pr)-len(ix)))
                for i,(a,b) in enumerate(target):
                    baseline=source[i][1]-source[i][0]
                    j=ix.get(i);pred=pr[j] if j is not None else (None,None)
                    events.append((rid,i,label,head,seed,baseline,b-a,*pred))
    ref=pd.DataFrame(refs,columns=["record","beat","limb_on","limb_off","chest_on","chest_off"])
    assert len(ref)==1746 and ref.record.nunique()==193
    ref.to_csv(out/"beat_reference.csv",index=False)
    ev=pd.DataFrame(events,columns=["record","beat","direction","head","seed","group_width","target_width","pred_on","pred_off"])
    ev.to_csv(out/"events.csv.gz",index=False)
    pd.DataFrame(totals,columns=["record","direction","head","seed","reference","predicted","matched","unassigned_extra"]).to_csv(out/"matching.csv",index=False)


def stats(a):
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.stack([a[...,2]/a[...,1],100*a[...,4]/a[...,0],a[...,3]/a[...,1],100*a[...,1]/a[...,0]],axis=-1)


def summarize(out,age_file):
    adult=pd.read_csv(age_file); people=sorted(adult.loc[adult.documented_adult,"record"].tolist())
    ev=pd.read_csv(out/"events.csv.gz")
    ev["stratum"]=np.where(ev.group_width>=120,"wide","narrow")
    ev["reference"]=1
    ev["matched"]=ev.pred_on.notna().astype(int)
    ev["sum_error"]=(ev.pred_off-ev.pred_on-ev.target_width).fillna(0)
    ev["sum_abs"]=ev.sum_error.abs()
    ev["correct20"]=((ev.sum_abs<=20)&(ev.matched==1)).astype(int)
    idx=["direction","head","seed","stratum","record"]
    nums=["reference","matched","sum_error","sum_abs","correct20"]
    counts=ev.groupby(idx)[nums].sum().astype("int64")
    counts.to_csv(out/"adult_counts.csv")
    check=pd.read_csv(out/"adult_counts.csv").set_index(idx)
    pd.testing.assert_frame_equal(counts,check)
    weights=np.vstack([np.ones(len(people)),np.random.default_rng(20261125).multinomial(len(people),[1/len(people)]*len(people),size=10000)])
    runs=[];summary=[]
    for label in DIRECTIONS:
        for head in ("KAN","GELU"):
            means={}
            for strat in ("narrow","wide"):
                vals=[]
                for seed in (1,2,3):
                    arr=check.loc[label,head,seed,strat].reindex(people,fill_value=0)[nums].values
                    agg=weights@arr
                    metrics=stats(agg)
                    assert np.isfinite(metrics).all()
                    vals.append(metrics)
                    single=ev[(ev.direction==label)&(ev["head"]==head)&(ev.seed==seed)&(ev.stratum==strat)]
                    runs.append(dict(direction=label,head=head,seed=seed,stratum=strat,participants=single.record.nunique(),
                                reference=int(agg[0,0]),matched=int(agg[0,1]),correct20=int(agg[0,4]),
                                median_group_width=single.group_width.median(),median_target_width=single.target_width.median(),
                                **dict(zip(METRICS,metrics[0].tolist()))))
                means[strat]=np.mean(vals,axis=0)
            for band,values in [*means.items(),("wide_minus_narrow",means["wide"]-means["narrow"])]:
                intervals=np.quantile(values[1:],[.025,.975],axis=0)
                for k,measure in enumerate(METRICS):
                    summary.append(dict(direction=label,head=head,stratum=band,metric=measure,estimate=values[0,k],lo=intervals[0,k],hi=intervals[1,k]))
    pd.DataFrame(runs).to_csv(out/"by_seed.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent",type=Path,required=True,help="Extracted parent archive directory")
    ap.add_argument("--out",type=Path,default=HERE/"results")
    ap.add_argument("--ages",type=Path,default=HERE/"adult_eligibility.csv")
    ap.add_argument("--refresh-headers",type=Path)
    ap.add_argument("--summaries-only",action="store_true")
    args=ap.parse_args()
    assert digest(HERE/"protocol.txt")==PROTOCOL
    args.out.mkdir(parents=True,exist_ok=True)
    if args.refresh_headers: create_eligibility(args.refresh_headers,args.ages)
    if not args.summaries_only: score(args.parent,args.out,args.ages)
    summarize(args.out,args.ages)
