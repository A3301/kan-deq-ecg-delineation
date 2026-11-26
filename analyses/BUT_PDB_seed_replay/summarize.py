"""Aggregate each locked ensemble's P-peak counts and pair by original source recording."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
FAMILY=("cnn_bilstm","unet_kan","convkan","deq")

def f1(counts):
    t,n,p=counts.T
    return np.divide(200.0*t,2*t+n+p)

def holm(p):
    a=np.argsort(p)
    out=np.empty(len(p))
    out[a]=np.minimum(1,np.maximum.accumulate(np.array(p)[a]*(len(p)-np.arange(len(p)))))
    return out

def nonoverlapping(ref):
    drop=[]
    for _,rows in ref.groupby("source"):
        keep=[]
        for r in rows.sort_values("segment").itertuples():
            if any(r.source_start<=end and begin<=r.source_end for begin,end in keep):
                drop.append(r.segment)
            else:
                keep.append((r.source_start,r.source_end))
    assert drop==["26"],drop
    return drop

def archive_check(counts,summary):
    hist=counts[(counts.seed==1)&(counts.tol_ms==150)&(counts.coordinate=="historical_grid")]
    expected=pd.read_csv(ROOT/"source"/"but_pooled_counts.csv").set_index(["model","source"])[["tp","fn","fp"]]
    sent=pd.read_csv(ROOT/"source"/"sentinel_predictions.csv")
    for ix,g in sent.groupby(["model","source"]):
        expected.loc[ix,:]=[0,0,g.predicted_P.sum()]
    observed=hist.groupby(["model","source"])[["tp","fn","fp"]].sum()
    pd.testing.assert_frame_equal(expected.sort_index(),observed.sort_index(),check_dtype=False)
    prior=pd.read_csv(ROOT/"source"/"archived_seed1_h3.csv").query("cohort=='full_corrected'").set_index("model")
    sub=summary[(summary.seed=="1")&(summary.cohort=="all50")&(summary.coordinate=="historical_grid")&
                (summary.tol_ms==150)].set_index("model")
    for name,key in [("model_F1","model_F1"),("comparator_F1","baseline_F1"),("difference","delta_pp"),
                     ("lo","CI_low"),("hi","CI_high"),("p_holm","p_holm")]:
        np.testing.assert_allclose(sub.loc[list(FAMILY[1:]),name],prior.loc[list(FAMILY[1:]),key],atol=5e-8)
    return {"archived_corrected_source_models_exact":len(expected),"archived_three_contrasts_and_intervals_verified":True}

def main(seed1=False,historical_archive_check=True):
    OUT.mkdir(exist_ok=True)
    seeds=[1] if seed1 else [1,2,3]
    parts=[ROOT/"parts"/f"{m}_s{s}" for m in FAMILY for s in seeds]
    assert all((p/"verified.json").is_file() for p in parts), "A full five-fold ensemble is missing"
    counts=pd.concat([pd.read_csv(p/"counts.csv",dtype={"segment":str}) for p in parts],ignore_index=True)
    preds=pd.concat([pd.read_csv(p/"predictions.csv",dtype={"segment":str}) for p in parts],ignore_index=True)
    ref=pd.read_csv(ROOT/"source"/"references.csv",dtype={"segment":str})
    assert len(counts)==len(seeds)*4*100*4
    assert len(preds)==len(seeds)*4*100
    assert counts.groupby(["model","seed","coordinate","tol_ms"]).apply(lambda d:(d.tp+d.fn).sum()).eq(10858).all()
    dropped=nonoverlapping(ref)
    rows=[]
    for cname,sub in [("all50",counts),("nonoverlap49",counts[~counts.segment.isin(dropped)])]:
        for (method,tol),g in sub.groupby(["coordinate","tol_ms"]):
            keys=sorted(g.source.unique())
            assert len(keys)==45
            table=g.groupby(["model","seed","source"])[["tp","fn","fp"]].sum()
            draws=np.random.default_rng(0).integers(0,len(keys),(10000,len(keys)))
            data={}
            for m in FAMILY:
                values=[]
                for s in seeds:
                    events=table.xs((m,s),level=["model","seed"]).loc[keys].to_numpy()
                    values.append(f1(np.vstack([events.sum(axis=0),events[draws].sum(axis=1)])))
                for s,v in zip(seeds,values):
                    data[m,str(s)]=v
                if not seed1:
                    data[m,"mean"]=np.mean(values,axis=0)
            for s in [str(x) for x in seeds]+([] if seed1 else ["mean"]):
                family_rows=[]
                for m in FAMILY[1:]:
                    val,cmp=data[m,s],data[FAMILY[0],s]
                    diff=val-cmp
                    lo,hi=np.quantile(diff[1:],[.025,.975])
                    p=min(1,2*min((diff[1:]>=0).mean(),(diff[1:]<=0).mean()))
                    family_rows.append(dict(cohort=cname,segments=sub.segment.nunique(),source_records=len(keys),
                                        coordinate=method,tol_ms=tol,seed=s,model=m,comparator=FAMILY[0],
                                        model_F1=val[0],comparator_F1=cmp[0],difference=diff[0],lo=lo,hi=hi,p_tail=p))
                for r,p in zip(family_rows,holm([r["p_tail"] for r in family_rows])):
                    r["p_holm"]=p
                rows.extend(family_rows)
    summary=pd.DataFrame(rows)
    if historical_archive_check:
        qa=archive_check(counts,summary)
    else:
        qa={"panel":"uniform GPU training","archived_CPU_seed1_comparison":"different fitted weights; primary-benchmark seed1 is verified in the mixed replay",
            "number_of_scored_fivefold_ensembles":4*len(seeds),"source_records":45}
    prefix="seed1_check_" if seed1 else ""
    summary.to_csv(OUT/f"{prefix}contrasts.csv",index=False)
    (OUT/f"{prefix}archive_check.json").write_text(json.dumps(qa,indent=2))
    if not seed1:
        counts.to_csv(OUT/"events.csv",index=False)
        preds.to_csv(OUT/"predictions.csv",index=False)
        counts[counts.segment.isin(["08","48"])].to_csv(OUT/"negative_reference_events.csv",index=False)
        weights={p.name:json.loads((p/"verified.json").read_text())["inputs"] for p in parts}
        (OUT/"weights.json").write_text(json.dumps(weights,indent=2))
    print(qa)
    print(summary[(summary.cohort=="all50")&(summary.coordinate=="historical_grid")&
                  (summary.seed==("1" if seed1 else "mean"))].round(3).to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed1",action="store_true")
    main(ap.parse_args().seed1)
