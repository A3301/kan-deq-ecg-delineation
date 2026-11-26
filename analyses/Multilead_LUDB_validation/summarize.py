"""Subject-cluster bootstrap; paired across leads, fitted heads, folds and endpoints."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
LEADS=["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
NUM=["reference","predicted","matched","missed","extras","correct10","correct20","correct30","sum_error","sum_abs","common","common_sum_error","common_sum_abs"]
POINTS=("P_on","P_off","QRS_on","QRS_off","T_on","T_off")
ARMS=[(a,s) for a in ("KAN","GELU") for s in (1,2,3)]

def metrics(totals):
    c={k:totals[:,i] for i,k in enumerate(NUM)}
    def pct(k,d):return 100*c[k]/c[d]
    out={"detected_pct":pct("matched","reference"),"ppv_pct":pct("matched","predicted"),"common_coverage_pct":pct("common","reference"),"bias_ms":c["sum_error"]/c["matched"],"mae_ms":c["sum_abs"]/c["matched"],"common_mae_ms":c["common_sum_abs"]/c["common"],"common_bias_ms":c["common_sum_error"]/c["common"]}
    for th in (10,20,30):
        key="correct"+str(th);out["allref_"+str(th)+"ms_pct"]=pct(key,"reference")
        out["allpred_"+str(th)+"ms_pct"]=pct(key,"predicted")
        out["duration_f1_"+str(th)+"ms_pct"]=200*c[key]/(c["reference"]+c["predicted"])
    return out

def boot(ids):
    n=len(ids);rng=np.random.default_rng(20261125)
    return np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=10000)]).astype(np.float64)
def add_result(out,base,vals):
    lo,hi=np.quantile(vals[1:],[.025,.975])
    out.append(dict(**base,estimate=vals[0],lower=lo,upper=hi))

def main(path):
    q=pd.read_csv(path/"qrs_counts.csv",dtype={"record":str});b=pd.read_csv(path/"boundary_counts.csv",dtype={"record":str})
    sums=q[q.scope.isin(LEADS)].groupby(["record","subset","head","seed"],as_index=False)[NUM].sum()
    sums["scope"]="all_leads";q=pd.concat([q,sums],ignore_index=True)
    output=[];landmarks=[];denom=[]
    for subset in ["all200","additional159","original41"]:
        sf=q if subset=="all200" else q[q.subset.eq(subset)]
        ids=sorted(sf.record.unique());weights=boot(ids);groups=sf.groupby("scope")
        for scope,r in groups:
            if subset!="all200" and scope not in ("all_leads","global8","global12","ii"):continue
            vals={}
            for (head,seed),g in r.groupby(["head","seed"]):
                arr=g.set_index("record").loc[ids,NUM].to_numpy(dtype="float64")
                total=weights@arr;values=metrics(total)
                vals[head,seed]=values
                denom.append(dict(subset=subset,scope=scope,head=head,seed=seed,participants=len(ids),participants_with_reference=int(np.sum(arr[:,0]>0)),**{k:int(v) for k,v in zip(NUM,total[0])}))
            for k in vals["KAN",1]:
                means={h:sum(vals[h,s][k] for s in (1,2,3))/3 for h in ("KAN","GELU")}
                for arm,v in [*means.items(),("KAN_minus_GELU",means["KAN"]-means["GELU"])]:
                    add_result(output,dict(subset=subset,participants=len(ids),scope=scope,arm=arm,metric=k),v)
        marks=b if subset=="all200" else b[b.subset.eq(subset)]
        for tol,sub in marks.groupby("tol_ms"):
            scores={}
            for head,seed in ARMS:
                model=sub[(sub["head"]==head)&(sub.seed==seed)]
                arr=np.stack([model[model.landmark.eq(k)].set_index("record").loc[ids,["tp","fn","fp"]].values for k in POINTS],axis=1)
                cnt=(weights@arr.reshape(len(ids),-1)).reshape(len(weights),6,3)
                f=200*cnt[:,:,0]/(2*cnt[:,:,0]+cnt[:,:,1]+cnt[:,:,2])
                scores[head,seed]=np.column_stack([f,f.mean(axis=1)])
            for seed in [1,2,3,"mean"]:
                a,z=[sum(scores[h,s] for s in [1,2,3])/3 if seed=="mean" else scores[h,seed] for h in ["KAN","GELU"]]
                for arm,curve in [("KAN",a),("GELU",z),("KAN_minus_GELU",a-z)]:
                    for j,metric in enumerate([*POINTS,"macro"]):
                        add_result(landmarks,dict(subset=subset,participants=len(ids),arm=arm,seed=seed,tol_ms=tol,landmark=metric),curve[:,j])
    pd.DataFrame(denom).to_csv(path/"duration_denominators.csv",index=False)
    pd.DataFrame(output).to_csv(path/"duration_summary.csv",index=False)
    pd.DataFrame(landmarks).to_csv(path/"boundary_summary.csv",index=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=HERE/"outputs")
    main(ap.parse_args().out)
