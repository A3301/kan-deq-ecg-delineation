"""Independent reread of the ISP headers/annotations and of the paired summary."""
import argparse
import ast
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

NAMES=["P_on","P_off","QRS_on","QRS_off","T_on","T_off"]


def check(isp,out):
    roster=pd.read_csv(out/"cohort.csv").set_index("record")
    with ZipFile(isp) as z:
        assert z.testzip() is None
        for split in ("train","test"):
            prefix="isp_delineation_dataset/"+split
            for row in csv.DictReader(StringIO(z.read(prefix+"_isp_delineation_data.csv").decode())):
                rid=split+"/"+row["file_name"];info=roster.loc[rid]
                stem=prefix+"_data/"+row["file_name"]
                header=z.read(stem+".hea").decode().splitlines()
                n=int(header[0].split()[3]);fs=int(header[0].split()[2])
                assert sha256(z.read(stem+".dat")).hexdigest()==info.raw_sha256
                assert (n,fs)==(info.native_samples,info.native_fs) and header[1+int(info.lead_index_zero_based)].split()[-1].lower()=="ii"
                tri=ast.literal_eval(row["target"])
                for k,name in enumerate(NAMES):
                    gold={t[k%2+1] for t in tri if t[0]==k//2 and t[1]<t[2] and 0<t[k%2+1]<n-1}
                    assert len(gold)==info[name],rid
    counts=pd.read_csv(out/"event_counts.csv")
    assert not counts.duplicated(["source","head","seed","record","tol_ms","landmark"]).any()
    assert len(roster)==475 and roster.raw_sha256.nunique()==475 and roster.status.eq("included").all()
    assert len(counts)==475*6*2*(2*3+2*2)
    truth=roster[NAMES].stack()
    truth.index.names=["record","landmark"]
    observed=counts.set_index(["record","landmark"])
    assert np.array_equal(truth.reindex(observed.index),observed.tp+observed.fn)
    summary=pd.read_csv(out/"comparison.csv",dtype={"seed":str})
    order=sorted(roster.index)
    rng=np.random.default_rng(20261125)
    b=rng.multinomial(len(order),np.ones(len(order))/len(order),size=10000)
    b=np.vstack([np.ones((1,len(order)),dtype=int),b])
    maxerr=0.
    for (source,tol),g in counts.groupby(["source","tol_ms"]):
        f={}
        for (head,seed,k),data in g.groupby(["head","seed","landmark"]):
            d=data.set_index("record").reindex(order)
            num=b@(2*d.tp.to_numpy())
            den=num+b@(d.fn+d.fp).to_numpy()
            f[head,seed,k]=100*num/den
        seeds=sorted(g.seed.unique())
        for seed in seeds:
            for head in ["KAN","GELU"]:
                f[head,seed,"macro"]=np.mean([f[head,seed,k] for k in NAMES],axis=0)
        for k in [*NAMES,"macro"]:
            for head in ["KAN","GELU"]:
                f[head,"mean",k]=np.mean([f[head,seed,k] for seed in seeds],axis=0)
        for _,v in summary[(summary.source==source)&(summary.tol_ms==tol)].iterrows():
            seed=int(v.seed) if v.seed!="mean" else "mean"
            a,bt=[f[h,seed,v.landmark] for h in ("KAN","GELU")];delta=a-bt
            lhs=[a[0],bt[0],delta[0],*np.quantile(delta[1:],[.025,.975])]
            rhs=[v.KAN_F1,v.GELU_F1,v.difference_pp,v.lower,v.upper]
            maxerr=max(maxerr,float(np.max(np.abs(np.array(lhs)-rhs))))
    assert maxerr<1e-9
    report={"input_records":475,"reference_landmarks":int(roster[NAMES].sum().sum()),
            "score_rows":len(counts),"summary_rows":len(summary),"max_abs_numeric_difference":maxerr,
            "source_annotations_and_header_verification":"passed","paired_CIs_and_estimates_verification":"passed"}
    (out/"independent_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--isp",type=Path,required=True);p.add_argument("--results",type=Path,required=True)
    a=p.parse_args();check(a.isp,a.results)
