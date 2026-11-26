"""Secondary post hoc test using literal across-reader medians per boundary."""
import argparse
import json
import numpy as np
import pandas as pd
from evaluate import RESULTS, SOURCE, retrieve_beat

def audit(wfdb_check=False):
    reference=pd.read_csv(SOURCE/"manual_reference.csv").set_index("id")
    storage=pd.read_csv(SOURCE/"cohort.csv").set_index("id")
    if wfdb_check:
        if hasattr(pd.options.future,"infer_string"):
            pd.options.future.infer_string=False
        import wfdb
        with np.load(SOURCE/"ptb_lead_ii_10s.npz",allow_pickle=False) as store:
            ids,waves=store["id"].tolist(),store["digital"]
        for rid,wave in zip(ids,waves):
            w=wfdb.rdrecord(str(SOURCE/"raw"/rid),sampfrom=0,sampto=10000,channel_names=["ii"])
            meta=storage.loc[rid]
            np.testing.assert_array_equal(w.p_signal[:,0],(wave.astype(float)-meta.baseline)/meta.gain)
        print(f"External WFDB {wfdb.__version__} verified all {len(ids)} native digital clips against official decoder")
    eligible=reference[(reference.Q>0)&(reference["T"]>reference.Q)].copy()
    medians={}
    for name in ["Q","T"]:
        data=eligible[[name.lower()+str(k) for k in range(1,6)]]
        np.testing.assert_array_equal(np.rint(data.mean(axis=1)),eligible[name])
        medians[name]=data.median(axis=1).astype(int)
    runs=pd.read_csv(RESULTS/"predicted_runs_at_250Hz.csv")
    reported=pd.read_csv(RESULTS/"per_record.csv").set_index(["id","head","seed"])
    rows=[]
    for r in runs.itertuples():
        if r.id not in eligible.index:
            continue
        qrs,ts=json.loads(r.qrs),json.loads(r.t)
        source=eligible.loc[r.id]
        a,b,code=retrieve_beat(qrs,ts,int(source.Q))
        recorded=reported.loc[(r.id,r.head,r.seed)]
        assert code==recorded.status
        assert (a is None and pd.isna(recorded.predicted_q_ms)) or a==recorded.predicted_q_ms
        assert (b is None and pd.isna(recorded.predicted_t_ms)) or b==recorded.predicted_t_ms
        mq,mt=medians["Q"][r.id],medians["T"][r.id]
        x,y,_=retrieve_beat(qrs,ts,mq)
        rows.append(dict(id=r.id,patient=r.id.split("/")[0],head=r.head,seed=r.seed,
                         success=int(y is not None and abs((y-x)-(mt-mq))<=40)))
    d=pd.DataFrame(rows)
    patient=d.groupby(["patient","head","seed"]).success.agg(["sum","size"])
    subjects=sorted(d.patient.unique())
    rng=np.random.default_rng(20261125)
    w=rng.multinomial(len(subjects),np.full(len(subjects),1/len(subjects)),size=10000)
    w=np.vstack([np.ones(len(subjects)),w])
    vals={}
    for h in ["KAN","GELU"]:
        arr=[]
        for s in [1,2,3]:
            z=patient.xs((h,s),level=["head","seed"]).loc[subjects]
            arr.append(100*(w@z["sum"].to_numpy())/(w@z["size"].to_numpy()))
        vals[h]=np.mean(arr,axis=0)
    vals["difference_pp"]=vals["KAN"]-vals["GELU"]
    result={k:dict(point=v[0],lo=np.quantile(v[1:],.025),hi=np.quantile(v[1:],.975)) for k,v in vals.items()}
    result["actual_median_diff_counts"]={k:int((v!=eligible[k]).sum()) for k,v in medians.items()}
    result["all_published_columns_match_rounded_five_reader_mean"]=True
    (RESULTS/"posthoc_median_sensitivity.json").write_text(json.dumps(result,indent=2))
    print(result)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--wfdb",action="store_true",help="Also reread all 549 native clips with the official WFDB library.")
    audit(args.parse_args().wfdb)
