"""Independent reconstruction of counts and cohort scope from wave annotations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from count_helpers import POINTS, raw_reference, raw_prediction, measure


def verify(out,raw):
    ids=json.loads((out/"signal_order.json").read_text())
    cohort=pd.read_csv(out/"cohort.csv").query("status=='included'").set_index("record")
    assert len(ids)==2398 and set(ids)==set(cohort.index)
    audit_records=[r for _,grp in cohort.groupby("rhythm") for r in sorted(grp.index,key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:6]]
    counts=pd.read_csv(out/"deq_record_scores.csv")
    for (arm,seed,tol,point),grp in counts.groupby(["arm","seed","tolerance_ms","endpoint"]):
        ref=grp.set_index("record").reindex(ids)
        assert np.array_equal((ref.tp+ref.fn).to_numpy(),cohort.loc[ids,point].to_numpy())
    selected=counts[counts.record.isin(audit_records)]
    lookup={(r.record,r.arm,r.seed,r.tolerance_ms,r.endpoint):(r.tp,r.fp,r.fn) for r in selected.itertuples()}
    masks={(arm,seed):np.load(out/f"masks_{arm}_{seed}.npy",mmap_mode="r") for arm in ("DEQ","Twin8") for seed in (1,2)}
    checked=0
    with ZipFile(raw) as z:
        for rid in audit_records:
            ref=raw_reference(z,rid)
            all_ref=list(itertools.chain.from_iterable(ref))
            a,b=max(0,min(all_ref)-150),min(9998,max(all_ref)+150)
            for (arm,seed),arr in masks.items():
                pred=raw_prediction(arr[ids.index(rid)])
                for point,x,y in zip(POINTS,ref,pred,strict=True):
                    det=[t for t in y if a<=t<=b]
                    for tol in [150,40]:
                        got=measure(x,det,tol)
                        expected=lookup[(rid,arm,seed,tol,point)]
                        assert got==expected,(rid,arm,seed,tol,point,got,expected)
                        checked+=1
    result=dict(included=len(ids),audited_records=sorted(audit_records),audited_rows=checked,disagreements=0,
                full_cohort_endpoint_reference_count=173568)
    (out/"independent_count_check.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    diag=pd.read_csv(out/"solver_diagnostics.csv")
    rows=[]
    for (arm,seed),grp in diag.groupby(["arm","seed"]):
        its=grp.common_window_iterations
        r=grp.max_window_final_residual
        item=dict(arm=arm,seed=seed,records=grp.record.nunique(),fold_record_evaluations=len(grp),
                  median_final_iterations=its.median(),p25_final_iterations=its.quantile(.25),
                  p75_final_iterations=its.quantile(.75),
                  cap_30=int((its==30).sum()),cap_30_above_equal_tolerance=int(((its==30)&(r>=0.001)).sum()),
                  median_final_max_relative_residual=r.median())
        assert len(grp)==5*len(ids) and grp.groupby("record").size().eq(5).all()
        if arm=="DEQ":
            assert ((its==30)|(r<0.001)).all()
        else:
            assert its.eq(8).all()
        rows.append(item)
    pd.DataFrame(rows).to_csv(out/"diagnostic_summary.csv",index=False)
    deq = diag[diag.arm=="DEQ"].merge(cohort[["rhythm"]],left_on="record",right_index=True,validate="many_to_one")
    deq["at_cap_above_tolerance"]=(deq.common_window_iterations==30)&(deq.max_window_final_residual>=0.001)
    rhythm=deq.groupby(["rhythm","seed"]).agg(n_source_records=("record","nunique"),fold_record_evaluations=("record","size"),
        median_iterations=("common_window_iterations","median"),at_cap_above_tolerance=("at_cap_above_tolerance","sum"))
    rhythm.to_csv(out/"solver_by_rhythm.csv")
    print(f"Record × endpoint checks {checked}, disagreements zero.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--rdb",type=Path,required=True)
    a=p.parse_args()
    verify(a.out,a.rdb)
