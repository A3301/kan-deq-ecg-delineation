"""Verify disjoint six-lead references, independent per-group statistics, and matching counts."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def digest(file):
    with file.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def verify(parent,out):
    refs=pd.read_csv(out/"beat_reference.csv")
    ev=pd.read_csv(out/"events.csv.gz")
    match=pd.read_csv(out/"matching.csv")
    by_seed=pd.read_csv(out/"by_seed.csv")
    summary=pd.read_csv(out/"summary.csv")
    native=json.loads((parent/"outputs"/"reference.json").read_text())
    expected=pd.read_csv(parent/"outputs"/"paired_events.csv.gz").query("scope=='global12' and arm=='KAN' and seed==1 and reference_index==reference_index")
    for i,g in refs.groupby("record"):
        old=expected[expected.record==i].sort_values("reference_index")
        np.testing.assert_array_equal(np.minimum(g.limb_on,g.chest_on),old.ref_on_ms)
        np.testing.assert_array_equal(np.maximum(g.limb_off,g.chest_off),old.ref_off_ms)
    keys=["record","beat","head","seed"]
    assert not ev.duplicated(keys+["direction"]).any()
    assert len(ev)==2*6*len(refs)
    x,y=[g.set_index(keys).sort_index() for _,g in ev.groupby("direction")]
    np.testing.assert_array_equal(x.group_width,y.target_width)
    np.testing.assert_array_equal(y.group_width,x.target_width)
    truth=refs.set_index(["record","beat"])
    collapsed=[]
    for direction,g in ev.groupby("direction"):
        a=g.set_index(["record","beat"]).index
        expected_width=truth.chest_off-truth.chest_on if direction.startswith("limb_group") else truth.limb_off-truth.limb_on
        np.testing.assert_array_equal(g.target_width,expected_width.loc[a])
        for (head,seed,wide),h in g.groupby(["head","seed",g.group_width.ge(120)]):
            row=by_seed[(by_seed.direction==direction)&(by_seed["head"]==head)&(by_seed.seed==seed)&(by_seed.stratum==("wide" if wide else "narrow"))].iloc[0]
            errors=(h.pred_off-h.pred_on-h.target_width).dropna()
            assert (len(h),len(errors),h.record.nunique())==(row.reference,row.matched,row.participants)
            success=100*errors.abs().le(20).sum()/len(h)
            np.testing.assert_allclose([errors.mean(),errors.abs().mean(),success],[row.bias_ms,row.mae_ms,row.allref20_pct])
        perkey=["record","direction","head","seed"]
        counted=g.groupby(perkey).pred_on.count()
        ct=match.set_index(perkey)
        np.testing.assert_array_equal(ct.loc[counted.index,"matched"],counted)
    # For a fixed, regularly spaced subset use a separate nearest-lead-II matcher.
    labels=list(native[0]["leads"])
    assert labels==["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
    rebuilt=0
    for rec in native[::12]:
        rid=int(rec["record"]);rr=refs[refs.record.eq(rid)]
        if rr.empty:continue
        dd=rec["leads"];span=[max(dd[l]["span"][0] for l in labels),min(dd[l]["span"][1] for l in labels)]
        published=set(zip(rr.limb_on,rr.limb_off,rr.chest_on,rr.chest_off))
        for a,b in dd["ii"]["qrs"]:
            midpoint=(a+b)/2; linked=[]
            for lead in labels:
                valid=[v for v in dd[lead]["qrs"] if abs(median(v)-midpoint)<=120]
                if len(valid)!=1:break
                linked.append(valid[0])
            if len(linked)!=12 or max(map(median,linked))-min(map(median,linked))>120:continue
            def extrema(pairs):return min(v[0] for v in pairs),max(v[1] for v in pairs)
            if not span[0]<=median(extrema(linked))<=span[1]:continue
            assert (*extrema(linked[:6]),*extrema(linked[6:])) in published,(rid,linked)
            rebuilt+=1
    # Count any target endpoint whose midpoint falls outside the parent common span.
    span_shift={}
    for half in ["limb","chest"]:
        n=0
        for rid,r in refs.groupby("record"):
            vals=native[int(rid)-1]["leads"].values()
            a,b=max(v["span"][0] for v in vals),min(v["span"][1] for v in vals)
            centers=(r[half+"_on"]+r[half+"_off"])/2
            n+=int((~centers.between(a,b)).sum())
        span_shift[half]=n
    for row in summary.itertuples():
        small=by_seed[(by_seed.direction==row.direction)&(by_seed["head"]==row.head)]
        means=small.groupby("stratum")[row.metric].mean()
        point=means.wide-means.narrow if row.stratum=="wide_minus_narrow" else means[row.stratum]
        np.testing.assert_allclose(point,row.estimate)
    audit={"parent_archive_sha256":"9e4ed873e0f7601bee2a8991dfa6d814ffd529f4d8262d76e0069afadbcb6b65","included_native_beats":len(refs),"participants":refs.record.nunique(),
           "head_seed_direction_event_rows":len(ev),"independent_lead_II_anchored_complete_cases":rebuilt,"parent_spans_target_midpoint_exceptions":span_shift,
           "native_12_envelopes_match_parent":True,"exposures_are_opposite_target_references":True,
           "counts_and_descriptive_statistics_verified_from_events":True,"checked_summary_ids":len(summary),
           "input_derived_reference_sha256":digest(parent/"outputs"/"reference.json")}
    (out/"qa.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--parent",required=True,type=Path);p.add_argument("--out",type=Path,default=HERE/"results");args=p.parse_args()
    verify(args.parent,args.out)
