"""External single-lead QRS-duration analysis on saved RDB probability-argmax masks."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd


GROUPS={"Matched_heads":("KAN","GELU"),"Equilibrium":("DEQ","Twin8")}
SEEDS=(1,2)
ARMS=["KAN","GELU","DEQ","Twin8"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def reference_intervals(raw,lo,hi):
    ref={}
    seen=set()
    flags=dict(raw_qrs=0,invalid=0,zero_length=0,edge=0,outside_span=0,duplicate=0)
    for j,row in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
        if row["TYPE"]!="1": continue
        flags["raw_qrs"]+=1
        s,e=float(row["START"]),float(row["END"])
        if not (np.isfinite(s) and np.isfinite(e) and 0<=s<=e<=4999):
            flags["invalid"]+=1
        elif s==e:
            flags["zero_length"]+=1
        elif s==0 or e==4999:
            flags["edge"]+=1
        elif (s,e) in seen:
            flags["duplicate"]+=1
        elif not lo<=s+e<=hi:
            flags["outside_span"]+=1
        else:
            seen.add((s,e))
            ref[j]=(s*2,e*2)
    return ref,flags


def predicted_intervals(mask,lo,hi):
    change=np.diff(np.r_[False,mask==2,False].astype(np.int8))
    starts=np.flatnonzero(change==1)
    ends=np.flatnonzero(change==-1)-1
    segs=[]
    for s,e in zip(starts,ends,strict=True):
        if segs and s-segs[-1][1]-1<5:
            segs[-1]=(segs[-1][0],int(e))
        else: segs.append((int(s),int(e)))
    return [(4*s,4*e) for s,e in segs if e-s+1>=10 and s>0 and e<len(mask)-1 and lo<=2*(s+e)<=hi]


def match(ref,pred):
    candidates=[]
    for ri,(a,b) in ref.items():
        for pi,(c,d) in enumerate(pred):
            mid=abs((a+b-c-d)/2)
            if min(b,d)>max(a,c) and mid<=150:
                candidates.append((mid,ri,pi))
    matched={}
    used=set()
    for mid,ri,pi in sorted(candidates):
        if ri not in matched and pi not in used:
            matched[ri]=pi
            used.add(pi)
    return matched


def collect(args):
    out=args.out
    ids=json.loads((args.heads/"signal_order.json").read_text())
    twin_ids=json.loads((args.equilibrium/"signal_order.json").read_text())
    assert ids==twin_ids and len(ids)==len(set(ids))==2398
    meta=pd.read_csv(args.heads/"cohort.csv").query("status=='included'").set_index("record").loc[ids]
    protocol=Path(__file__).resolve().parent/"protocol.txt"
    inputs={"protocol":sha(protocol),"dataset.zip":sha(args.rdb),
            "heads/cohort.csv":sha(args.heads/"cohort.csv"),"equilibrium/cohort.csv":sha(args.equilibrium/"cohort.csv"),
            "heads/signal_order.json":sha(args.heads/"signal_order.json"),"equilibrium/signal_order.json":sha(args.equilibrium/"signal_order.json")}
    assert inputs["heads/cohort.csv"]==inputs["equilibrium/cohort.csv"]
    paths={(arm,seed):(args.heads if arm in GROUPS["Matched_heads"] else args.equilibrium)/f"masks_{arm}_{seed}.npy" for arm in ARMS for seed in SEEDS}
    masks={k:np.load(f,mmap_mode="r",allow_pickle=False) for k,f in paths.items()}
    assert all(x.shape==(len(ids),2500) for x in masks.values())
    inputs.update({f.name:sha(f) for f in paths.values()})
    (out/"inputs_sha256.json").write_text(json.dumps(inputs,indent=2),encoding="utf8")
    account,common_rows,qc_rows=[],[],[]
    head=["record","rhythm","arm","seed","reference_row","reference_on_ms","reference_off_ms","predicted_on_ms","predicted_off_ms","duration_error_ms","four_way_common"]
    with ZipFile(args.rdb) as z, gzip.open(out/"matched_qrs_events.csv.gz","wt",encoding="utf8",newline="") as file:
        writer=csv.writer(file);writer.writerow(head)
        for i,(rid,row) in enumerate(meta.iterrows()):
            lo,hi=row.score_start_ms,row.score_end_ms
            ref,flags=reference_intervals(z.read(f"ann_txt/{rid}.ii.txt"),lo,hi)
            qc_rows.append(dict(record=rid,rhythm=row.rhythm,**flags,eligible=len(ref)))
            for group,arms in GROUPS.items():
                pairs={};preds={}
                for arm in arms:
                    for seed in SEEDS:
                        pred=predicted_intervals(masks[arm,seed][i],lo,hi)
                        preds[arm,seed]=pred
                        pairs[arm,seed]=match(ref,pred)
                common=set.intersection(*(set(matches) for matches in pairs.values()))
                seed_pair_n={seed:len(set(pairs[arms[0],seed])&set(pairs[arms[1],seed])) for seed in SEEDS}
                for arm in arms:
                    for seed in SEEDS:
                        pred,links=preds[arm,seed],pairs[arm,seed]
                        total_abs=total_sig=all_abs=all_sig=0.
                        for ri,pi in sorted(links.items()):
                            a,b=ref[ri];c,d=pred[pi];err=(d-c)-(b-a)
                            all_abs+=abs(err);all_sig+=err
                            if ri in common:
                                total_abs+=abs(err);total_sig+=err
                            writer.writerow([rid,row.rhythm,arm,seed,ri,a,b,c,d,err,int(ri in common)])
                        account.append(dict(record=rid,rhythm=row.rhythm,group=group,arm=arm,seed=seed,
                                references=len(ref),predictions=len(pred),matches=len(links),unmatched_predictions=len(pred)-len(links),
                                missed_references=len(ref)-len(links),shared_in_seed=seed_pair_n[seed],
                                all_matched_absolute_error_sum_ms=all_abs,all_matched_signed_error_sum_ms=all_sig))
                        common_rows.append(dict(record=rid,rhythm=row.rhythm,group=group,arm=arm,seed=seed,common=len(common),
                                references=len(ref),abs_sum_ms=total_abs,signed_sum_ms=total_sig))
    pd.DataFrame(account).to_csv(out/"record_coverage.csv",index=False)
    pd.DataFrame(common_rows).to_csv(out/"record_duration.csv",index=False)
    pd.DataFrame(qc_rows).to_csv(out/"annotation_qc.csv",index=False)


def estimates(s):
    # trailing axes arm × seed × statistic, with common counts identical across all four cells
    n=s[...,0]
    a=(s[...,1]/n).mean(axis=-1)
    b=(s[...,2]/n).mean(axis=-1)
    return np.stack([a[...,0],a[...,1],a[...,0]-a[...,1],b[...,0],b[...,1],b[...,0]-b[...,1]],axis=-1)


def bootstrap(x,strata):
    n=len(x);a=np.ascontiguousarray(x.reshape(n,-1),dtype="float64")
    rng=np.random.default_rng(20261125)
    levels=[np.flatnonzero(strata==k) for k in sorted(set(strata))]
    metrics=[]
    for _ in range(20):
        w=np.zeros((100,n),dtype="float64")
        for ix in levels:
            w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=100)
        sums=(w@a).reshape((100,)+x.shape[1:])
        metrics.append(estimates(sums))
    return np.quantile(np.concatenate(metrics),[.025,.975],axis=0)


def summarize(out):
    coverage=pd.read_csv(out/"record_coverage.csv")
    duration=pd.read_csv(out/"record_duration.csv")
    qc=pd.read_csv(out/"annotation_qc.csv").set_index("record").sort_index()
    ids=qc.index.tolist()
    assert (qc.raw_qrs==qc[["invalid","zero_length","edge","outside_span","duplicate","eligible"]].sum(axis=1)).all()
    check=coverage.groupby("record").references.agg(["min","max"]).loc[ids]
    assert np.array_equal(check["min"],qc.eligible) and (check["min"]==check["max"]).all()
    c=coverage.groupby(["group","arm","seed"]).sum(numeric_only=True).reset_index()
    c["reference_coverage_pct"]=c.matches/c.references*100
    c["prediction_match_pct"]=c.matches/c.predictions*100
    c["all_matched_mae_ms"]=c.all_matched_absolute_error_sum_ms/c.matches
    c["all_matched_bias_ms"]=c.all_matched_signed_error_sum_ms/c.matches
    c.to_csv(out/"coverage.csv",index=False)
    summ=[];sub=[]
    norms=["mae_first","mae_second","mae_first_minus_second","bias_first","bias_second","bias_first_minus_second"]
    for group,arms in GROUPS.items():
        x=duration[duration.group==group].set_index(["record","arm","seed"]).loc[
            pd.MultiIndex.from_product([ids,arms,SEEDS],names=["record","arm","seed"]),["common","abs_sum_ms","signed_sum_ms"]]
        x=x.to_numpy().reshape(len(ids),2,2,3)
        assert (x[:,:,:,0]==x[:,0,0,0][:,None,None]).all()
        est=estimates(x.sum(axis=0))
        ci=bootstrap(x,qc.rhythm.to_numpy())
        n=int(x[:,0,0,0].sum())
        for j,norm in enumerate(norms):
            summ.append(dict(group=group,first=arms[0],second=arms[1],estimand=norm,estimate_ms=est[j],lo_ms=ci[0,j],
                       hi_ms=ci[1,j],reference_intervals=int(qc.eligible.sum()),common_intervals=n,
                       common_coverage_pct=100*n/qc.eligible.sum(),common_records=int((x[:,0,0,0]>0).sum())))
        for rhythm in sorted(set(qc.rhythm)):
            sel=(qc.rhythm==rhythm).to_numpy()
            xx=x[sel].sum(axis=0);q=qc[sel]
            for field,val in zip(norms,estimates(xx),strict=True):
                sub.append(dict(group=group,rhythm=rhythm,estimand=field,estimate_ms=val,records=len(q),references=int(q.eligible.sum()),common=int(xx[0,0,0])))
    totals=pd.DataFrame(summ)
    totals.to_csv(out/"duration_summary.csv",index=False)
    pd.DataFrame(sub).to_csv(out/"rhythm_duration.csv",index=False)
    by_seed=duration.groupby(["group","arm","seed"])[["references","common","abs_sum_ms","signed_sum_ms"]].sum()
    by_seed["mae_ms"]=by_seed.abs_sum_ms/by_seed.common
    by_seed["bias_ms"]=by_seed.signed_sum_ms/by_seed.common
    by_seed.to_csv(out/"duration_by_seed.csv")
    print(totals.round(3).to_string(index=False),flush=True)
    print(c[["arm","seed","references","predictions","matches","reference_coverage_pct","prediction_match_pct"]].round(3).to_string(index=False),flush=True)
    print(qc.select_dtypes("number").sum().to_dict(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rdb",type=Path,required=True)
    p.add_argument("--heads",type=Path,required=True)
    p.add_argument("--equilibrium",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    collect(a);summarize(a.out)
