"""Summarize paired counts from rdb_external.py; uncertainty conditions on fitted models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


POINTS = ["P_on","P_off","QRS_on","QRS_off","T_on","T_off"]
ARMS = ["KAN","GELU"]
COLS = ["tp","fp","fn","abs_error_sum_ms","error_sum_ms"]
ORDER = ["arm","seed","endpoint"]


def divide(a,b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan), where=b!=0)


def metrics(s):
    tp, fp, fn, ae, err = np.moveaxis(s,-1,0)
    return (divide(2*tp,2*tp+fp+fn), divide(tp,tp+fn), divide(tp,tp+fp), divide(ae,tp), divide(err,tp))


def score_arr(df,ids,arms=ARMS):
    index = pd.MultiIndex.from_product([ids,arms,[1,2],POINTS], names=["record",*ORDER])
    return df.set_index(["record",*ORDER]).loc[index, COLS].to_numpy().reshape(len(ids),2,2,6,5)


def seed_mean_f1(s):
    f1 = metrics(s)[0]
    per_endpoint = f1.mean(axis=-2)
    macro = per_endpoint.mean(axis=-1,keepdims=True)
    return np.concatenate([per_endpoint,macro],axis=-1)


def interval_arrays(counts, strata, rng):
    # (records, heads, fitted seeds, endpoints, 3 counts). Resampling an entire
    # row jointly preserves every within-record dependency.
    n = len(counts)
    flat = np.ascontiguousarray(counts.reshape(n,-1), dtype="float64")
    levels = [np.flatnonzero(strata==lvl) for lvl in sorted(set(strata))]
    draws = []
    for k in range(20):
        weights = np.zeros((100,n),dtype="float64")
        for pos in levels:
            weights[:,pos] = rng.multinomial(len(pos), np.full(len(pos),1/len(pos)), size=100)
        totals = (weights@flat).reshape(100, 2,2,6,5)
        means = seed_mean_f1(totals)
        draws.append(np.concatenate([means,(means[:,0,:]-means[:,1,:])[:,None,:]],axis=1))
    return np.quantile(np.concatenate(draws),[0.025,0.975],axis=0)*100


def main(folder):
    df = pd.read_csv(folder/"record_scores.csv")
    meta = pd.read_csv(folder/"cohort.csv")
    kept = meta[meta.status=="included"].sort_values("record").set_index("record")
    ids = kept.index.tolist()
    df2 = df[df.tolerance_ms==150]
    for seed in (1,2):
        subset = df2[df2.seed==seed]
        r1 = subset[subset.arm=="KAN"].sort_values(["record","endpoint"])
        r2 = subset[subset.arm=="GELU"].sort_values(["record","endpoint"])
        assert np.array_equal((r1.tp+r1.fn).to_numpy(),(r2.tp+r2.fn).to_numpy())
        for ep in POINTS:
            block = r1[r1.endpoint==ep].set_index("record").reindex(ids)
            assert np.array_equal((block.tp+block.fn).to_numpy(),kept[ep].to_numpy())
    assert len(df)==len(ids)*2*2*2*len(POINTS)
    summary, by_seed, by_rhythm = [], [], []
    counts_frame = df.groupby(["tolerance_ms",*ORDER])[COLS].sum().reset_index()
    counts_frame.to_csv(folder/"aggregate_counts.csv",index=False)
    for tol in (150,40):
        a = score_arr(df[df.tolerance_ms==tol],ids)
        all_counts = a.sum(axis=0)
        f1, sens, ppv, mae, err = metrics(all_counts)
        for h,arm in enumerate(ARMS):
            for j,seed in enumerate([1,2]):
                for p,key in enumerate(POINTS):
                    by_seed.append(dict(tolerance_ms=tol,arm=arm,seed=seed,endpoint=key,
                        f1_pct=100*f1[h,j,p],sensitivity_pct=100*sens[h,j,p],ppv_pct=100*ppv[h,j,p],
                        matched_mae_ms=mae[h,j,p],matched_bias_ms=err[h,j,p]))
                by_seed.append(dict(tolerance_ms=tol,arm=arm,seed=seed,endpoint="macro",f1_pct=100*f1[h,j].mean()))
        pooled = seed_mean_f1(all_counts)*100
        boot = interval_arrays(a,kept.rhythm.to_numpy(),np.random.default_rng(20261125))
        for k,key in enumerate(POINTS+["macro"]):
            summary.append(dict(tolerance_ms=tol,endpoint=key,kan_f1=pooled[0,k],kan_lo=boot[0,0,k],kan_hi=boot[1,0,k],
                                gelu_f1=pooled[1,k],gelu_lo=boot[0,1,k],gelu_hi=boot[1,1,k],
                                kan_minus_gelu_pp=pooled[0,k]-pooled[1,k],delta_lo=boot[0,2,k],delta_hi=boot[1,2,k]))
        for rhythm in kept.rhythm.unique():
            sub = kept.rhythm==rhythm
            rtot = a[sub.to_numpy()].sum(axis=0)
            rf1 = metrics(rtot)[0]
            for h,arm in enumerate(ARMS):
                for p,key in enumerate(POINTS):
                    reference_count = int(rtot[h,0,p,0]+rtot[h,0,p,2])
                    by_rhythm.append(dict(tolerance_ms=tol,rhythm=rhythm,record_count=int(sub.sum()),
                                          arm=arm,endpoint=key,reference_count=reference_count,
                                          f1_pct=(100*rf1[h,:,p].mean()) if reference_count else np.nan,
                                          mean_unmatched_predictions=rtot[h,:,p,1].mean()))
    summ = pd.DataFrame(summary)
    summ.to_csv(folder/"comparison.csv",index=False)
    pd.DataFrame(by_seed).to_csv(folder/"seed_performance.csv",index=False)
    pd.DataFrame(by_rhythm).to_csv(folder/"rhythm_performance.csv",index=False)
    af = kept.index[kept.rhythm.isin(["AF","AFIB"])]
    dur = ((kept.loc[af,"score_end_ms"]-kept.loc[af,"score_start_ms"])/60000).sum()
    p = df2[df2.record.isin(af)&(df2.endpoint=="P_on")].groupby(["arm","seed"])[["fp","fn","tp"]].sum().reset_index()
    p["annotated_minutes"]=dur
    p["unmatched_P_onsets_per_min"]=p.fp/dur
    p.to_csv(folder/"af_flutter_p_detections.csv",index=False)
    summary_qc = dict(
        release_rows=len(meta),included=len(kept),statuses=meta.status.value_counts().to_dict(),
        excluded=meta[meta.status!="included"][["record","source","status"]].to_dict(orient="records"),
        rhythm_counts=kept.rhythm.value_counts().to_dict(),qc_counts={col:int(kept[col].sum()) for col in
            ["annotation_rows","invalid_interval","zero_length","outside","censored_boundary","duplicate_mark",*POINTS]},
        identical_files_all_12=int((kept.distinct_lead_annotation_files==1).sum()),
        distinct_annotation_files_median=float(kept.distinct_lead_annotation_files.median()),
        score_minutes=((kept.score_end_ms-kept.score_start_ms)/60000).sum(),
        median_score_seconds=float(((kept.score_end_ms-kept.score_start_ms)/1000).median()),
        records_without_reference_P=int((kept.P_on==0).sum()),
        af_flutter_records=len(af),af_flutter_reference_P_on=int(kept.loc[af,"P_on"].sum()))
    (folder/"qc_summary.json").write_text(json.dumps(summary_qc,indent=2),encoding="utf-8")
    print(summ[summ.endpoint=="macro"].round(3).to_string(index=False),flush=True)
    print(json.dumps(summary_qc,indent=1),flush=True)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
    main(ap.parse_args().out)
