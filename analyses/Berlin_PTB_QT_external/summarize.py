"""Paired patient-cluster bootstrap for the six fixed fits; no ensembling of predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
N_BOOT = 10000

def main():
    frame = pd.read_csv(OUT/"per_record.csv")
    assert len(frame)==548*6 and not frame.duplicated(["id","head","seed"]).any()
    frame = frame.sort_values(["id","head","seed"])
    assert frame.groupby("id").size().eq(6).all() and frame.patient.nunique()==289
    order = ["KAN","GELU"]
    index = pd.MultiIndex.from_product([order,[1,2,3]],names=["head","seed"])
    def pivot(values):
        return frame.assign(value=values).pivot(index="id",columns=["head","seed"],values="value").loc[:,index]
    returned = pivot(frame.status.eq("returned"))
    apt = frame.predicted_t_ms-frame.predicted_q_ms-(frame.reference_t_ms-frame.reference_q_ms)
    eq, et = frame.predicted_q_ms-frame.reference_q_ms, frame.predicted_t_ms-frame.reference_t_ms
    patient = pd.Index([r.partition("/")[0] for r in returned.index],name="patient")
    subj, group = np.unique(patient,return_inverse=True)
    rng = np.random.default_rng(20261125)
    draws = rng.multinomial(len(subj),np.full(len(subj),1/len(subj)),size=N_BOOT)
    draws = np.vstack([np.ones(len(subj),int),draws]).astype(float)
    common = returned.all(axis=1).to_numpy()
    all_recs = np.ones(len(common),bool)
    def grouped(data):
        totals = np.zeros((len(subj),data.shape[1]),dtype=float)
        np.add.at(totals,group,data)
        return totals
    metrics = {}
    for tolerance in [40,20,60]:
        metrics[f"QT_success_{tolerance}ms"]=(pivot(apt.abs().le(tolerance)), all_recs, "percent",False)
    metrics["QT_success_40ms_equal_patient"]=(pivot(apt.abs().le(40)), all_recs,"percent",True)
    metrics["QT_returned"]=(returned,all_recs,"percent",False)
    for tolerance in [40,150]:
        metrics[f"both_landmarks_within_{tolerance}ms"]=(pivot(eq.abs().le(tolerance)&et.abs().le(tolerance)),
                                                        all_recs,"percent",False)
    for name,values in dict(QT_MAE=apt.abs(),QT_bias=apt,Q_onset_MAE=eq.abs(),T_end_MAE=et.abs()).items():
        metrics[name+"_common"]=(pivot(values),common,"ms",False)
    metrics["QT_MAE_each_returned"]=(pivot(apt.abs()),returned.to_numpy(),"ms",False)
    estimates, seeds = [], []
    for name,(values,include,unit,equal_patient) in metrics.items():
        mask = np.broadcast_to(include[:,None],values.shape) if include.ndim==1 else include
        v = values.to_numpy(dtype=float)
        assert np.isfinite(v[mask]).all()
        sums = grouped(np.where(mask,v,0))
        denom = grouped(mask.astype(float))
        if equal_patient:
            assert np.all(denom>0)
            sums,denom = sums/denom,np.ones_like(denom)
        results = draws@sums/(draws@denom)
        if unit=="percent":
            results *= 100
        def estimate(label,series):
            low, high = np.quantile(series[1:],[.025,.975])
            return {label:series[0],label+"_lo":low,label+"_hi":high}
        def row(a,b):
            return dict(metric=name,unit=unit,**estimate("KAN",a),**estimate("GELU",b),**estimate("difference",a-b))
        estimates.append(row(results[:,:3].mean(axis=1),results[:,3:].mean(axis=1)))
        for seed in range(3):
            seeds.append(dict(seed=seed+1,**row(results[:,seed],results[:,3+seed])))
    pd.DataFrame(estimates).to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(seeds).to_csv(OUT/"by_seed.csv",index=False)
    pd.DataFrame(dict(id=returned.index,all_six_returned=common)).to_csv(OUT/"common_set.csv",index=False)
    counts = []
    for (head,seed),g in frame.groupby(["head","seed"]):
        diff = g.predicted_t_ms-g.predicted_q_ms-(g.reference_t_ms-g.reference_q_ms)
        counts.append(dict(head=head,seed=seed,total=len(g),returned=g.status.eq("returned").sum(),
                           qt_success_40ms=diff.abs().le(40).sum(),
                           qrs_missed=g.predicted_q_ms.isna().sum(),
                           t_missing_after_q=(g.predicted_q_ms.notna()&g.predicted_t_ms.isna()).sum()))
    pd.DataFrame(counts).to_csv(OUT/"counts.csv",index=False)
    ref = pd.read_csv(ROOT/"source"/"manual_reference.csv").set_index("id").loc[returned.index]
    q = ref[[f"q{x}" for x in range(1,6)]].to_numpy()
    t = ref[[f"t{x}" for x in range(1,6)]].to_numpy()
    assert (t>q).all() and (q>0).all()
    reader_spread = np.ptp(t-q,axis=1)
    pd.DataFrame(dict(id=ref.index,reader_QT_range_ms=reader_spread)).to_csv(OUT/"readers.csv",index=False)
    notes=dict(eligible_records=len(common),eligible_patients=len(subj),released_records=549,
               released_patients=len(pd.read_csv(ROOT/"source"/"cohort.csv").patient.unique()),
               common_records=int(common.sum()),common_patients=len(set(patient[common])),
               eligible_records_per_patient=pd.Series(group).value_counts().value_counts().sort_index().to_dict(),
               reference_QT_ms_quantiles=dict(zip(["min","q25","median","q75","max"],
                                                  np.quantile(ref["T"]-ref.Q,[0,.25,.5,.75,1]))),
               final_reader_range_quartiles_ms=np.quantile(reader_spread,[0,.25,.5,.75,1]).tolist(),
               final_reader_range_mean_ms=reader_spread.mean(),bootstrap_draws=N_BOOT,bootstrap_seed=20261125)
    (OUT/"denominators.json").write_text(json.dumps(notes,indent=2))
    print(notes)
    print(pd.DataFrame(estimates).round(3).to_string(index=False))

if __name__=="__main__":
    main()
