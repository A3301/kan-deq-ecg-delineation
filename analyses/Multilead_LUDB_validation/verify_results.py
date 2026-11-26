"""Independent checks against native annotations, saved event rows and archived 41-record results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEADS = "i ii iii avr avl avf v1 v2 v3 v4 v5 v6".split()
SCOPES = LEADS + ["global8", "global12"]
GROUP = ["record", "scope", "head", "seed"]
NUM = "reference predicted matched missed extras correct10 correct20 correct30 sum_error sum_abs common common_sum_error common_sum_abs".split()


def duration_check(output):
    events = pd.read_csv(output / "paired_events.csv.gz").rename(columns={"arm": "head"})
    q = pd.read_csv(output / "qrs_counts.csv").set_index(GROUP).sort_index()
    ref = events.reference_index.notna()
    pred = events.pred_index.notna()
    matched = ref & pred
    keys = ["record", "scope", "reference_index"]
    assert not events[ref].duplicated(GROUP + ["reference_index"]).any()
    assert not events[pred].duplicated(GROUP + ["pred_index"]).any()
    error = (events.pred_off_ms - events.pred_on_ms) - (events.ref_off_ms - events.ref_on_ms)
    amt = events[matched].groupby(keys).size()
    common = pd.MultiIndex.from_frame(events[keys]).isin(amt[amt == 6].index)
    assert np.all(matched[common])
    events["reference"] = ref
    events["predicted"] = pred
    events["matched"] = matched
    events["missed"] = ref & ~pred
    events["extras"] = pred & ~ref
    events["sum_error"] = error.fillna(0)
    events["sum_abs"] = error.abs().fillna(0)
    for cut in [10, 20, 30]:
        events[f"correct{cut}"] = error.abs().le(cut)
    events["common"] = common
    events["common_sum_error"] = error.where(common,0).fillna(0)
    events["common_sum_abs"] = error.abs().where(common,0).fillna(0)
    actual = events.groupby(GROUP)[NUM].sum().reindex(q.index,fill_value=0).astype("int64")
    pd.testing.assert_frame_equal(q[NUM],actual)
    summary = pd.read_csv(output / "duration_summary.csv")
    for row in summary[(summary.subset=="all200") & (summary.metric=="allref_20ms_pct")].itertuples():
        scope = LEADS if row.scope=="all_leads" else [row.scope]
        counts = events[events.scope.isin(scope)].groupby(["head","seed"])[["correct20","reference"]].sum()
        pcts = (100*counts.correct20/counts.reference).groupby("head").mean()
        target = pcts.KAN - pcts.GELU if row.arm=="KAN_minus_GELU" else pcts[row.arm]
        np.testing.assert_allclose(row.estimate,target,atol=1e-12)
    # Recompute candidate distances with a matrix, independently of the scorer.
    pairing_groups = 0
    for _, rows in events[events.record.isin([1,25,50,75,100,125,150,175,200])].groupby(GROUP):
        ra = rows[rows.reference_index.notna()].sort_values("reference_index")
        pa = rows[rows.pred_index.notna()].sort_values("pred_index")
        aa = ra[["ref_on_ms","ref_off_ms"]].to_numpy()
        bb = pa[["pred_on_ms","pred_off_ms"]].to_numpy()
        distances = np.abs(aa.mean(1)[:,None]-bb.mean(1)[None,:])
        overlap = np.minimum(aa[:,1,None],bb[None,:,1])-np.maximum(aa[:,0,None],bb[None,:,0])
        distances[(overlap<=0)|(distances>150)] = np.inf
        links = set()
        while distances.size and np.isfinite(distances.min()):
            i,j = np.unravel_index(np.argmin(distances),distances.shape)
            links.add((int(ra.iloc[i].reference_index),int(pa.iloc[j].pred_index)))
            distances[i,:] = np.inf
            distances[:,j] = np.inf
        paired = rows[rows.reference_index.notna() & rows.pred_index.notna()]
        assert links == set(zip(paired.reference_index,paired.pred_index))
        pairing_groups += 1
    return events, {"event_rows":len(events),"independently_verified_count_rows":len(q),"integer_fields_per_row":len(NUM),"event_groups_with_independent_matrix_matching":pairing_groups}


def sources_check(work,archive_counts,training_signals):
    out = work/"outputs"
    key = ["record","head","seed","tol_ms","landmark"]
    archived = pd.read_csv(archive_counts).set_index(key).sort_index()
    new = pd.read_csv(out/"boundary_counts.csv").query("subset=='original41'").set_index(key).sort_index()
    assert len(archived)==2952
    pd.testing.assert_frame_equal(archived[["tp","fn","fp"]],new[["tp","fn","fp"]])
    result = {"identical_archived_41_boundary_rows":len(archived)}
    if training_signals is not None:
        def fingerprint(sig):
            return hashlib.sha256(np.round(sig,4).astype("<f4").tobytes()).hexdigest()
        rdb = np.load(training_signals,mmap_mode="r")
        source = {fingerprint(s) for s in rdb}
        local = np.load(work/"waves.npy",mmap_mode="r")
        matches = [(n+1,LEADS[j]) for n,sig in enumerate(local) for j,lead in enumerate(sig) if fingerprint(lead) in source]
        assert not matches, matches
        result.update({"rdb_records_fingerprinted":len(rdb),"rdb_unique_rounded_waveforms":len(source),"ludb_rounded_waveforms_fingerprinted":200*12,"exact_normalised_rounded_waveform_matches":len(matches)})
    return result


def native_checks(work,ev):
    if hasattr(pd.options.future,"infer_string"):
        pd.options.future.infer_string=False
    import wfdb
    subjects = json.loads((work/"outputs"/"reference.json").read_text())
    checked = 0
    for person in subjects:
        for lead in LEADS:
            ann = wfdb.rdann(str(work/"raw"/person["record"]),lead)
            times = ann.sample*2
            positions = [m.start() for m in re.finditer(r"\(N\)","".join(ann.symbol))]
            elapsed = [[int(times[k]),int(times[k+2])] for k in positions if 0<times[k]<times[k+2]<9998]
            assert elapsed == person["leads"][lead]["qrs"]
            checked+=1
    # Alternate beat identity for uncluttered 12-lead examples: anchor lead II,
    # match each other lead's nearest midpoint within 120 ms, and reconstruct envelopes.
    examples = []
    for pos in [0,24,49,74,99,124,149,174,199]:
        person = subjects[pos]
        data = person["leads"]
        bounds = max(data[l]["span"][0] for l in LEADS), min(data[l]["span"][1] for l in LEADS)
        actual = ev[(ev.record==int(person["record"]))&(ev.scope=="global12")&(ev["head"]=="KAN")&(ev.seed==1)&ev.reference_index.notna()]
        envelope = set(zip(actual.ref_on_ms,actual.ref_off_ms))
        eligible = 0
        for p in data["ii"]["qrs"]:
            ctr=median(p); coupled=[]
            for l in LEADS:
                nearby=[x for x in data[l]["qrs"] if abs(median(x)-ctr)<=120]
                if len(nearby)!=1:break
                coupled.append(nearby[0])
            if len(coupled)!=12 or max(map(median,coupled))-min(map(median,coupled))>120:continue
            pair = min(x[0] for x in coupled),max(x[1] for x in coupled)
            if bounds[0]<=median(pair)<=bounds[1]:
                assert pair in envelope,(person["record"],pair)
                eligible+=1
        examples.append({"record":person["record"],"independently_identified_complete_beats":eligible})
    return {"native_lead_files_regex_verified":checked,"global_interval_examples":examples,
            "global_intervals_independently_reconstructed":sum(r["independently_identified_complete_beats"] for r in examples)}


def main(args):
    events,stats = duration_check(args.work/"outputs")
    stats.update(sources_check(args.work,args.original_counts,args.rdb_waveforms))
    if not args.skip_native:
        stats.update(native_checks(args.work,events))
    (args.work/"outputs"/"qa_verified.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    print(json.dumps(stats),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work",type=Path,default=HERE)
    parser.add_argument("--original-counts",type=Path,default=HERE/"inputs"/"original_41_landmark_counts.csv")
    parser.add_argument("--rdb-waveforms",type=Path)
    parser.add_argument("--skip-native",action="store_true")
    main(parser.parse_args())
