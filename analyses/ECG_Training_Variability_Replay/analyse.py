"""Recalculate paired, record-clustered estimates from per-subject TP, FN and FP."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = {
    "unet": "U-Net", "cnn_bilstm": "CNN–BiLSTM", "transformer": "Transformer",
    "unet_kan": "U-Net–KAN", "convkan": "ConvKAN", "deq": "Implicit DEQ",
    "deq_unrolled": "Unrolled DEQ (8 steps)", "explicit_stack": "Explicit stack",
}
MODELS = list(NAMES)
LANDMARKS = ["P onset", "P offset", "QRS onset", "QRS offset", "T onset", "T offset"]
COMPARISONS = [
    ("unet_kan", "unet"), ("unet_kan", "cnn_bilstm"), ("convkan", "unet"),
    ("convkan", "cnn_bilstm"), ("deq", "deq_unrolled"), ("deq", "explicit_stack"),
]

def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def macro_f1(counts):
    tp, fn, fp = counts[..., 0], counts[..., 1], counts[..., 2]
    den = 2*tp + fn + fp
    assert (den > 0).all()
    return (200 * tp / den).mean(-1)

def analyse(tag,source=None,out=None):
    source = source or ROOT / "results"
    out = out or ROOT / "results"
    out.mkdir(exist_ok=True,parents=True)
    all_counts, long = [], []
    participants = sorted(json.loads((ROOT/"splits.json").read_text())["test"], key=int)
    for model in MODELS:
        seeds = []
        for seed in range(1, 4):
            res = np.load(source/f"{tag}_{model}_s{seed}.npz", allow_pickle=False)
            counts = res["counts"]
            assert counts.dtype.kind == "i" and (counts >= 0).all()
            assert counts.shape == (2, 41, 6, 3) and list(res["subjects"]) == participants
            assert list(res["folds"]) == (list(range(5)) if tag in ("ensemble","harmonized_ensemble") else [0])
            seeds.append(counts)
            for k, tol in enumerate([150, 40]):
                for i, subject in enumerate(participants):
                    for j, landmark in enumerate(LANDMARKS):
                        tp, fn, fp = map(int, counts[k,i,j])
                        long.append(dict(architecture=model, seed=seed, tolerance_ms=tol, subject=subject,
                                         landmark=landmark, tp=tp, fn=fn, fp=fp))
        all_counts.append(seeds)
    values = np.asarray(all_counts)
    references = values[..., 0] + values[..., 1]
    assert np.all(references == references[0, 0, 0])
    assert np.all(values[:,:,0,:,:,0] >= values[:,:,1,:,:,0])
    write(out/f"{tag}_subject_counts.csv", long)
    totals = values.sum(3)
    score = macro_f1(totals)
    ranks = 1 + np.argsort(np.argsort(-score, axis=0), axis=0)
    tables, contrasts = [], []
    rng = np.random.default_rng(412026)
    ids = rng.integers(0, len(participants), (10000,len(participants)))
    for k, tolerance in enumerate([150,40]):
        for mi, model in enumerate(MODELS):
            sc=score[mi,:,k]
            tables.append(dict(architecture=model,label=NAMES[model],tolerance_ms=tolerance,
                               **{f"seed_{s+1}":sc[s] for s in range(3)},
                               mean=sc.mean(),sample_sd=sc.std(ddof=1),minimum=sc.min(),maximum=sc.max(),
                               **{f"rank_{s+1}":int(ranks[mi,s,k]) for s in range(3)}))
        for a,b in COMPARISONS:
            ia=MODELS.index(a); ib=MODELS.index(b)
            diff=score[ia,:,k] - score[ib,:,k]
            boot=[]
            for ids_chunk in np.array_split(ids,50):
                def estimate(ix):
                    # Each sampled participant carries all six landmark vectors across all leads.
                    return macro_f1(np.take(values[ix,:,k], ids_chunk, axis=1).sum(axis=2))
                boot.append(estimate(ia)-estimate(ib))
            dist=np.concatenate(boot,axis=1)
            for j, level in enumerate(["seed_1", "seed_2", "seed_3", "mean_of_three"]):
                estimate=diff[j] if j<3 else diff.mean()
                lower,upper=np.quantile(dist[j] if j<3 else dist.mean(0),[.025,.975])
                contrasts.append(dict(tolerance_ms=tolerance,model_a=a,model_b=b,level=level,
                                      difference_pp=estimate,lower_95=lower,upper_95=upper))
    write(out/f"{tag}_scores.csv", tables)
    write(out/f"{tag}_contrasts.csv", contrasts)
    assert len(long)==11808 and len(tables)==16 and len(contrasts)==48
    qa={"panel":tag,"participants":len(participants),"lead_recordings":12*len(participants),
        "patient_ids":participants,"models":len(MODELS),"training_seeds":3,"npz_files":24,
        "six_reference_counts":list(map(int,references[0,0,0].sum(0))),"subject_rows":len(long),
        "identical_reference_counts_across_all_fits_and_tolerances":True,"threshold_monotonicity":True,
        "bootstrap_draws":len(ids),"bootstrap_rng_seed":412026,"bootstrap_unit":"participant",
        "estimand":"percentage-point difference in six equally weighted landmark F1 scores"}
    (out/f"{tag}_qa.json").write_text(json.dumps(qa,indent=2))
    return tables,contrasts

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("tag",choices=["ensemble","fold0","homogeneous","harmonized_ensemble"])
    ap.add_argument("--source",type=Path);ap.add_argument("--output",type=Path)
    args=ap.parse_args();analyse(args.tag,args.source,args.output)
