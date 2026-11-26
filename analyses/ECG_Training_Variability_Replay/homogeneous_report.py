"""Explain the uniform GPU sensitivity panel alongside both the archive-mixed and full analyses."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from analyse import NAMES,write
from report import read

H=Path(__file__).resolve().parent
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def make():
    res=H/"results"
    now=read(res/"homogeneous_scores.csv"); old=read(res/"fold0_scores.csv")
    oo={(q["architecture"],q["tolerance_ms"]):q for q in old}
    lr=json.loads((H/"chosen_lr.json").read_text())
    prior={q["name"]:q for q in read(H/"source/archived_timestamps.csv")}
    baseline_manifest=[];points=[]
    for row in now:
        m=row["architecture"];tol=row["tolerance_ms"];p_old=oo[(m,tol)]
        for seed in (1,2,3):
            name=f"lr{lr[m]}_{m}_f0_s{seed}.pt";oldp=H/"weights"/name
            fitp=H/"homogeneous_weights"/name if name in prior else oldp
            curr=np.load(res/f"homogeneous_{m}_s{seed}.npz",allow_pickle=False)
            bse=np.load(res/f"fold0_{m}_s{seed}.npz",allow_pickle=False)
            assert curr["weights_sha256"][0]==sha(fitp) and bse["weights_sha256"][0]==sha(oldp)
            f,b=curr["counts"],bse["counts"];assert np.array_equal(f[...,0]+f[...,1],b[...,0]+b[...,1])
            if name not in prior:assert np.array_equal(f,b)
            logdir="homogeneous_logs" if name in prior else "logs"
            log=json.loads((H/logdir/(fitp.stem+".json")).read_text())
            assert log["weight_sha256"]==sha(fitp)
            if tol=="150":
                checkpoint=torch.load(fitp,weights_only=True,map_location="cpu")
                assert all(torch.isfinite(t).all().item() for t in checkpoint["state"].values() if t.is_floating_point())
                baseline_manifest.append(dict(architecture=m,fold=0,seed=seed,all_gpu_weight=fitp.relative_to(H),
                    all_gpu_sha=sha(fitp),additional_refit=name in prior,
                    mixed_panel_sha=sha(oldp),mixed_panel_archive_mtime=prior.get(name,{}).get("local_lastwrite",""),
                    gpu_selected_epoch=log["best_epoch"],gpu_selected_validation=log["best_validation_f1"]))
            points.append(dict(architecture=m,tolerance_ms=tol,seed=seed,extra_refit=name in prior,
                       mixed_panel_f1=p_old[f"seed_{seed}"],uniform_gpu_f1=row[f"seed_{seed}"],
                       change_pp=float(row[f"seed_{seed}"])-float(p_old[f"seed_{seed}"])))
    write(res/"homogeneous_checkpoint_manifest.csv",baseline_manifest)
    write(res/"homogeneous_vs_archived_fold0.csv",points)
    assert len(points)==48 and sum(q["additional_refit"] for q in baseline_manifest)==12
    at150=[q for q in now if q["tolerance_ms"]=="150"]
    ranked=sorted(at150,key=lambda x:float(x["mean"]),reverse=True)
    spread=[float(q["maximum"])-float(q["minimum"]) for q in at150]
    delta=[p["change_pp"] for p in points if p["extra_refit"] and p["tolerance_ms"]=="150"]
    effects=[r for r in read(res/"homogeneous_contrasts.csv") if r["tolerance_ms"]=="150" and r["level"]=="mean_of_three"]
    lines=[]
    for r in effects:
        lines.append(f"{NAMES[r['model_a']]} versus {NAMES[r['model_b']]}: {float(r['difference_pp']):+.2f} ({float(r['lower_95']):+.2f}, {float(r['upper_95']):+.2f}) pp.")
    content=(
        "### Matched hardware and validation in a fixed fold\n\n"
        "The original experiment logged an annotation-edge scoring correction on 23 November 2026. "
        "Archive timestamps placed 35 of 60 pre-existing fold models before this rule change. "
        "To examine training variation under common implementation conditions, we evaluated all eight architectures "
        "using exclusively GPU-trained models for seeds 1–3 on the same development fold (fold 0). We refitted "
        "the 12 models in this panel originally trained on CPU and reused the 12 models already trained "
        "on the GPU. All 24 checkpoints were selected using the same edge-tolerant validation score. "
        "Single-model predictions were evaluated on the same 41 held-out participants. "
        f"Mean six-boundary F1 at 150 ms ranged from {float(ranked[-1]['mean']):.2f}% "
        f"({ranked[-1]['label']}) to {float(ranked[0]['mean']):.2f}% ({ranked[0]['label']}). "
        f"Within-architecture spreads across the three seeds ranged from {min(spread):.2f} to {max(spread):.2f} pp. "
        f"The 12 replacement fits shifted individual scores by {min(delta):+.2f} to {max(delta):+.2f} pp "
        "relative to their same-seed, same-fold archived counterparts. These shifts combine numerical training "
        "differences and any differences in historic checkpoint selection. "
        "This analysis isolates variation among the new fits for one development fold.\n\n"
        "Three-seed average contrasts with 95% participant-cluster bootstrap limits follow. "
        +" ".join(lines)+"\n")
    (H/"Homogeneous_fold_insert.md").write_text(content,encoding="utf8");print(content)

if __name__=="__main__":make()
