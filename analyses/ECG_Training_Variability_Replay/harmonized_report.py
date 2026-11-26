"""Validate the 120 uniform GPU-trained checkpoints and write the main sensitivity manuscript insert."""
import json
from pathlib import Path
import numpy as np
import torch
from analyse import MODELS,NAMES,write
from report import digest,read
from harmonized import homogeneous_weight,original_path,archive

R=Path(__file__).resolve().parent

def build():
    result=R/"results";scores=read(result/"harmonized_ensemble_scores.csv")
    contrasts=read(result/"harmonized_ensemble_contrasts.csv")
    historical=read(result/"ensemble_scores.csv")
    old={(r["architecture"],r["tolerance_ms"]):r for r in historical}
    rows=[];schedules={};unique=set();delta=[];changes=[]
    for m in MODELS:
        for seed in (1,2,3):
            member=np.load(result/f"harmonized_ensemble_{m}_s{seed}.npz",allow_pickle=False)
            mixed=np.load(result/f"ensemble_{m}_s{seed}.npz",allow_pickle=False)
            assert list(member["subjects"])==list(mixed["subjects"])
            assert np.array_equal(member["counts"][...,0]+member["counts"][...,1],
                                  mixed["counts"][...,0]+mixed["counts"][...,1])
            same=0
            for fold in range(5):
                base=original_path(m,fold,seed);weight=homogeneous_weight(m,fold,seed)
                h=digest(weight);assert member["weights_sha256"][fold]==h
                assert h not in unique;unique.add(h)
                is_replace=base.name in archive
                log=R/("homogeneous_logs" if is_replace else "logs")/(weight.stem+".json")
                evidence=json.loads(log.read_text())
                assert h==evidence["weight_sha256"] and evidence["device"]=="NVIDIA GeForce RTX 5070 Ti"
                assert len(evidence["history"])==8 and evidence["history"][-1]["step"]==2400
                schedule=(evidence["schedule"]["x_sha256"],evidence["schedule"]["y_sha256"])
                key=(fold,seed)
                if key in schedules:assert schedules[key]==schedule
                schedules[key]=schedule
                state=torch.load(weight,weights_only=True,map_location="cpu")
                assert state["args"]["fold"]==fold and state["args"]["seed"]==seed and state["model"]==m
                assert all(torch.isfinite(x).all().item() for x in state["state"].values() if x.is_floating_point())
                same+=int(h==mixed["weights_sha256"][fold])
                rows.append(dict(architecture=m,fold=fold,seed=seed,learning_rate=state["args"]["lr"],
                    selected_epoch=evidence["best_epoch"],validation_f1=evidence["best_validation_f1"],
                    matched_file=weight.relative_to(R).as_posix(),sha256=h,replaces_archive_cpu_weight=is_replace,
                    historical_panel_weight=base.relative_to(R).as_posix(),historical_sha256=mixed["weights_sha256"][fold],
                    training_x_sha256=schedule[0],training_y_sha256=schedule[1],elapsed_seconds=evidence["elapsed_seconds"],
                    device=evidence["device"],torch=evidence["torch"]))
            assert same in (0,5)
            if same==5:assert np.array_equal(member["counts"],mixed["counts"])
            changes.append(dict(architecture=m,seed=seed,shared_identical_members=same,replaced_members=5-same))
    assert len(rows)==120 and len(unique)==120 and len(schedules)==15
    assert sum(r["replaces_archive_cpu_weight"] for r in rows)==60 and sum(r["shared_identical_members"] for r in changes)==60
    write(result/"harmonized_ensemble_checkpoint_manifest.csv",rows)
    write(result/"harmonized_ensemble_shared_members.csv",changes)
    for r in scores:
        b=old[(r["architecture"],r["tolerance_ms"])]
        for s in (1,2,3):
            key=f"seed_{s}"
            delta.append(dict(architecture=r["architecture"],seed=s,tolerance_ms=r["tolerance_ms"],uniform_f1=r[key],
                         mixed_f1=b[key],uniform_minus_mixed_pp=float(r[key])-float(b[key])))
    write(result/"harmonized_ensemble_vs_mixed.csv",delta)
    tt=[r for r in scores if r["tolerance_ms"]=="150"];ranked=sorted(tt,key=lambda q:float(q["mean"]),reverse=True)
    spreads=[float(q["maximum"])-float(q["minimum"]) for q in tt]
    ranks=np.array([[int(q[f"rank_{s}"]) for q in tt] for s in (1,2,3)])
    correlations=[float(np.corrcoef(ranks[a],ranks[b])[0,1]) for a,b in [(0,1),(0,2),(1,2)]]
    leaders=[tt[int(i)]["label"] for i in np.argmin(ranks,axis=1)]
    e=[r for r in contrasts if r["tolerance_ms"]=="150" and r["level"]=="mean_of_three"]
    e_lines=[f"{NAMES[r['model_a']]} versus {NAMES[r['model_b']]}: {float(r['difference_pp']):+.2f} pp (95% CI {float(r['lower_95']):+.2f} to {float(r['upper_95']):+.2f})" for r in e]
    g={r["architecture"]:r["label"] for r in tt}
    replaced={(r["architecture"],r["seed"]) for r in changes if r["replaced_members"]}
    diff=[r["uniform_minus_mixed_pp"] for r in delta if r["tolerance_ms"]=="150" and (r["architecture"],r["seed"]) in replaced]
    meta=dict(architectures=8,folds=5,seeds=[1,2,3],all_gpu_weights=120,all_new_fits=120,unique_weights_including_archive=180,
        uniform_ensembles=24,shared_ensembles_with_mixed=sum(r["shared_identical_members"]==5 for r in changes),
        common_architecture_fold_seed_crop_hash_pairs=len(schedules),checked_checkpoint_states=120,
        test_participants=41,development_participants=159,baselines_before_logged_metric_amendment=35,
        score_spread_150ms_across_models_mean=[float(ranked[-1]["mean"]),float(ranked[0]["mean"])],
        within_model_seed_spread_150ms=[min(spreads),max(spreads)],paired_rank_correlations=correlations,
        best_point_estimate_by_seed=leaders,refit_minus_mixed_range_pp=[min(diff),max(diff)])
    (result/"harmonized_ensemble_design_audit.json").write_text(json.dumps(meta,indent=2))
    method=("For each of eight delineation architectures, three training seeds and five fixed participant-level development folds yielded 120 models. We trained every model on the same GPU. Within a fold and seed, all architectures received byte-identical, randomly augmented input crops and masks. We retained the original architecture-specific learning rates, optimizer, batch size of 32 and 2,400 updates per fit. The same corrected validation F1 criterion selected the checkpoint every 300 updates. We averaged the five fold-model probability maps before decoding and evaluated all 24 ensembles on the same 41 held-out LUDB participants. The endpoint was the unweighted mean of six landmark F1 scores, each computed from pooled counts across the participants' leads. We formed paired 95% intervals for the between-model difference averaged over the three observed seeds, with 10,000 participant-level bootstrap draws. These intervals quantify sampling uncertainty among the held-out participants, conditional on the fitted ensembles.")
    method+=" At 250 Hz, the nominal matching tolerances of 150 and 40 ms corresponded to 38 and 10 samples (152 and 40 ms), respectively."
    finding=(f"Across the uniformly trained ensembles, mean boundary F1 at 150 ms ranged from {float(ranked[-1]['mean']):.2f}% for {ranked[-1]['label']} to {float(ranked[0]['mean']):.2f}% for {ranked[0]['label']}. Within an architecture, the maximum minus minimum observed seed-specific F1 ranged from {min(spreads):.2f} to {max(spreads):.2f} percentage points. Pairwise rank correlations across seeds ranged from {min(correlations):.2f} to {max(correlations):.2f}. The highest numeric estimate for seeds 1, 2 and 3 was observed for {', '.join(leaders)}, respectively. " + ". ".join(e_lines)+".")
    sensitivity=(f"The historical sensitivity panel combined 60 archived CPU-trained members with the 60 new GPU-trained members absent from the archive. Twelve of the 24 ensemble predictions were identical between the panels. For the 12 ensembles whose five members were all replaced, the same-seed F1 change at 150 ms ranged from {min(diff):+.2f} to {max(diff):+.2f} pp. Archive timestamps placed 35 historical checkpoints before the logged annotation-edge scoring amendment. The historical comparison incorporates training-runtime and potential validation-selection differences. The uniform panel holds these factors constant. All comparisons use the same 41 test participants; the alternative panels do not increase the independent test sample.")
    lines=["# Training variability across all architectures","## Methods",method,"## Results",finding,"## Historical sensitivity",sensitivity]
    for tol in ["150","40"]:
        table=["| Architecture | Seed 1 | Seed 2 | Seed 3 | Mean (SD) |","|---|---:|---:|---:|---:|"]
        for q in [r for r in scores if r["tolerance_ms"]==tol]:
            table.append(f"| {q['label']} | {float(q['seed_1']):.2f} | {float(q['seed_2']):.2f} | {float(q['seed_3']):.2f} | {float(q['mean']):.2f} ({float(q['sample_sd']):.2f}) |")
        lines.extend([f"## {tol}-ms tolerance","\n".join(table)])
    (R/"Harmonized_ensemble_insert.md").write_text("\n\n".join(lines)+"\n",encoding="utf8")
    print(json.dumps(meta,indent=2));print(finding)

if __name__=="__main__":build()
