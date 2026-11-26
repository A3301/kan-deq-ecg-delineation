"""Validate all fitting/checkpoint provenance and write a manuscript-ready addendum."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import torch
import scipy
import wfdb
from analyse import MODELS, NAMES, write

ROOT=Path(__file__).resolve().parent

def digest(path):
    with path.open("rb") as fd:return hashlib.file_digest(fd,"sha256").hexdigest()

def read(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def fmt(num):return f"{float(num):.2f}"

def report():
    results=ROOT/"results"
    original={Path(p.replace("\\","/")).name for p in (ROOT/"inventory.txt").read_text().splitlines() if p.strip()}
    assert len(original)==60
    scores=read(results/"ensemble_scores.csv")
    effects=read(results/"ensemble_contrasts.csv")
    learning_rates=json.loads((ROOT/"chosen_lr.json").read_text())
    sp=json.loads((ROOT/"splits.json").read_text())
    folds=[set(f) for f in sp["folds"]];locked=set(sp["test"])
    assert len(folds)==5 and len(locked)==41 and sum(map(len,folds))==159
    assert len(set.union(*folds))==159 and not locked.intersection(set.union(*folds))
    weights=[]
    for m in MODELS:
        for seed in (1,2,3):
            npz=np.load(results/f"ensemble_{m}_s{seed}.npz",allow_pickle=False)
            for fold in range(5):
                p=ROOT/"weights"/f"lr{learning_rates[m]}_{m}_f{fold}_s{seed}.pt"
                rolled=npz["weights_sha256"][fold];hash_=digest(p)
                assert hash_==rolled
                ck=torch.load(p,weights_only=True,map_location="cpu")
                assert all(torch.isfinite(t).all().item() for t in ck["state"].values() if t.is_floating_point())
                assert ck["model"]==m and ck["args"]["fold"]==fold and ck["args"]["seed"]==seed
                assert ck["args"]["bs"]==32 and ck["args"]["epochs"]==40 and ck["args"]["steps"]==60
                assert ck["args"]["lr"]==float(learning_rates[m]); log={}
                if p.name not in original:
                    log=json.loads((ROOT/"logs"/(p.stem+".json")).read_text())
                    assert log["weight_sha256"]==hash_ and log["protocol_sha256"]==digest(ROOT/"protocol.txt")
                    assert [h["step"] for h in log["history"]]==list(range(300,2401,300))
                    assert log["best_validation_f1"]==max(h["val_mean_f1"] for h in log["history"])
                weights.append(dict(model=m,seed=seed,fold=fold,learning_rate=learning_rates[m],
                                    origin="archived CPU" if p.name in original else "new GPU",filename=p.name,
                                    sha256=hash_,new_selected_epoch=log.get("best_epoch",""),
                                    new_best_validation_f1=log.get("best_validation_f1",""),
                                    new_fit_seconds=log.get("elapsed_seconds","")))
    assert len(weights)==120 and sum(r["origin"]=="new GPU" for r in weights)==60
    write(results/"checkpoint_manifest.csv",weights)
    source_map={"U-Net":"unet","U-Net-KAN":"unet_kan","ConvKAN":"convkan","CNN-BiLSTM":"cnn_bilstm",
                "DEQ (implicit)":"deq","DEQ-unrolled":"deq_unrolled","Explicit stack":"explicit_stack"}
    now={r["architecture"]:float(r["seed_1"]) for r in scores if r["tolerance_ms"]=="150"}
    anchors={}
    for row in read(ROOT/"source/published_primary.csv"):
        for a in ("A","B"):
            m=source_map[row[a]]; old=float(row["mean F1 "+a]); diff=now[m]-old
            anchors[m]={"reference_published_rounded":old,"rerun_exact":now[m],"difference_pp":diff}
    (results/"archived_primary_anchors.json").write_text(json.dumps(anchors,indent=2))
    largest=max(abs(row["difference_pp"]) for row in anchors.values())
    env=dict(python=platform.python_version(),os=platform.platform(),pytorch=str(torch.__version__),numpy=np.__version__,
             scipy=scipy.__version__,wfdb=wfdb.__version__,cuda_runtime=torch.version.cuda,
             gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else "",new_fits=60,
             new_training_steps=60*2400,checkpoint_files=120,ensembles=24,all_floating_checkpoint_tensors_finite=True,
             held_out_participants=len(locked),validation_fold_sizes=[len(f) for f in folds],
             archived_seed1_anchor_max_abs_pp=largest)
    (results/"run_environment.json").write_text(json.dumps(env,indent=2))
    twos=[r for r in scores if r["tolerance_ms"]=="150"]
    order=sorted(twos,key=lambda r:float(r["mean"]),reverse=True)
    spread={r["label"]:float(r["maximum"])-float(r["minimum"]) for r in twos}
    rank_pairs=[];seed_leaders={}
    for tol in ["150","40"]:
        subset=[r for r in scores if r["tolerance_ms"]==tol]
        rank_values=np.array([[int(r[f"rank_{s}"]) for r in subset] for s in (1,2,3)])
        seed_leaders[tol]=[subset[int(np.argmin(one))]["label"] for one in rank_values]
        for a,b in [(0,1),(0,2),(1,2)]:
            rank_pairs.append(dict(tolerance_ms=tol,seed_A=a+1,seed_B=b+1,
                                   spearman_rho=float(np.corrcoef(rank_values[a],rank_values[b])[0,1])))
    write(results/"ensemble_seed_rank_correlations.csv",rank_pairs)
    rho=[r["spearman_rho"] for r in rank_pairs if r["tolerance_ms"]=="150"]
    eff=[r for r in effects if r["tolerance_ms"]=="150" and r["level"]=="mean_of_three"]
    methods=("We repeated the eight original architectures with training seeds 1, 2 and 3. For each seed, five models per architecture were fitted on the original development folds, retaining all 12 leads from a participant in the same partition. The 159 development participants and 41 held-out participants were identical across runs. Every fold used the model-specific learning rate, 2,400 updates and a batch size of 32. Sixty models came from the CPU archive and 60 were trained on a GPU. New fits used the original crop and augmentation generator with a shared schedule within each fold and seed. The highest six-boundary F1 on the fold validation participants selected the checkpoint at 300-update intervals. We averaged the five members’ predicted probabilities before decoding. For each training seed we pooled counts by landmark on the 41 test participants and averaged the six landmark F1 values. We estimated paired 95% percentile intervals for differences averaged over the three observed training seeds using 10,000 participant-cluster bootstrap draws. These intervals quantify test-participant uncertainty conditional on the fitted models.")
    compare=[f"{NAMES[r['model_a']]} minus {NAMES[r['model_b']]}: {float(r['difference_pp']):+.2f} pp (95% CI {float(r['lower_95']):+.2f} to {float(r['upper_95']):+.2f})" for r in eff]
    result=(f"Across three five-member ensembles per architecture, mean six-boundary F1 at the 150-ms tolerance ranged from {fmt(order[-1]['mean'])}% for {order[-1]['label']} to {fmt(order[0]['mean'])}% for {order[0]['label']}. Within-architecture spreads across seeds ranged from {min(spread.values()):.2f} to {max(spread.values()):.2f} percentage points. The empirical rank correlations across pairs of seeds ranged from {min(rho):.2f} to {max(rho):.2f} (Spearman rho). The models with the highest point estimate by seed were {', '.join(seed_leaders['150'])}, respectively. " + ". ".join(compare)+".")
    limitations=("The three training seeds characterize the observed variation in this experiment. The test cohort, folds and learning rates were fixed across seeds. The archived models were trained on CPU and new models on GPU, and the archived model-selection runs may predate the correction to the annotation-edge scoring rule. Variation across the resulting ensembles combines training variation with these implementation differences. These are secondary within-LUDB findings. Independent-domain effects of the matched KAN and GELU heads are reported in the separate head-only experiment.")
    rows=["| Architecture | Seed 1 | Seed 2 | Seed 3 | Mean (SD) |","|---|---:|---:|---:|---:|"]
    for tol in ["150","40"]:
        subset=[r for r in scores if r["tolerance_ms"]==tol]
        if tol=="40": rows+=["","**40-ms tolerance**","",rows[0],rows[1]]
        for r in subset:
            rows.append(f"| {r['label']} | {fmt(r['seed_1'])} | {fmt(r['seed_2'])} | {fmt(r['seed_3'])} | {fmt(r['mean'])} ({fmt(r['sample_sd'])}) |")
    text="\n\n".join(["# Training variability across eight delineation architectures","## Methods",methods,"## Results",result,"**150-ms tolerance**","\n".join(rows),
                       "## Interpretation",limitations,"**Figure caption**","Held-out LUDB six-boundary F1 at the 150-ms and 40-ms tolerances. Symbols identify the three training seeds. Each symbol represents the probability ensemble of five development-fold models evaluated on the same 41 participants. The segment connects the observed minimum and maximum across seeds."])
    (ROOT/"Manuscript_insert.md").write_text(text+"\n",encoding="utf8")
    print(result,flush=True)
    print("Largest difference from reported seed-1 macro-F1",largest,flush=True)

if __name__=="__main__":report()
