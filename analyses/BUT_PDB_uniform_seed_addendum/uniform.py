"""Select homogeneous fold weights and reuse only exactly identical five-fold ensembles."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
import summarize
import verify

BASE=Path(__file__).resolve().parent
UNI=BASE/"uniform"
HOMOGENEOUS_PROTOCOL="9bbf2398bb8e1112adbc9dfd1257c24cdb8ee39d5d1bb501cb10712f54679230"

def score():
    summarize.ROOT=UNI
    summarize.OUT=UNI/"results"
    verify.ROOT=UNI
    summarize.main(historical_archive_check=False)
    verify.main()

def ready_set(training):
    archive={Path(s.replace("\\","/")).name for s in (training/"inventory.txt").read_text().splitlines() if s.strip()}
    assert len(archive)==60
    lr=json.loads((BASE/"chosen_lr.json").read_text())
    for s in [1,2,3]:
        for m in summarize.FAMILY:
            names=[f"lr{lr[m]}_{m}_f{f}_s{s}.pt" for f in range(5)]
            legacy=[name in archive for name in names]
            assert len(set(legacy))==1
            changed=legacy[0]
            assert changed==(s==1 or (s==2 and m in ["deq","unet_kan"]))
            weights=[training/("homogeneous_weights" if changed else "weights")/n for n in names]
            yield m,s,weights,changed

def available(training,threads):
    import torch
    import run
    assert run.sha(BASE/"uniform_addendum.txt")==HOMOGENEOUS_PROTOCOL
    torch.set_num_threads(threads)
    run.ROOT=BASE
    samples=run.read_data()
    (UNI/"parts").mkdir(parents=True,exist_ok=True)
    (UNI/"source").mkdir(exist_ok=True)
    shutil.copyfile(BASE/"source"/"references.csv",UNI/"source"/"references.csv")
    for name in ["scoring.py","inference_definitions.py"]:
        if not (UNI/name).exists():shutil.copyfile(BASE/name,UNI/name)
        assert run.sha(BASE/name)==run.sha(UNI/name)
    run.ROOT=UNI
    lineage=[]
    for m,s,weights,changed in ready_set(training):
        name=f"{m}_s{s}"
        mixed=BASE/"parts"/name
        target=UNI/"parts"/name
        if not (mixed/"verified.json").is_file() or not all(p.is_file() for p in weights):
            continue
        old=json.loads((mixed/"verified.json").read_text())
        equal=all(old["inputs"][p.name]==run.sha(p) for p in weights)
        assert equal != changed,(m,s)
        if equal:
            target.mkdir(exist_ok=True)
            for n,h in old["outputs"].items():
                assert run.sha(mixed/n)==h
                if not (target/n).exists() or run.sha(target/n)!=h:
                    shutil.copyfile(mixed/n,target/n)
            shutil.copyfile(mixed/"verified.json",target/"verified.json")
        else:
            run.evaluate(m,s,weights,samples)
        new=json.loads((target/"verified.json").read_text())
        for fold,p in enumerate(weights):
            assert run.sha(p)==new["inputs"][p.name]
            lineage.append(dict(model=m,seed=s,fold=fold,weight=p.name,source_directory=p.parent.name,
                                mixed_sha256=old["inputs"][p.name],uniform_sha256=new["inputs"][p.name],
                                reused_same_five_weights=equal))
    run.ROOT=BASE
    df=pd.DataFrame(lineage)
    if len(df)==60:
        assert df.reused_same_five_weights.sum()==30
        (UNI/"results").mkdir(exist_ok=True)
        df.to_csv(UNI/"results"/"panel_lineage.csv",index=False)
        score()
    print(f"Verified {len(df)//5}/12 uniform five-fold ensembles",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--training",type=Path,help="Extracted accompanying training archive with inventory.txt and its two weight folders.")
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    if a.training:available(a.training,a.threads)
    else:score()
