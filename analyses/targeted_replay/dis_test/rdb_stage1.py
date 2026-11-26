"""Additional full U-Net-KAN versus U-Net transport check on the same RDB lead II."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from frozen_model import UNetKAN
from rdb_external import HERE, infer, score, sha
from summarize_rdb import POINTS, interval_arrays, metrics, score_arr, seed_mean_f1


STAGE = ["UNetKAN","UNet"]


def checkpoint_loader(arm, seed, folder):
    nets, hashes = [], {}
    tag, model = ("lr3e-3","unet_kan") if arm=="UNetKAN" else ("lr1e-3","unet")
    for fold in range(5):
        path = folder/f"{tag}_{model}_f{fold}_s{seed}.pt"
        ck = torch.load(path, weights_only=True, map_location="cpu")
        assert ck["model"]==model and ck["kw"]=={} and ck["args"]["seed"]==seed and ck["args"]["fold"]==fold
        assert not ck["args"].get("xdb_qtdb",0) and not ck["args"].get("multi",0)
        m = UNetKAN(where=("bottleneck","head") if arm=="UNetKAN" else ())
        m.load_state_dict(ck["state"], strict=True)
        nets.append(m.eval())
        hashes[path.name]=sha(path)
    return nets,hashes


def summarize(folder):
    df = pd.read_csv(folder/"stage1_record_scores.csv")
    cohort = pd.read_csv(folder/"cohort.csv")
    kept = cohort[cohort.status=="included"].set_index("record").sort_index()
    assert len(df)==len(kept)*2*2*2*6
    res, seeds = [], []
    for tol in (150,40):
        counts = score_arr(df[df.tolerance_ms==tol],kept.index.tolist(),STAGE)
        sums = counts.sum(axis=0)
        pooled = seed_mean_f1(sums)*100
        boot = interval_arrays(counts,kept.rhythm.to_numpy(),np.random.default_rng(20261125))
        for k,point in enumerate(POINTS+["macro"]):
            res.append(dict(tolerance_ms=tol,endpoint=point,unet_kan=pooled[0,k],unet_kan_lo=boot[0,0,k],
                        unet_kan_hi=boot[1,0,k],unet=pooled[1,k],unet_lo=boot[0,1,k],unet_hi=boot[1,1,k],
                        unet_kan_minus_unet_pp=pooled[0,k]-pooled[1,k],delta_lo=boot[0,2,k],delta_hi=boot[1,2,k]))
        f1 = metrics(sums)[0]*100
        for i,arm in enumerate(STAGE):
            for j,seed in enumerate([1,2]):
                seeds.append(dict(tolerance_ms=tol,arm=arm,seed=seed,macro_f1=f1[i,j].mean()))
    pd.DataFrame(res).to_csv(folder/"stage1_comparison.csv",index=False)
    pd.DataFrame(seeds).to_csv(folder/"stage1_seeds.csv",index=False)
    print(pd.DataFrame(res).query("endpoint=='macro'").round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--stage",choices=("all","infer","score","summarize"),default="all")
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if a.stage in ("all","infer"): infer(a,STAGE,checkpoint_loader)
    if a.stage in ("all","score"): score(a,STAGE,"stage1_record_scores.csv")
    if a.stage in ("all","summarize"): summarize(a.out)
