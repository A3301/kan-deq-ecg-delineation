"""RDB DEQ transport. Single-record inference batching preserves Anderson stopping."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
import pandas as pd
from deq_frozen import DEQSeg
from frozen_model import predict_proba
from rdb_external import score, sha, json_out
from summarize_rdb import POINTS, interval_arrays, metrics, score_arr, seed_mean_f1


BASE=Path(__file__).resolve().parent
ARMS=["DEQ","Twin8"]
STATE={}


def get_models(arm,seed,path):
    key=arm,seed
    if key in STATE: return STATE[key]
    model,tag,mode=("deq","lr1e-3","deq") if arm=="DEQ" else ("deq_unrolled","lr3e-3","unrolled")
    nets,hashes=[],{}
    for fold in range(5):
        weight=Path(path)/f"{tag}_{model}_f{fold}_s{seed}.pt"
        ck=torch.load(weight,weights_only=True,map_location="cpu")
        args=ck["args"]
        assert ck["model"]==model and ck["kw"]=={} and args["fold"]==fold and args["seed"]==seed
        assert not args.get("xdb_qtdb",0) and not args.get("multi",0)
        net=DEQSeg(mode=mode)
        net.load_state_dict(ck["state"],strict=True)
        nets.append(net.eval())
        hashes[weight.name]=sha(weight)
    STATE[key]=(nets,hashes)
    return nets,hashes


def worker_start():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def pilot_proba(nets, signal):
    assert len(signal)==2500
    two=torch.from_numpy(np.stack((signal[:2048],signal[-2048:]))).unsqueeze(1)
    # No window from any other ECG is present in this batch.
    probabilities=sum(net(two).softmax(dim=1) for net in nets)/len(nets)
    spline=np.hanning(2048).astype("float32")+0.001
    den=np.zeros(2500,dtype="float32")
    num=np.zeros((4,2500),dtype="float32")
    for k,start in enumerate((0,452)):
        den[start:start+2048]+=spline
        num[:,start:start+2048]+=probabilities[k].numpy()*spline
    return num/den


def pilot(args):
    ids=json.loads((args.cohort/"signal_order.json").read_text())
    signals=np.load(args.cohort/"signals.npy",mmap_mode="r")
    indices=np.random.default_rng(20261126).choice(len(ids),size=4,replace=False)
    tests=[]
    with torch.inference_mode():
        for arm in ARMS:
            for seed in [1,2]:
                nets,weights=get_models(arm,seed,args.weights)
                audits=[]
                for idx in indices:
                    canonical=predict_proba(nets,signals[idx])
                    it1=[n.last_info["n_iter"] for n in nets]
                    separate=pilot_proba(nets,signals[idx])
                    it2=[n.last_info["n_iter"] for n in nets]
                    maxerr=float(abs(canonical-separate).max())
                    discord=int((canonical.argmax(0)!=separate.argmax(0)).sum())
                    assert discord==0 and maxerr<1e-6 and it1==it2
                    audits.append(dict(record=ids[idx],max_probability_error=maxerr,
                                       different_class_samples=discord,n_iter_5_folds=it1))
                tests.append(dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in nets[0].parameters()),
                                  weights=weights,pilot=audits))
    json_out(args.out/"numerical_pilot.json",tests)


def shard(task):
    arm,seed,start,stop,cohort,weights,out=task
    out=Path(out)
    part=out/f"part_{arm}_{seed}_{start}_{stop}.npz"
    if part.exists(): return (arm,seed,start,stop)
    signals=np.load(Path(cohort)/"signals.npy",mmap_mode="r")
    nets,_=get_models(arm,seed,weights)
    preds=np.empty((stop-start,2500),dtype="uint8")
    iterations=np.empty((stop-start,5),dtype="uint8")
    residuals=np.empty((stop-start,5),dtype="float32")
    with torch.inference_mode():
        for i in range(start,stop):
            prob=predict_proba(nets,signals[i])
            assert np.isfinite(prob).all(), (arm,seed,i)
            for j,n in enumerate(nets):
                iterations[i-start,j]=n.last_info["n_iter"]
                residuals[i-start,j]=float(n.last_info["rel_res"].max().item())
            preds[i-start]=prob.argmax(0)
    partial=part.with_suffix(".part")
    with partial.open("wb") as f:
        np.savez_compressed(f,mask=preds,iterations=iterations,residuals=residuals)
    partial.replace(part)
    return arm,seed,start,stop


def infer(args):
    # Child processes each process a complete trace serially through all five folds.
    n=len(json.loads((args.cohort/"signal_order.json").read_text()))
    parts=args.out/"parts"
    parts.mkdir(exist_ok=True)
    size=60
    tasks=[(arm,seed,s,min(s+size,n),str(args.cohort),str(args.weights),str(parts))
           for s in range(0,n,size) for arm in ARMS for seed in (1,2)]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=worker_start) as pool:
        futures=[pool.submit(shard,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i==1 or i%8==0:
                print(f"{i}/{len(tasks)} shards finished in {time.monotonic()-started:.1f} s",flush=True)
    ids=json.loads((args.cohort/"signal_order.json").read_text())
    diagnostics=[]
    for arm in ARMS:
        for seed in (1,2):
            arrays={"mask":[],"iterations":[],"residuals":[]}
            for a,j,s,e,*_ in tasks:
                if arm!=a or seed!=j: continue
                with np.load(parts/f"part_{arm}_{seed}_{s}_{e}.npz") as pack:
                    for k in arrays: arrays[k].append(pack[k])
            masks,iterations,residuals=[np.concatenate(arrays[k]) for k in arrays]
            assert masks.shape==(len(ids),2500)
            np.save(args.out/f"masks_{arm}_{seed}.npy",masks,allow_pickle=False)
            for i,rid in enumerate(ids):
                for j in range(5):
                    diagnostics.append((rid,arm,seed,j,int(iterations[i,j]),float(residuals[i,j])))
    pd.DataFrame(diagnostics,columns=["record","arm","seed","fold","common_window_iterations","max_window_final_residual"]).to_csv(args.out/"solver_diagnostics.csv",index=False)


def summarize(args):
    df=pd.read_csv(args.out/"deq_record_scores.csv")
    meta=pd.read_csv(args.out/"cohort.csv").query("status=='included'").set_index("record").sort_index()
    ids=meta.index.tolist()
    assert len(df)==len(ids)*48
    summary,seeds=[],[]
    for tol in [150,40]:
        counts=score_arr(df[df.tolerance_ms==tol],ids,ARMS)
        full=counts.sum(axis=0)
        pooled=seed_mean_f1(full)*100
        bounds=interval_arrays(counts,meta.rhythm.to_numpy(),np.random.default_rng(20261125))
        for j,k in enumerate([*POINTS,"macro"]):
            summary.append(dict(tolerance_ms=tol,endpoint=k,deq_f1=pooled[0,j],deq_lo=bounds[0,0,j],deq_hi=bounds[1,0,j],
                twin8_f1=pooled[1,j],twin8_lo=bounds[0,1,j],twin8_hi=bounds[1,1,j],deq_minus_twin8_pp=pooled[0,j]-pooled[1,j],
                delta_lo=bounds[0,2,j],delta_hi=bounds[1,2,j]))
        for i,arm in enumerate(ARMS):
            for j,seed in enumerate((1,2)):
                fs=metrics(full)[0][i,j]*100
                for k,val in zip([*POINTS,"macro"],[*fs,fs.mean()],strict=True):
                    seeds.append(dict(tolerance_ms=tol,arm=arm,seed=seed,endpoint=k,f1=val))
    pd.DataFrame(summary).to_csv(args.out/"deq_comparison.csv",index=False)
    pd.DataFrame(seeds).to_csv(args.out/"deq_seeds.csv",index=False)
    print(pd.DataFrame(summary).query("endpoint=='macro'").round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    import shutil
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,default=BASE/"weights")
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--out",type=Path,default=BASE/"results")
    p.add_argument("--jobs",type=int,default=4)
    p.add_argument("--stage",choices=["all","pilot","infer","score","summarize"],default="all")
    a=p.parse_args(); a.out.mkdir(exist_ok=True,parents=True)
    os.environ.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")
    worker_start()
    for name in ["cohort.csv","signal_order.json","references_ms.json"]:
        if (a.cohort/name).resolve()!=(a.out/name).resolve(): shutil.copy2(a.cohort/name,a.out/name)
    if a.stage in ["all","pilot"]: pilot(a)
    if a.stage in ["all","infer"]: infer(a)
    if a.stage in ["all","score"]: score(a,ARMS,"deq_record_scores.csv")
    if a.stage in ["all","summarize"]: summarize(a)
