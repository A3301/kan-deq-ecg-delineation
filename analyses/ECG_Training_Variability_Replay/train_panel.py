"""Run missing Stage-1 fits, followed by standardized locked-subject evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

import models
from metrics import match
from postprocess import runs,merge

HERE=Path(__file__).resolve().parent
LR=json.loads((HERE/"chosen_lr.json").read_text())
SPLITS=json.loads((HERE/"splits.json").read_text())
POINTS=("P_on","P_off","QRS_on","QRS_off","T_on","T_off")
FAMILIES=("cnn_bilstm","transformer","convkan","explicit_stack","unet","unet_kan","deq_unrolled","deq")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
RESULTS=HERE/"results"
LOGS=HERE/"logs"
ONE_LEAD_PER_CALL=False
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def file_sha(path):
    with open(path,"rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def metadata():
    root=HERE/"prepared";x=np.load(root/"x.npy");rows=json.loads((root/"rows.json").read_text())
    assert x.shape==(2400,2500) and len(rows)==len(x)
    return x,rows

def output_path(model,fold,seed):
    return HERE/"weights"/f"lr{LR[model]}_{model}_f{fold}_s{seed}.pt"

@torch.no_grad()
def predict(net,x):
    ans=[]
    taper=torch.as_tensor(np.hanning(2048).astype(np.float32)+.001,device=DEVICE)
    denom=torch.zeros(2500,device=DEVICE);denom[:2048]+=taper;denom[-2048:]+=taper
    batch=1 if ONE_LEAD_PER_CALL or getattr(net,"mode","")=="deq" else 6
    for g in range(0,len(x),batch):
        a=torch.as_tensor(np.ascontiguousarray(x[g:g+batch]),device=DEVICE)
        inputs=torch.stack([a[:,:2048],a[:,-2048:]],1).reshape(-1,1,2048)
        probs=net(inputs).softmax(dim=1).reshape(len(a),2,4,2048)
        p=torch.zeros(len(a),4,2500,device=DEVICE);p[:,:,:2048]+=probs[:,0]*taper;p[:,:,-2048:]+=probs[:,1]*taper
        ans.append((p/denom).cpu().numpy())
    return np.concatenate(ans)

def counts_for(prob,rows,tolerances):
    accum=np.zeros((len(tolerances),len(rows),6,3),dtype=np.int64)
    for j,(mask,row) in enumerate(zip(prob.argmax(1),rows)):
        s0,s1=row["span"]
        for cls in (1,2,3):
            rr=[(a,b) for a,b in merge(runs(mask,cls)) if b-a+1>=(15 if cls==3 else 10)]
            for side in (0,1):
                qi=(cls-1)*2+side
                det=np.asarray([r[side] for r in rr],dtype=int)
                ref=np.asarray(row["points"][POINTS[qi]],dtype=int);ref=ref[(ref>=s0)&(ref<s1)]
                for k,tol in enumerate(tolerances):
                    matched,d,fn,_=match(ref,det,round(tol*.25))
                    unused=np.setdiff1d(det,d);fp=int(((unused>=s0)&(unused<s1)).sum())
                    accum[k,j,qi]=[len(matched),fn,fp]
    return accum

def f1(counts):
    tp,fn,fp=np.moveaxis(counts,-1,0)
    return 200.*tp/(2*tp+fn+fp)

def train(m,fold,seed,x,rows):
    dest=output_path(m,fold,seed);log=LOGS/(dest.stem+".json")
    if dest.is_file():return
    cache=HERE/"batch_cache"/f"f{fold}_s{seed}"
    if not cache.with_suffix(".json").is_file():raise RuntimeError("Run prepare.py first")
    x_path=cache.with_suffix(".x.npy");y_path=cache.with_suffix(".y.npy")
    xb=torch.from_numpy(np.load(x_path)).to(DEVICE);yb=torch.from_numpy(np.load(y_path)).to(DEVICE)
    ix=[i for i,r in enumerate(rows) if r["subject"] in SPLITS["folds"][fold]]
    val_x=x[ix];val_rows=[rows[i] for i in ix]
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=models.build(m).to(DEVICE);rate=float(LR[m])
    opt=torch.optim.AdamW(net.parameters(),lr=rate,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=rate,total_steps=2400,pct_start=.1)
    history=[];best=-1;best_epoch=None;state=None;start=time.perf_counter()
    for step in range(1,2401):
        net.train()
        result=net(xb[step-1,:,None]);loss=F.cross_entropy(result,yb[step-1].long(),ignore_index=-1)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step();sched.step()
        if step==60:
            if DEVICE=="cuda":
                torch.cuda.synchronize()
            print("TIMING",m,fold,seed,"first 60 updates",round(time.perf_counter()-start,1),flush=True)
        if step%300==0:
            net.eval();v=f1(counts_for(predict(net,val_x),val_rows,[150])[0].sum(axis=0)).mean()
            history.append({"step":step,"loss":float(loss.detach().cpu()),"val_mean_f1":float(v)})
            if float(v)>best:
                best=float(v);best_epoch=step//60
                state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            print("VAL",m,fold,seed,step,round(v,4),round(time.perf_counter()-start,1),flush=True)
    if not all(np.isfinite(r['loss']) for r in history) or state is None:raise RuntimeError((m,fold,seed,history))
    tmp=dest.with_suffix(".tmp")
    ck={"model":m,"kw":{},"args":{"model":m,"fold":fold,"seed":seed,"epochs":40,"steps":60,"bs":32,"lr":rate,
                                 "eval_every":5,"tag":"GPU_replication"},"state":state}
    torch.save(ck,tmp);tmp.replace(dest)
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(json.dumps(dict(history=history,best_epoch=best_epoch,best_validation_f1=best,elapsed_seconds=time.perf_counter()-start,
                   device=torch.cuda.get_device_name() if DEVICE=="cuda" else "CPU",torch=torch.__version__,weight_sha256=file_sha(dest),
                   schedule=json.loads(cache.with_suffix(".json").read_text()),protocol_sha256=file_sha(HERE/"protocol.txt")),indent=2),encoding="utf8")
    print("FIT",m,fold,seed,round(time.perf_counter()-start,1),flush=True)

def score(m,seed,folds,x,rows,tag):
    dest=RESULTS/f"{tag}_{m}_s{seed}.npz"
    if dest.is_file():return
    accumulated=None;files=[]
    for fold in folds:
        path=output_path(m,fold,seed);ck=torch.load(path,weights_only=True,map_location="cpu")
        assert ck["model"]==m and ck["kw"]=={} and ck["args"]["seed"]==seed and ck["args"]["fold"]==fold
        net=models.build(m);net.load_state_dict(ck["state"],strict=True);net.to(DEVICE).eval()
        p=predict(net,x);accumulated=p if accumulated is None else accumulated+p
        files.append(file_sha(path))
    counts=counts_for(accumulated/len(folds),rows,[150,40]);subjects=sorted({r["subject"] for r in rows},key=int)
    totals=np.stack([counts[:,[i for i,r in enumerate(rows) if r["subject"]==s]].sum(axis=1) for s in subjects],1)
    assert totals.shape==(2,41,6,3)
    np.savez_compressed(dest,counts=totals,subjects=subjects,weights_sha256=files,folds=folds,seed=seed)
    print("SCORED",tag,m,seed,f1(totals.sum(axis=1)).mean(axis=1),flush=True)

def main():
    global RESULTS
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",action="store_true");mode.add_argument("--score-only",action="store_true")
    ap.add_argument("--result-dir", type=Path, default=HERE/"results",help="Use a fresh directory to force safe independent inference.")
    ap.add_argument("--first-fold-only",action="store_true");args=ap.parse_args()
    RESULTS=args.result_dir.resolve()
    x,rows=metadata();ix=[i for i,r in enumerate(rows) if r["subject"] in SPLITS["test"]]
    test=x[ix];tr=[rows[i] for i in ix];assert len(tr)==492
    RESULTS.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        if args.train:
            for seed in (2,3):
                for m in FAMILIES:train(m,fold,seed,x,rows)
        if fold==0:
            for m in FAMILIES:
                for seed in (1,2,3):score(m,seed,[0],test,tr,"fold0")
            if args.first_fold_only:return
    for m in FAMILIES:
        for seed in (1,2,3):score(m,seed,list(range(5)),test,tr,"ensemble")

if __name__=="__main__":main()
