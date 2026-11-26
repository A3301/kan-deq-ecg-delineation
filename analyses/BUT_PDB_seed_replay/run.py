"""Apply each completed, fixed five-fold H3 ensemble on the native BUT PDB ECGs."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
if hasattr(pd.options.future,"infer_string"):
    pd.options.future.infer_string=False
import wfdb
from inference_definitions import CNNBiLSTM,ConvKANNet,DEQSeg,UNetKAN,merge,peak_of,predict_proba,runs
from scoring import counts

ROOT=Path(__file__).resolve().parent
NATIVE=ROOT/"source"/"native"
MODELS={"cnn_bilstm":CNNBiLSTM,"unet_kan":UNetKAN,"convkan":ConvKANNet,"deq":DEQSeg}
SEEDS=(1,2,3)
PROTOCOL="aafe3150d1c2626e7f05048eb171bef86ea1fd8b5a98f18534b374ee9b38e649"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read_data():
    assert sha(ROOT/"protocol.txt")==PROTOCOL
    assert sha(NATIVE/"SHA256SUMS.txt")=="b7231f90c5c167dec5a151e099e0b9732a90b1ee48c2a0c8d0ebcb563187850c"
    hashes=dict((name,h) for h,name in
                (line.split(maxsplit=1) for line in (NATIVE/"SHA256SUMS.txt").read_text().splitlines() if line.strip()))
    assert sha(NATIVE/"README.txt")==hashes["README.txt"]
    pattern=re.compile(r"^\s*(\d+)\s+'[^']*'\s+(.+?Database)\s+(\d+)\s+\[(\d+),\s*(\d+)\]\s+(\d+)\s*$")
    roster={}
    for line in (NATIVE/"README.txt").read_text().splitlines():
        m=pattern.match(line)
        if m:
            rid,db,source,start,end,fs=m.groups()
            roster[f"{int(rid):02d}"]=(db.strip()+":"+source,int(start),int(end),int(fs))
    assert len(roster)==50 and len({r[0] for r in roster.values()})==45
    filt=butter(2,[.5,40],btype="bandpass",fs=250,output="sos")
    samples=[]
    references=[]
    for rid,(source,start,end,source_fs) in roster.items():
        for ext in ["hea","dat","pwave"]:
            file=rid+"."+ext
            assert sha(NATIVE/file)==hashes[file],file
        sig,meta=wfdb.rdsamp(str(NATIVE/rid))
        fs=int(meta["fs"])
        assert fs==source_fs and sig.shape[1]==2
        labels=wfdb.rdann(str(NATIVE/rid),"pwave")
        ref=labels.sample.tolist()
        assert set(labels.symbol)=={"N"}
        if rid in ["08","48"]:
            assert ref==[-1]
            ref=[]
        assert all(0<=v<len(sig) for v in ref) and ref==sorted(set(ref))
        frac=Fraction(250,fs)
        traces=resample_poly(sig,frac.numerator,frac.denominator,axis=0) if fs!=250 else sig
        references.append(dict(segment=rid,source=source,native_hz=fs,native_samples=len(sig),samples_250Hz=len(traces),
                               source_start=start,source_end=end,valid_ref=len(ref),native_ref=" ".join(map(str,ref))))
        for lead,t in enumerate(traces.T):
            v=sosfiltfilt(filt,np.nan_to_num(t.astype(np.float64)))
            q25,q75=np.percentile(v,[25,75])
            sc=(q75-q25)/1.349
            x=((v-np.median(v))/(sc if sc>=1e-8 else v.std()+1e-8)).astype(np.float32)
            samples.append((rid,source,fs,lead,x,ref))
    assert sum(r["valid_ref"] for r in references)==5429
    pd.DataFrame(references).to_csv(ROOT/"source"/"references.csv",index=False)
    return samples

def evaluate(model,seed,paths,samples):
    part=ROOT/"parts"/f"{model}_s{seed}"
    part.mkdir(exist_ok=True,parents=True)
    stamp={p.name:sha(p) for p in paths}
    stamp["protocol"]=PROTOCOL
    stamp["inference_definitions"]=sha(ROOT/"inference_definitions.py")
    stamp["scoring"]=sha(ROOT/"scoring.py")
    stamp["run"]=sha(Path(__file__))
    flag=part/"verified.json"
    if flag.is_file():
        saved=json.loads(flag.read_text())
        if saved["inputs"]==stamp and all(sha(part/name)==h for name,h in saved["outputs"].items()):
            return
    nets=[]
    for fold,path in enumerate(paths):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        assert ck["model"]==model
        if "args" in ck and "fold" in ck["args"]:
            assert ck["args"]["fold"]==fold and ck["args"]["seed"]==seed
        net=MODELS[model](**ck["kw"])
        net.load_state_dict(ck["state"],strict=True)
        nets.append(net.eval())
    predictions,stats=[],[]
    start=time.perf_counter()
    with torch.inference_mode():
        for rid,source,fs,lead,x,ref in samples:
            mask=predict_proba(nets,x).argmax(0)
            peaks=[peak_of(x,i,j) for i,j in merge(runs(mask,1)) if j-i+1>=10]
            predictions.append(dict(model=model,seed=seed,segment=rid,source=source,lead=lead,
                                    peaks_250Hz=" ".join(map(str,peaks))))
            for coordinate in ["historical_grid","native_ms"]:
                for tol in [150,50]:
                    tp,fn,fp=counts(ref,peaks,fs,len(x),tol,coordinate)
                    stats.append(dict(model=model,seed=seed,segment=rid,source=source,lead=lead,
                                      coordinate=coordinate,tol_ms=tol,tp=tp,fn=fn,fp=fp))
    pd.DataFrame(predictions).to_csv(part/"predictions.csv",index=False)
    pd.DataFrame(stats).to_csv(part/"counts.csv",index=False)
    outputs={name:sha(part/name) for name in ["predictions.csv","counts.csv"]}
    flag.write_text(json.dumps(dict(inputs=stamp,outputs=outputs,
               runtime_s=time.perf_counter()-start,torch=str(torch.__version__),wfdb=wfdb.__version__),indent=2))
    print(f"Completed {model}, seed {seed}, {time.perf_counter()-start:.1f} s",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("weights",type=Path)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--models",nargs="+",choices=list(MODELS),default=list(MODELS))
    parser.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=list(SEEDS))
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    lr=json.loads((ROOT/"chosen_lr.json").read_text())
    data=read_data()
    for seed in args.seeds:
        for model in args.models:
            paths=[args.weights/f"lr{lr[model]}_{model}_f{f}_s{seed}.pt" for f in range(5)]
            if all(p.is_file() for p in paths):
                evaluate(model,seed,paths,data)
            else:
                print(f"Pending weights for {model} seed {seed}: {sum(p.is_file() for p in paths)} of 5 folds.",flush=True)

if __name__=="__main__":
    main()
