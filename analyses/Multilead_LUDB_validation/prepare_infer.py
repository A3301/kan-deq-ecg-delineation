"""Validate official LUDB release, read native annotations and run frozen CPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch

if hasattr(pd.options.future, "infer_string"):
    pd.options.future.infer_string = False
import wfdb
from frozen_model import UNetKAN, predict_proba

HERE = Path(__file__).resolve().parent
LEADS = ["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
POINTS = ["P_on","P_off","QRS_on","QRS_off","T_on","T_off"]
ARCHIVE = "d03192c7361ab5deeaba3f0d46e274b9ecb0b74c9caf8964151e20f1b5a7df06"
PROTOCOL = "3342345f1f879cb306e74a9466b9172b99a4876697d1a0b4a02e02812d3ebe57"
def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def extract(archive, raw):
    if sha(archive) != ARCHIVE: raise ValueError("Unexpected LUDB archive digest")
    raw.mkdir(exist_ok=True,parents=True)
    with ZipFile(archive) as z:
        index=next(n for n in z.namelist() if n.endswith("/SHA256SUMS.txt"))
        root=index.rsplit("/",1)[0]+"/"
        expected={s.split()[-1].lstrip("*").removeprefix("./"):s.split()[0] for s in z.read(index).decode().splitlines()}
        for rid in range(1,201):
            for ext in ["hea","dat",*LEADS]:
                key=f"data/{rid}.{ext}";data=z.read(root+key)
                if hashlib.sha256(data).hexdigest()!=expected[key]:raise ValueError(key)
                (raw/f"{rid}.{ext}").write_bytes(data)
    return 200*14

def parse(path,lead):
    ann=wfdb.rdann(str(path),lead); times=(ann.sample.astype(int)*2).tolist();syms=ann.symbol
    if not set(syms)<=set(["(",")","N","p","t","u"]):raise ValueError((path,lead,set(syms)))
    ref={q:[] for q in POINTS}; pairs=[]; invalid=[]
    for i,sym in enumerate(syms):
        if sym not in ("N","p","t"): continue
        kind={"N":"QRS","p":"P","t":"T"}[sym]
        lo=times[i-1] if i and syms[i-1]=="(" else None
        hi=times[i+1] if i+1<len(syms) and syms[i+1]==")" else None
        if lo is not None:ref[kind+"_on"].append(lo)
        if hi is not None:ref[kind+"_off"].append(hi)
        if sym=="N":
            if lo is not None and hi is not None and 0<lo<hi<9998:pairs.append((lo,hi))
            else:invalid.append((lo,times[i],hi))
    return dict(span=[min(times),max(times)] if times else None,points=ref,qrs=pairs,incomplete_qrs=invalid,source_hash=sha(path.with_suffix("."+lead)))

def prepare(archive,work,split):
    raw=work/"raw";out=work/"outputs";out.mkdir(parents=True,exist_ok=True)
    files=extract(archive,raw)
    sos=butter(2,[.5,40],fs=250,btype="bandpass",output="sos")
    hold=set(map(str,json.loads(split.read_text(encoding="utf8"))["test"]));assert len(hold)==41
    subjects=[];signals=np.empty((200,12,2500),dtype=np.float32)
    for position,ident in enumerate(range(1,201)):
        rid=str(ident);path=raw/rid; sig,meta=wfdb.rdsamp(str(path)); leads=[n.lower() for n in meta["sig_name"]]
        assert meta["fs"]==500 and sig.shape==(5000,12) and set(LEADS)==set(leads)
        leads_data={}
        for j,lead in enumerate(LEADS):
            leads_data[lead]=parse(path,lead)
            a=sosfiltfilt(sos,resample_poly(sig[:,leads.index(lead)],1,2));q25,q75=np.quantile(a,[.25,.75]);scale=(q75-q25)/1.349
            if scale<1e-8:scale=a.std()+1e-8
            signals[position,j]=((a-np.median(a))/scale).astype(np.float32)
        subjects.append(dict(record=rid,subset="original41" if rid in hold else "additional159",leads=leads_data))
    np.save(work/"waves.npy",signals);(out/"reference.json").write_text(json.dumps(subjects,separators=(",",":")),encoding="utf8")
    audit={"archive_sha256":sha(archive),"official_source":"https://physionet.org/content/ludb/1.0.1/",
           "hash_verified_files":files,"participants":len(subjects),"lead_annotations":200*12,"split_sha256":sha(split),
           "records_all_12_annotation_hashes_distinct":sum(len(set(a["source_hash"] for a in s["leads"].values()))==12 for s in subjects),
           "empty_lead_annotations":sum(a["span"] is None for s in subjects for a in s["leads"].values()),
           "incomplete_or_edge_qrs_peaks":sum(len(a["incomplete_qrs"]) for s in subjects for a in s["leads"].values())}
    (out/"reference_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8");print(audit,flush=True)

def load(weights,head,seed):
    path=weights/f"{head}_{seed}.pt"; info=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
    assert info["head"]==head and info["steps"]==2400 and info["seed"]==seed and info["weight_sha256"]==sha(path)
    net=UNetKAN(where=["head"]) if head=="KAN" else UNetKAN(where=[],mlp_head=29)
    net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=True)
    return net.eval(),info["weight_sha256"]

@torch.inference_mode()
def infer(work,weights):
    out=work/"outputs";sig=np.load(work/"waves.npy",mmap_mode="r")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    expected=(200,12,2500);assert sig.shape==expected
    manifest={"protocol":sha(HERE/"protocol.txt"),"torch":str(torch.__version__),"wfdb":wfdb.__version__,
              "weights":{},"masks":{},"native_code_sha256":sha(HERE/"frozen_model.py")}
    for kind in ["KAN","GELU"]:
        for seed in (1,2,3):
            key=f"{kind}_{seed}"; net,h=load(weights,kind,seed);manifest["weights"][key]=h
            storage=out/f"{key}_masks.npz"
            masks=np.empty(expected,dtype=np.uint8)
            for record in range(len(sig)):
                for j in range(len(LEADS)):
                    masks[record,j]=predict_proba([net],sig[record,j]).argmax(axis=0)
            np.savez_compressed(storage,masks=masks);manifest["masks"][key]=sha(storage)
            print("Scored",key,flush=True)
    (out/"input_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("stage",choices=["prepare","infer","all"]);a.add_argument("--archive",type=Path);a.add_argument("--split",type=Path);a.add_argument("--weights",type=Path);a.add_argument("--work",type=Path,default=HERE);args=a.parse_args()
    assert sha(HERE/"protocol.txt")==PROTOCOL
    if args.stage in ("prepare","all"):prepare(args.archive,args.work,args.split)
    if args.stage in ("infer","all"):infer(args.work,args.weights)
