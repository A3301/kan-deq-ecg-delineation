"""Prepare the LUDB Stage-1 inputs and exact shared crop streams without pickle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import data

HERE = Path(__file__).resolve().parent
CROP = 1024
POINTS = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def native():
    data.LUDB = HERE / "native"
    data.LUDB_DATA = data.LUDB / "data"
    leads = data.load_ludb()
    assert len(leads) == 2400 and data.FS == 250
    rows = []
    for s in leads:
        rows.append(dict(subject=str(s.subject),lead=s.lead,span=list(map(int,s.span)),
                         points={q:s.points[q].tolist() for q in POINTS}))
    root=HERE/"prepared";root.mkdir(exist_ok=True)
    np.save(root/"x.npy",np.stack([s.x for s in leads]))
    np.save(root/"y.npy",np.stack([s.y for s in leads]))
    (root/"rows.json").write_text(json.dumps(rows),encoding="utf8")
    print("Prepared",len(leads),"lead recordings from",len({s.subject for s in leads}),"subjects",flush=True)

def augment(x,rng):
    x=x*rng.uniform(.6,1.6)
    t=np.arange(len(x))/250.
    if rng.random()<.5:
        x=x+rng.uniform(0,.5)*np.sin(2*np.pi*rng.uniform(.05,.5)*t+rng.uniform(0,6.3))
    if rng.random()<.5:
        x=x+rng.normal(0,rng.uniform(.01,.15),len(x))
    if rng.random()<.2:
        x=x+rng.uniform(0,.2)*np.sin(2*np.pi*50*t+rng.uniform(0,6.3))
    return x.astype(np.float32)

def batches(fold,seed):
    root=HERE/"prepared"; dest=HERE/"batch_cache";dest.mkdir(exist_ok=True)
    key=f"f{fold}_s{seed}"
    done=dest/(key+".json")
    if done.is_file():return
    sp=json.loads((HERE/"splits.json").read_text());rows=json.loads((root/"rows.json").read_text())
    selected={s for i,subjects in enumerate(sp["folds"]) if i!=fold for s in subjects}
    assert not selected.intersection(sp["test"])
    order=[i for i,s in enumerate(rows) if s["subject"] in selected and s["span"][1]>s["span"][0]]
    x=np.load(root/"x.npy",mmap_mode="r");y=np.load(root/"y.npy",mmap_mode="r")
    rng=np.random.default_rng(seed)
    xp=dest/(key+".x.npy");yp=dest/(key+".y.npy")
    xx=np.lib.format.open_memmap(xp,mode="w+",dtype="float32",shape=(2400,32,CROP))
    yy=np.lib.format.open_memmap(yp,mode="w+",dtype="int8",shape=(2400,32,CROP))
    for step in range(2400):
        ids=rng.integers(0,len(order),32)
        for col,j in enumerate(ids):
            i=order[j];s0,s1=rows[i]["span"]
            lo=max(0,min(s0-CROP//4,x.shape[1]-CROP));hi=max(lo,min(s1-3*CROP//4,x.shape[1]-CROP))
            start=int(rng.integers(lo,hi+1));sl=slice(start,start+CROP)
            xx[step,col]=augment(x[i,sl],rng);yy[step,col]=y[i,sl]
    xx.flush();yy.flush();del xx,yy
    done.write_text(json.dumps({"x_sha256":digest(xp),"y_sha256":digest(yp),"training_subjects":len(selected),
                                "validation_subjects":len(sp["folds"][fold]),"seed":seed,"fold":fold}),encoding="utf8")
    print("Ready",key,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--native",action="store_true")
    ap.add_argument("--signals-only",action="store_true",help="Build only the arrays needed for inference.")
    a=ap.parse_args()
    if a.native or a.signals_only:native()
    if not a.signals_only:
        for fold in range(5):
            for seed in [2,3]:batches(fold,seed)
