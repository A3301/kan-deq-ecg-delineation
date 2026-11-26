"""Independent re-inference of all complete ensembles on CPU, one lead at a time."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_panel as p
from analyse import MODELS, macro_f1, write

ROOT=Path(__file__).resolve().parent

def replay():
    p.DEVICE="cpu";p.ONE_LEAD_PER_CALL=True;p.RESULTS=ROOT/"cpu_replay";p.RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    x,rows=p.metadata();ix=[i for i,r in enumerate(rows) if r["subject"] in p.SPLITS["test"]]
    x=x[ix];rows=[rows[i] for i in ix]
    pending=[(m,s) for s in [1,2,3] for m in MODELS]
    while pending:
        ready=[(m,s) for m,s in pending if all(p.output_path(m,f,s).is_file() for f in range(5))]
        for m,s in ready:
            p.score(m,s,list(range(5)),x,rows,"ensemble")
            pending.remove((m,s))
        if pending and not ready:time.sleep(7)

def compare(tag="ensemble"):
    result=[];differences=0
    cpu_dir="cpu_harmonized" if tag=="harmonized_ensemble" else "cpu_replay"
    for m in MODELS:
        for s in (1,2,3):
            f=f"{tag}_{m}_s{s}.npz";gpu=np.load(ROOT/"results"/f);cpu=np.load(ROOT/cpu_dir/f)
            assert np.array_equal(cpu["subjects"],gpu["subjects"])
            assert np.array_equal(cpu["weights_sha256"],gpu["weights_sha256"])
            c=cpu["counts"];g=gpu["counts"]
            assert np.array_equal(c[...,0]+c[...,1],g[...,0]+g[...,1])
            ff_c=macro_f1(c.sum(axis=1));ff_g=macro_f1(g.sum(axis=1))
            for i,t in enumerate([150,40]):
                affected=int(np.any(c[i]!=g[i],axis=-1).sum());differences+=affected
                result.append(dict(architecture=m,seed=s,tolerance_ms=t,cpu_f1=ff_c[i],gpu_f1=ff_g[i],
                                   gpu_minus_cpu_pp=ff_g[i]-ff_c[i],different_subject_landmark_triplets=affected))
    prefix="harmonized_" if tag=="harmonized_ensemble" else ""
    write(ROOT/"results"/(prefix+"cpu_gpu_inference_replay.csv"),result)
    summary=dict(ensembles=24,scores=48,participant_landmark_triplets_compared=24*2*41*6,
                 changed_triplets=differences,max_absolute_f1_difference_pp=max(abs(r["gpu_minus_cpu_pp"]) for r in result))
    (ROOT/"results"/(prefix+"cpu_gpu_inference_audit.json")).write_text(json.dumps(summary,indent=2));print(summary)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["replay","compare"])
    ap.add_argument("--tag",choices=["ensemble","harmonized_ensemble"],default="ensemble");a=ap.parse_args()
    if a.mode=="replay":replay()
    else:compare(a.tag)
