"""Train the archive-replacement members, then evaluate 24 five-fold ensembles fit entirely on one GPU."""
import argparse
import os
from pathlib import Path
import time
import train_panel as panel
from prepare import batches

ROOT=Path(__file__).resolve().parent
original_path=panel.output_path
archive={Path(x.replace("\\","/")).name for x in (ROOT/"inventory.txt").read_text().splitlines() if x.strip()}

def homogeneous_weight(model,fold,seed):
    p=original_path(model,fold,seed)
    return ROOT/"homogeneous_weights"/p.name if p.name in archive else p

def configure():
    (ROOT/"homogeneous_weights").mkdir(exist_ok=True)
    panel.LOGS=ROOT/"homogeneous_logs";panel.output_path=homogeneous_weight

def fit(folds):
    x,rows=panel.metadata()
    for fold in folds:
        batches(fold,1)
        for seed in (1,2):
            for arch in panel.FAMILIES:
                if original_path(arch,fold,seed).name in archive:panel.train(arch,fold,seed,x,rows)

def score(cpu=False):
    second_outputs=ROOT/"cpu_harmonized"
    if cpu:
        panel.DEVICE="cpu";panel.ONE_LEAD_PER_CALL=True
        panel.RESULTS=second_outputs;second_outputs.mkdir(exist_ok=True)
    x,rows=panel.metadata();ix=[i for i,r in enumerate(rows) if r["subject"] in panel.SPLITS["test"]]
    x=x[ix];rows=[rows[i] for i in ix]
    pending=[(m,s) for s in (3,2,1) for m in panel.FAMILIES]
    while pending:
        ready=[(m,s) for m,s in pending if all(homogeneous_weight(m,f,s).is_file() for f in range(5))]
        for m,s in ready:
            # Seed 3 is byte-identical to the original mixed-panel ensemble for all
            # families; four seed-2 ensembles also have identical five-member hashes.
            # Reuse already evaluated counts only after exact member-hash verification.
            counterpart=(ROOT/"cpu_replay" if cpu else ROOT/"results")/f"ensemble_{m}_s{s}.npz"
            dest=panel.RESULTS/f"harmonized_ensemble_{m}_s{s}.npz"
            if counterpart.is_file() and not dest.exists():
                import numpy as np
                import shutil
                source=np.load(counterpart,allow_pickle=False)
                target_hashes=[panel.file_sha(homogeneous_weight(m,fold,s)) for fold in range(5)]
                if list(source["weights_sha256"])==target_hashes:
                    shutil.copyfile(counterpart,dest)
            panel.score(m,s,list(range(5)),x,rows,"harmonized_ensemble")
            pending.remove((m,s))
        if pending and not ready:time.sleep(10)

def guarded_gpu_score():
    # Let numerical analysis attach to an already running incremental scorer.
    # Advisory locks are released by the OS if the process is interrupted.
    with (ROOT/".uniform_gpu_scoring.lock").open("a+b") as lock:
        if os.name=="nt":
            import msvcrt
            if lock.tell()==0:
                lock.write(b"1");lock.flush()
            while True:
                try:
                    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:time.sleep(5)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        score(False)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--folds",nargs="+",type=int,choices=range(5))
    ap.add_argument("--score",action="store_true");ap.add_argument("--cpu-score",action="store_true")
    a=ap.parse_args();configure()
    if a.folds:fit(a.folds)
    if a.score:guarded_gpu_score()
    if a.cpu_score:score(True)
