"""Run one targeted analysis into a new working directory and check its output."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parent


def verify_package():
    for line in (ROOT/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ",1)
        if sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(name)
    print("All files listed in SHA256SUMS.txt verified.")


def same_csv(path, published):
    with path.open(encoding="utf-8-sig",newline="") as a, published.open(encoding="utf-8-sig",newline="") as b:
        if list(csv.reader(a))!=list(csv.reader(b)):
            raise ValueError(f"Recomputed output differs: {path.name}")
    print(f"Recomputed output matches {published.relative_to(ROOT)}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["verify","paired","channels","hybrid","latency","memory"])
    ap.add_argument("--data",type=Path,default=ROOT/"kan_gelu"/"LUDB")
    ap.add_argument("--weights",type=Path,default=ROOT/"channel_weights")
    ap.add_argument("--out",type=Path,default=ROOT/"recomputed")
    a=ap.parse_args()
    verify_package()
    if a.stage=="verify":
        return
    out=a.out/a.stage
    out.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    if a.stage=="paired":
        name="paired_training_runs.csv"
        subprocess.run([py,ROOT/"paired_training_runs.py",ROOT/"head_counts.csv",out/name],check=True)
        same_csv(out/name,ROOT/name)
    elif a.stage in ("channels","hybrid"):
        args=[py,ROOT/"kan_gelu"/"channel_test.py","--data",a.data,"--weights",a.weights,
              "--splits",ROOT/"kan_gelu"/"splits.json","--out",out]
        if a.stage=="hybrid":
            args.append("--hybrid-only")
        subprocess.run(args,check=True)
        saved=ROOT/"kan_gelu"/("hybrid" if a.stage=="hybrid" else "results")
        for name in ("intervention_counts.csv","selected_channels.csv","intervention_estimates.csv"):
            same_csv(out/name,saved/name)
    else:
        prof=ROOT/"profil"
        if a.stage=="latency":
            args=[py,prof/"measure_latency.py","--ludb",a.data,"--splits",ROOT/"kan_gelu"/"splits.json"]
        else:
            args=[py,prof/"measure_training_memory.py"]
        subprocess.run([*args,"--resources",prof/"resources","--output",out],check=True)


if __name__=="__main__":
    main()
