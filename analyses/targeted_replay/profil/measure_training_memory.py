"""Reproduce the paper's fresh-process Windows peak-working-set procedure."""
import argparse
import csv
import gc
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys


HERE=Path(__file__).resolve().parent
LABELS=["DEQ","Unrolled8","Unrolled18"]


def measure(label, resources):
    import psutil
    import torch
    from torch.nn.functional import cross_entropy
    sys.path.insert(0,str(resources.resolve()))
    import models
    assert Path(models.__file__).resolve() == (resources/"models.py").resolve()
    torch.manual_seed(0)
    torch.set_num_threads(6)
    if label == "DEQ":
        m = models.build("deq")
    else:
        m = models.build("deq_unrolled",unroll=int(label.removeprefix("Unrolled")))
    m.train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(32,1,1024)
    y = torch.randint(0,4,(32,1024))
    p = psutil.Process()
    gc.collect()
    baseline = p.memory_info().rss
    before_peak = p.memory_info().peak_wset
    for _ in range(5):
        out = m(x)
        loss = cross_entropy(out,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    peak = p.memory_info().peak_wset
    assert peak > before_peak
    return dict(label=label,threads=torch.get_num_threads(),params=models.n_params(m),
                rss_baseline_bytes=baseline,peak_before_steps_bytes=before_peak,peak_after_steps_bytes=peak,
                increase_mib=(peak-baseline)/2**20)


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument("--child",choices=LABELS)
    ap.add_argument("--resources",type=Path,default=HERE/"resources")
    ap.add_argument("--output",type=Path,default=HERE)
    args=ap.parse_args()
    if args.child:
        print(json.dumps(measure(args.child,args.resources)),flush=True)
        return
    rng=random.Random(20261125)
    rows=[]
    for repeat in range(1,4):
        ordered=LABELS.copy()
        rng.shuffle(ordered)
        for label in ordered:
            cmd=[sys.executable,str(Path(__file__).resolve()),"--child",label,"--resources",str(args.resources.resolve())]
            proc=subprocess.run(cmd,capture_output=True,text=True,check=True)
            result=json.loads(proc.stdout.strip().splitlines()[-1])
            result.update(repeat=repeat,order=len(rows)+1)
            rows.append(result)
            print(repeat,label,round(result["increase_mib"],2),flush=True)
    with (args.output/"training_memory_raw.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    median={s:statistics.median(r["increase_mib"] for r in rows if r["label"]==s) for s in LABELS}
    summary=dict(median_mib=median,
                 relative_deq_saving_pc={s:100*(1-median["DEQ"]/median[s]) for s in LABELS[1:]},
                 steps=5,threads=6,batch=32,samples=1024,seed=0,num_fresh_processes=9,
                 definition="Windows process lifetime peak working set after synthetic training minus resident set immediately before the five steps; MiB = 2^20 bytes")
    (args.output/"training_memory_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":
    run()
