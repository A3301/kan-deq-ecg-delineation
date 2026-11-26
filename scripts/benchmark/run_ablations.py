"""Resumable ablation queue (frozen plan section 9), LUDB 5-fold CV, chosen lr of the parent model.

  KAN grid x order    : unet_kan grid {4,8,16} x order {1,3}       (grid 8 / order 3 = main model, reused)
  KAN placement       : head only, bottleneck only                  (both = main model)
  KAN basis           : rbf vs bspline (main)
  DEQ backward        : phantom (1-step) vs implicit (main)
Solver tol / max_iter, 12-lead fusion and 500 Hz are run by other scripts (deq_analysis.py, lead_fusion.py).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CV = ROOT / "results" / "cv"
LOGD = ROOT / "experiments" / "run_logs"

ABL = []
for g in [4, 8, 16]:
    for o in [1, 3]:
        if (g, o) != (8, 3):
            ABL.append((f"abl-g{g}o{o}", "unet_kan", {"grid": g, "order": o}))
ABL.append(("abl-headonly", "unet_kan", {"where": ["head"]}))
ABL.append(("abl-bnonly", "unet_kan", {"where": ["bottleneck"]}))
ABL.append(("abl-rbf", "unet_kan", {"basis": "rbf"}))
ABL.append(("abl-phantom", "deq_phantom", {}))


def run(tag, model, kw, fold, lr, threads):
    nm = f"{tag}_{model}_f{fold}_s1"
    if (CV / f"{nm}.json").exists():
        return
    cmd = [PY, str(ROOT / "src" / "train.py"), "--model", model, "--fold", str(fold), "--lr", lr, "--seed", "1",
           "--threads", str(threads), "--tag", tag, "--kw", json.dumps(kw)]
    t = time.time()
    with open(LOGD / f"{nm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    print(time.strftime("%H:%M:%S"), nm, f"rc={rc} {time.time() - t:.0f}s", flush=True)


def main(workers=2, threads=3):
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    jobs = []
    for f in range(5):
        for tag, model, kw in ABL:
            parent = "deq" if model == "deq_phantom" else model
            jobs.append((tag, model, kw, f, chosen[parent], threads))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: run(*j), jobs))
    print("ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
