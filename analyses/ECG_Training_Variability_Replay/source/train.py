"""Train one model on one LUDB CV fold (subject-level) and evaluate on its validation fold.

Usage:
  python src/train.py --model unet --fold 0 [--epochs 40] [--seed 1] [--threads 6] [--tag base]
Writes: experiments/checkpoints/<tag>_<model>_f<fold>_s<seed>.pt, results/cv/<...>.json, appends experiments/log.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from evaluate import LUDB_POINTS, ludb_eval  # noqa: E402
from paths import ROOT  # noqa: E402

CROP = 1024 * data.FS // 250     # same 4.1 s crop at any sampling rate


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def augment(x: np.ndarray, rng) -> np.ndarray:
    x = x * rng.uniform(0.6, 1.6)
    n = len(x)
    t = np.arange(n) / data.FS
    if rng.random() < 0.5:   # baseline wander
        x = x + rng.uniform(0, 0.5) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 6.3))
    if rng.random() < 0.5:   # white noise
        x = x + rng.normal(0, rng.uniform(0.01, 0.15), n)
    if rng.random() < 0.2:   # powerline-like interference
        x = x + rng.uniform(0, 0.2) * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6.3))
    return x.astype(np.float32)


def batches(samples, bs, rng, steps):
    """Random 4.1-s crops around the annotated span, with amplitude, baseline, noise and mains augmentation."""
    for _ in range(steps):
        idx = rng.integers(0, len(samples), bs)
        xs, ys = [], []
        for i in idx:
            s = samples[i]
            s0, s1 = s.span
            lo = max(0, min(s0 - CROP // 4, len(s.x) - CROP))
            hi = max(lo, min(s1 - 3 * CROP // 4, len(s.x) - CROP))
            st = int(rng.integers(lo, hi + 1))
            xc, yc = s.x[st:st + CROP], s.y[st:st + CROP]
            xs.append(augment(xc, rng))
            ys.append(yc.astype(np.int64))
        yield torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))


def seg_loss(out, y):
    """Per-sample cross-entropy; samples labelled -1 (outside the annotated span) are ignored."""
    return F.cross_entropy(out, y, ignore_index=-1)


def mean_f1(table):
    keys = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
    return float(np.nanmean([table[k]["F1"] for k in keys]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold", type=int, required=True)        # 0..4 ; -1 = train on all 160 CV subjects
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)          # optimisation steps per epoch
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--kw", default="{}", help="JSON dict of model kwargs")
    ap.add_argument("--eval_every", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    kw = json.loads(a.kw)

    sp = splits.load()
    ludb = data.get("ludb")
    if a.fold >= 0:
        val_subj = set(sp["folds"][a.fold])
        tr_subj = {k for i, f in enumerate(sp["folds"]) if i != a.fold for k in f}
    else:
        val_subj = set()
        tr_subj = {k for f in sp["folds"] for k in f}
    assert not (tr_subj & set(sp["test"])) and not (val_subj & set(sp["test"])) and not (tr_subj & val_subj)
    tr = [s for s in ludb if s.subject in tr_subj and s.span[1] - s.span[0] > 0]
    va = [s for s in ludb if s.subject in val_subj]

    model = M.build(a.model, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    name = f"{a.tag}_{a.model}_f{a.fold}_s{a.seed}"
    ck_dir = ROOT / "experiments" / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "results" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    peak_rss = rss0
    t0 = time.perf_counter()
    cpu0 = proc.cpu_times()
    hist, best, best_state, deq_iters = [], -1.0, None, []
    step = 0
    for ep in range(a.epochs):
        model.train()
        losses = []
        for xb, yb in batches(tr, a.bs, rng, a.steps):
            out = model(xb)
            loss = seg_loss(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            peak_rss = max(peak_rss, proc.memory_info().rss)
            info = getattr(model, "last_info", None)
            if info and "n_iter" in info:
                deq_iters.append(int(info["n_iter"]))
        rec = {"epoch": ep + 1, "loss": float(np.mean(losses)), "time_s": time.perf_counter() - t0}
        if va and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            model.eval()
            tab = ludb_eval([model], va).table()
            rec["val_meanF1"] = mean_f1(tab)
            if rec["val_meanF1"] > best:
                best = rec["val_meanF1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "model": a.model, "kw": kw, "args": vars(a)}, ck_dir / f"{name}.pt")

    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times()
    res = {"name": name, "args": vars(a), "params": M.n_params(model), "wall_s": wall,
           "cpu_s": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
           "peak_rss_mb": peak_rss / 2**20, "rss_increase_mb": (peak_rss - rss0) / 2**20, "history": hist,
           "train_deq_iters_mean": float(np.mean(deq_iters)) if deq_iters else None}
    if va:
        model.eval()
        acc = ludb_eval([model], va)
        res["val"] = acc.table()
        res["val_meanF1"] = mean_f1(res["val"])
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    log = ROOT / "experiments" / "log.csv"
    with open(log, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), 3 if a.model in
                                 ("unet", "cnn_bilstm", "transformer") else 4, name, "src/train.py",
                                 json.dumps({k: v for k, v in vars(a).items() if k not in ("threads",)}),
                                 a.seed, round(res["cpu_s"], 1), round(res["peak_rss_mb"], 1),
                                 f"val_meanF1={res.get('val_meanF1', float('nan')):.2f}",
                                 f"params={res['params']}; wall_s={wall:.0f}"])
    print("DONE", name, "val_meanF1", res.get("val_meanF1"), "wall", round(wall))


if __name__ == "__main__":
    main()
