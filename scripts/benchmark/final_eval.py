"""evaluate the 5-fold ensemble (and each fold model) of every trained model on
LUDB internal test, QTDB, BUT PDB and pwave, following the frozen plan. Saves per-record accumulators for stats.

Usage: python src/final_eval.py --models unet,deq,... [--threads 6]
Output: results/final/<model>.json (tables) and results/final/<model>_records.pkl (per-subject counts/errors)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from evaluate import ludb_eval, p_only_eval, qtdb_eval  # noqa: E402
from paths import ROOT  # noqa: E402

OUT = ROOT / "results" / "final"
OVERLAP_BUT = {"01", "02", "09", "13", "16", "27", "33", "36", "38"}  # BUT PDB segments from pwave records


def out_name(model: str) -> str:
    """'arch' -> 'arch'; 'arch@tag' -> 'arch__tag'."""
    return model.replace("@", "__").replace("~", "__seed")


def load_models(model: str):
    """model = '<arch>' (checkpoint lr<lr>_<arch>_f*) or '<arch>@<tag>' (checkpoint <tag>_<arch>_f*)."""
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    model, _, seed = model.partition("~")          # optional '~<seed>' suffix (additional: extra training seeds)
    seed = int(seed) if seed else 1
    arch, _, tag = model.partition("@")
    lr = chosen[arch]
    prefix = tag if tag else f"lr{lr}"
    ms = []
    for f in range(5):
        ck = torch.load(ROOT / "experiments" / "checkpoints" / f"{prefix}_{arch}_f{f}_s{seed}.pt", weights_only=False)
        m = M.build(ck["model"], **ck["kw"])
        m.load_state_dict(ck["state"])
        m.eval()
        ms.append(m)
    return ms, lr


def pack(acc):
    return {"table": acc.table(), "subj": {p: v["subj"] for p, v in acc.c.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tol", type=int, default=150)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    sp = splits.load()
    ludb = data.get("ludb")
    test = [s for s in ludb if s.subject in set(sp["test"])]
    qt, but, pw = data.get("qtdb"), data.get("butpdb"), data.get("pwave")
    suffix = "" if a.tol == 150 else f"_tol{a.tol}"
    for model in a.models.split(","):
        t0 = time.perf_counter()
        ms, lr = load_models(model)
        res, recs = {"model": model, "lr": lr, "params": M.n_params(ms[0]), "tol_ms": a.tol}, {}
        acc = ludb_eval(ms, test, a.tol)
        res["ludb_test"] = acc.table()
        recs["ludb_test"] = pack(acc)
        res["ludb_test_per_fold"] = [ludb_eval([m], test, a.tol).table() for m in ms]
        q = qtdb_eval(ms, qt, a.tol)
        for k, v in q.items():
            res[f"qtdb_{k}"] = v.table()
            recs[f"qtdb_{k}"] = pack(v)
        for nm, ds, excl in [("butpdb", but, ()), ("butpdb_nooverlap", but, OVERLAP_BUT), ("pwave", pw, ())]:
            r = p_only_eval(ms, ds, a.tol, exclude_records=excl)
            for k, v in r.items():
                res[f"{nm}_{k}"] = v.table()
                recs[f"{nm}_{k}"] = pack(v)
        res["eval_wall_s"] = time.perf_counter() - t0
        (OUT / f"{out_name(model)}{suffix}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        with open(OUT / f"{out_name(model)}{suffix}_records.pkl", "wb") as fh:
            pickle.dump(recs, fh)
        b = res["butpdb_each_lead"]["P_peak"]
        print(model, "LUDB test P_on F1 %.2f" % res["ludb_test"]["P_on"]["F1"], "| BUT P_peak F1 %.2f" % b["F1"],
              "| pwave %.2f" % res["pwave_each_lead"]["P_peak"]["F1"],
              "| QTDB P_on F1 %.2f" % res["qtdb_each_lead"]["P_on"]["F1"], f"| {res['eval_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
