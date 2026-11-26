"""Subject-level LUDB split: 40-subject internal test + 5-fold CV on the remaining 160 (stratified by rhythm group).

Written once to configs/splits.json (seed 2026) and never regenerated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

SPLIT_FILE = ROOT / "configs" / "splits.json"


def rhythm_group(r: str) -> str:
    r = r.lower()
    if "fibrillation" in r or "flutter" in r:
        return "af_flutter"
    if r == "sinus rhythm":
        return "sinus"
    return "other_sinus"


def make(seed: int = 2026) -> dict:
    import data
    ludb = data.get("ludb")
    subj = {}
    for s in ludb:
        subj[s.subject] = rhythm_group(s.meta["rhythm"])
    rng = np.random.default_rng(seed)
    groups = {}
    for k, g in subj.items():
        groups.setdefault(g, []).append(k)
    test, folds = [], [[] for _ in range(5)]
    for g, ids in sorted(groups.items()):
        ids = sorted(ids, key=int)
        rng.shuffle(ids)
        n_test = int(round(len(ids) * 40 / 200))
        test += ids[:n_test]
        for i, k in enumerate(ids[n_test:]):
            folds[i % 5].append(k)
    out = {"seed": seed, "test": sorted(test, key=int), "folds": [sorted(f, key=int) for f in folds],
           "rhythm_group": subj}
    SPLIT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load() -> dict:
    if not SPLIT_FILE.exists():
        return make()
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    s = load()
    from collections import Counter
    print("test", len(s["test"]), Counter(s["rhythm_group"][k] for k in s["test"]))
    for i, f in enumerate(s["folds"]):
        print("fold", i, len(f), Counter(s["rhythm_group"][k] for k in f))
