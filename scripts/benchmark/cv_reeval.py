"""Re-evaluate every saved fold model on its own LUDB validation fold with the edge-tolerant metric
(evaluation rule of 2026-11-23), so that the CV summary and the 'best baseline' choice use one metric definition.
Checkpoints are those already selected during training (selection itself is not redone).
Output: results/cv_reeval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from evaluate import ludb_eval  # noqa: E402
from paths import RESULTS, ROOT  # noqa: E402

ONOFF = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]


def main(models):
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    sp = splits.load()
    ludb = data.get("ludb")
    path = RESULTS / "cv_reeval.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for m in models:
        scores = []
        for f in range(5):
            ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{chosen[m]}_{m}_f{f}_s1.pt", weights_only=False)
            net = M.build(ck["model"], **ck["kw"])
            net.load_state_dict(ck["state"])
            net.eval()
            va = [s for s in ludb if s.subject in set(sp["folds"][f])]
            tab = ludb_eval([net], va).table()
            scores.append(float(np.mean([tab[k]["F1"] for k in ONOFF])))
        out[m] = {"fold_meanF1": scores, "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1))}
        print(m, out[m], flush=True)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
