"""stability and causal check of the KAN-head rule of U-Net-KAN.

For each of the five fold models:
  - normalised head inputs on the LUDB internal test (lead II of all 41 test subjects + all leads of 10 subjects);
  - P-selectivity of each channel = mean inside true P minus the largest mean in background/QRS/T;
  - the most P-selective channel, and the fraction of its phi_{P,i} range over which it is increasing;
  - ablation: that channel replaced by its mean (pre-hook on the head) -> change in P_on F1 on the internal test;
    control: the least P-selective channel ablated in the same way.
Output: results/kan_robustness.json, results/tables/kan_robustness.md
"""
from __future__ import annotations

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
from paths import RESULTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(3)


def load(fold):
    lr = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())["unet_kan"]
    ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{lr}_unet_kan_f{fold}_s1.pt", weights_only=False)
    m = M.build("unet_kan", **ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def p_f1(m, samples):
    return ludb_eval([m], samples).table()["P_on"]["F1"]


def main():
    test_subj = set(splits.load()["test"])
    allt = [s for s in data.get("ludb") if s.subject in test_subj]
    sel = sorted(test_subj)[:10]
    samples = [s for s in allt if s.lead == "ii" or s.subject in sel]
    out = []
    for fold in range(5):
        m = load(fold)
        feats = {}
        hook = m.head.register_forward_hook(lambda mod, inp, o: feats.__setitem__("h", inp[0].detach()))
        H, Y, R = [], [], []
        with torch.no_grad():
            for s in samples:
                m(torch.from_numpy(np.pad(s.x, (0, 2512 - len(s.x))))[None, None])
                R.append(feats["h"][0].T[:len(s.x)].numpy())                    # raw head inputs [L, C]
                H.append(m.head.kan.norm(feats["h"][0].T[:len(s.x)]).numpy())   # normalised inputs
                Y.append(s.y)
        hook.remove()
        H, Y, R = np.concatenate(H), np.concatenate(Y), np.concatenate(R)
        k = Y >= 0
        H, Y, R = H[k], Y[k], R[k]
        means = np.array([[H[Y == c, i].mean() for i in range(H.shape[1])] for c in range(4)])  # [class, ch]
        selectivity = means[1] - np.max(means[[0, 2, 3]], axis=0)
        ip, iq = int(np.argmax(selectivity)), int(np.argmin(selectivity))
        lo, hi = np.percentile(H[:, ip], 1), np.percentile(H[:, ip], 99)
        xs = torch.linspace(float(lo), float(hi), 200)
        with torch.no_grad():
            ph = m.head.kan.univariate(ip, 1, xs).numpy()
        incr = float((np.diff(ph) > 0).mean())
        base = p_f1(m, allt)
        res = {"fold": fold, "p_channel": ip, "p_selectivity": float(selectivity[ip]),
               "mean_in_P": float(means[1, ip]), "max_mean_other": float(np.max(means[[0, 2, 3], ip])),
               "phi_P_increasing_fraction": incr, "P_on_F1": base}
        for tag, ch in [("ablate_P_channel", ip), ("ablate_least_selective", iq)]:
            mu = float(R[:, ch].mean())

            def pre(mod, inp, ch=ch, mu=mu):
                x = inp[0].clone()
                x[:, ch, :] = mu
                return (x,)
            hk = m.head.register_forward_pre_hook(pre)
            res[tag + "_channel"] = ch
            res[tag + "_P_on_F1"] = p_f1(m, allt)
            hk.remove()
        out.append(res)
        print(res, flush=True)
    (RESULTS / "kan_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    rows = ["### U-Net-KAN head rule across the five fold models (LUDB internal test)", "",
            "| fold | P-selective channel | mean in P vs max other | φ_P increasing over | P_on F1 | P_on F1, P channel ablated | P_on F1, least selective channel ablated |",
            "|---|---|---|---|---|---|---|"]
    for r in out:
        rows.append(f"| {r['fold']} | {r['p_channel']} | {r['mean_in_P']:.2f} vs {r['max_mean_other']:.2f} | "
                    f"{100 * r['phi_P_increasing_fraction']:.0f}% | {r['P_on_F1']:.2f} | {r['ablate_P_channel_P_on_F1']:.2f} | "
                    f"{r['ablate_least_selective_P_on_F1']:.2f} |")
    (RESULTS / "tables" / "kan_robustness.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
