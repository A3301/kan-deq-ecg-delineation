"""Channel-intervention control: the KAN-head P channel selected on validation data only, locked, and
tested causally on the LUDB internal test against random and energy-matched channel controls.

Per fold model of U-Net-KAN (seed given on the command line):
  - selection (validation leads of that fold only, never the test): normalised head inputs; P-selectivity = mean
    inside true P minus the largest mean inside background/QRS/T; locked channel = argmax;
  - replacement value of every channel = its mean raw head input over the validation leads;
  - energy-matched control = the other channel whose mean squared raw input on the validation leads is closest;
  - random control = each of the other seven channels (their average is the expectation of a random pick);
  - test: all 12 leads of the 41 internal-test subjects, F1 (+-150 ms) of the six onset/offset points per subject.
The encoder/decoder runs once per window; only the head is re-evaluated per condition, with the same windowing and
taper as evaluate.predict_proba.
Statistics: per subject, F1 change (ablated - intact) averaged over the five fold models; subject bootstrap (10 000).
Output: results/kan_channel_control_s{seed}.json, results/tables/kan_channel_control_s{seed}.md
Usage: python src/kan_channel_control.py [--seed 1] [--threads 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import extra_data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from evaluate import HOP, TOL_MS, WIN  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POINTS = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]


def load(fold, seed):
    lr = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())["unet_kan"]
    ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{lr}_unet_kan_f{fold}_s{seed}.pt", weights_only=False)
    m = M.build("unet_kan", **ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def window_feats(m, x):
    """Head inputs per window [B, C, W], window starts, taper (same scheme as predict_proba)."""
    n = len(x)
    if n <= WIN:
        L = max(16, int(np.ceil(n / 16) * 16))
        xp = np.zeros(L, np.float32)
        xp[:n] = x
        t, starts, taper = torch.from_numpy(xp)[None, None], [0], None
    else:
        starts = list(range(0, n - WIN + 1, HOP))
        if starts[-1] != n - WIN:
            starts.append(n - WIN)
        t = torch.from_numpy(np.stack([x[s:s + WIN] for s in starts]))[:, None]
        taper = np.hanning(WIN).astype(np.float32) + 1e-3
    z, s = m.enc(t)
    if "bottleneck" in m.where:
        z = z + m.kb(z)
    return m.dec(z, s), starts, taper


def merge(arr, starts, taper, n):
    """arr [B, K, W] -> [K, n] with the predict_proba taper (or crop for a single padded window)."""
    if taper is None:
        return arr[0, :, :n]
    acc = np.zeros((arr.shape[1], n), np.float32)
    w = np.zeros(n, np.float32)
    for j, s in enumerate(starts):
        acc[:, s:s + WIN] += arr[j] * taper
        w[s:s + WIN] += taper
    return acc / w


@torch.no_grad()
def select_channel(m, val):
    H, R, Y = [], [], []
    for s in val:
        f, st, tp = window_feats(m, s.x)
        raw = merge(f.numpy(), st, tp, len(s.x)).T                       # [n, C]
        R.append(raw)
        H.append(m.head.kan.norm(torch.from_numpy(raw)).numpy())
        Y.append(s.y)
    H, R, Y = np.concatenate(H), np.concatenate(R), np.concatenate(Y)
    k = Y >= 0
    H, R, Y = H[k], R[k], Y[k]
    means = np.array([[H[Y == c, i].mean() for i in range(H.shape[1])] for c in range(4)])
    sel = means[1] - np.max(means[[0, 2, 3]], axis=0)
    ip = int(np.argmax(sel))
    energy = (R ** 2).mean(0)
    others = [i for i in range(R.shape[1]) if i != ip]
    ie = min(others, key=lambda i: abs(energy[i] - energy[ip]))
    return ip, ie, sel, energy, R.mean(0)


@torch.no_grad()
def test_conditions(m, test, mu):
    C = len(mu)
    conds = ["intact"] + [f"ch{i}" for i in range(C)]
    accs = {c: Accumulator() for c in conds}
    tol = int(round(TOL_MS * data.FS / 1000))
    for s in test:
        f, st, tp = window_feats(m, s.x)
        n = len(s.x)
        s0, s1 = s.span
        for c in conds:
            g = f
            if c != "intact":
                g = f.clone()
                i = int(c[2:])
                g[:, i, :] = float(mu[i])
            p = torch.softmax(m.head(g), 1).numpy()
            prob = merge(p, st, tp, n)
            pred = mask_to_points(prob.argmax(0), s.x)
            for q in POINTS:
                if q not in s.points:
                    continue
                ref = s.points[q]
                ref = ref[(ref >= s0) & (ref < s1)]
                accs[c].add(q, s.subject, ref, pred[q], tol, fp_windows=[(s0, s1)])
    out = {}
    for c, a in accs.items():
        out[c] = {"pooled": {q: a.summarize(a.c[q])["F1"] for q in POINTS},
                  "subj": {q: {k: (200 * v["tp"] / (2 * v["tp"] + v["fn"] + v["fp"]) if 2 * v["tp"] + v["fn"] + v["fp"]
                                   else float("nan")) for k, v in a.c[q]["subj"].items()} for q in POINTS}}
    return out


def boot(d, rng, B=10000):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), (B, len(d)))
    return float(d.mean()), np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--db", default="ludb", choices=["ludb", "zj"],
                    help="test set: LUDB internal test or the independent Zhejiang database (selection always on LUDB validation)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sp = splits.load()
    ludb = data.get("ludb")
    if a.db == "ludb":
        test = [s for s in ludb if s.subject in set(sp["test"])]
    else:
        test = [s for s in extra_data.get("zj") if s.span[1] > s.span[0]]
    sfx = f"_s{a.seed}" + ("" if a.db == "ludb" else "_zj")
    subjects = sorted({s.subject for s in test}, key=str)
    folds = []
    for fold in range(5):
        m = load(fold, a.seed)
        val = [s for s in ludb if s.subject in set(sp["folds"][fold])]
        ip, ie, sel, energy, mu = select_channel(m, val)
        res = test_conditions(m, test, mu)
        folds.append({"fold": fold, "p_channel": ip, "energy_channel": ie, "selectivity": sel.tolist(),
                      "energy": energy.tolist(), "results": res})
        drops = {i: res[f"ch{i}"]["pooled"]["P_on"] - res["intact"]["pooled"]["P_on"] for i in range(len(mu))}
        print(f"fold {fold}: P channel {ip} (energy-matched {ie}); P_on intact {res['intact']['pooled']['P_on']:.2f}; "
              f"drops " + " ".join(f"{i}:{d:+.2f}" for i, d in drops.items()), flush=True)
    rng = np.random.default_rng(2026 + a.seed)

    def delta(q, which):
        """per-subject F1 change averaged over folds; which(fold) -> list of channels (averaged)."""
        D = []
        for f in folds:
            r = f["results"]
            base = r["intact"]["subj"][q]
            chans = which(f)
            D.append([np.mean([r[f"ch{i}"]["subj"][q][k] for i in chans]) - base[k] for k in subjects])
        return np.nanmean(np.array(D, float), 0)

    sel_f = lambda f: [f["p_channel"]]                                                    # noqa: E731
    en_f = lambda f: [f["energy_channel"]]                                                # noqa: E731
    rnd_f = lambda f: [i for i in range(len(f["energy"])) if i != f["p_channel"]]         # noqa: E731
    summary = {"seed": a.seed, "p_channels": [f["p_channel"] for f in folds],
               "energy_channels": [f["energy_channel"] for f in folds], "points": {}}
    for q in POINTS:
        ds, de, dr = delta(q, sel_f), delta(q, en_f), delta(q, rnd_f)
        pooled = lambda key: float(np.mean([f["results"][key(f)]["pooled"][q] for f in folds]))  # noqa: E731
        summary["points"][q] = {
            "F1_intact": pooled(lambda f: "intact"), "F1_selected": pooled(lambda f: f"ch{f['p_channel']}"),
            "F1_energy": pooled(lambda f: f"ch{f['energy_channel']}"),
            "F1_random_mean": float(np.mean([np.mean([f["results"][f"ch{i}"]["pooled"][q] for i in rnd_f(f)]) for f in folds])),
            "delta_selected": boot(ds, rng), "delta_energy": boot(de, rng), "delta_random": boot(dr, rng),
            "selected_minus_energy": boot(ds - de, rng), "selected_minus_random": boot(ds - dr, rng),
            "subjects_worse_selected": int((ds < 0).sum()), "n_subjects": int((~np.isnan(ds)).sum())}
    ranks = []
    for f in folds:
        r = f["results"]
        drops = [r[f"ch{i}"]["pooled"]["P_on"] - r["intact"]["pooled"]["P_on"] for i in range(len(f["energy"]))]
        ranks.append(int(np.argsort(drops).tolist().index(f["p_channel"]) + 1))
    summary["rank_of_selected_by_P_on_drop"] = ranks
    rob = RESULTS / "kan_robustness.json"
    if rob.exists() and a.seed == 1:
        summary["test_selected_channels_previous"] = [r["p_channel"] for r in json.loads(rob.read_text())]
    (RESULTS / f"kan_channel_control{sfx}.json").write_text(json.dumps({"summary": summary, "folds": folds}, indent=1),
                                                                 encoding="utf-8")
    fm = lambda t: f"{t[0]:+.2f} [{t[1][0]:+.2f}, {t[1][1]:+.2f}]"                       # noqa: E731
    md = [f"### KAN-head P channel selected on validation data and locked (U-Net-KAN, seed {a.seed}, " + ("LUDB internal test" if a.db == "ludb" else "Zhejiang, each lead") + ")", "",
          f"Selected channel per fold: {summary['p_channels']}; energy-matched control: {summary['energy_channels']}; "
          f"rank of the selected channel among the 8 by P-onset F1 drop: {ranks}.", "",
          "| point | F1 intact | selected channel ablated | energy-matched ablated | random channel (mean of 7) | Δ selected [95% CI] | Δ energy-matched | Δ random | selected − energy-matched | selected − random |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for q, v in summary["points"].items():
        md.append(f"| {q} | {v['F1_intact']:.2f} | {v['F1_selected']:.2f} | {v['F1_energy']:.2f} | {v['F1_random_mean']:.2f} | "
                  f"{fm(v['delta_selected'])} | {fm(v['delta_energy'])} | {fm(v['delta_random'])} | "
                  f"{fm(v['selected_minus_energy'])} | {fm(v['selected_minus_random'])} |")
    md.append("")
    md.append("Δ: per-subject F1 change (ablated − intact), averaged over the five fold models; mean over subjects (records) with "
              "subject bootstrap 95% CI. Pooled F1 columns are averages of the five fold models.")
    (RESULTS / "tables" / f"kan_channel_control{sfx}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written", flush=True)


if __name__ == "__main__":
    main()
