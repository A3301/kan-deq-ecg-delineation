"""Figures from saved outputs.
figures/deq/deq_iterations_difficulty.png: (a) iterations by database, (b) record-level mean iterations vs mean
|boundary error| on the LUDB internal test (n = 41 records), (c) ROC for flagging the worst-quartile LUDB leads by
iterations, predictive entropy and both (rank sum).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as ss  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGURES, RESULTS  # noqa: E402

plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})


def main():
    w = pd.read_csv(RESULTS / "deq_windows.csv")
    l = w[w.db == "LUDB"]
    g = l.groupby("record").agg(iters=("iters", "mean"), mae=("mae_ms", "mean"), group=("group", "first"))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    order, names = ["LUDB", "QTDB", "BUTPDB", "PWAVE"], ["LUDB", "QTDB", "BUT PDB", "P-wave"]
    ax[0].boxplot([w[w.db == d].iters for d in order], tick_labels=names)
    ax[0].axhline(30, ls=":", c="grey", lw=.8)
    ax[0].set_ylabel("Anderson iterations per window")
    ax[0].set_title("(a) Solver effort by database", fontsize=9)
    rho, p = ss.spearmanr(g.iters, g.mae)
    c = np.where(g.group == "sinus", "#1f77b4", "#d62728")
    ax[1].scatter(g.mae, g.iters, c=c, s=18)
    ax[1].scatter([], [], c="#1f77b4", s=18, label="sinus rhythm")
    ax[1].scatter([], [], c="#d62728", s=18, label="other rhythm")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_xlabel("record mean |boundary error| (ms)")
    ax[1].set_ylabel("record mean iterations")
    ax[1].set_title(f"(b) LUDB records (n = {len(g)}): Spearman ρ = {rho:.2f}", fontsize=9)
    y = (l.mae_ms >= l.mae_ms.quantile(0.75)).astype(int)
    for score, lab, col in [(l.iters, "iterations", "#d62728"), (l.entropy, "predictive entropy", "#1f77b4"),
                            (ss.rankdata(l.iters) + ss.rankdata(l.entropy), "both (rank sum)", "k")]:
        fpr, tpr, _ = roc_curve(y, score)
        ax[2].plot(fpr, tpr, color=col, lw=1.4, label=f"{lab}: AUROC {roc_auc_score(y, score):.2f}")
    ax[2].plot([0, 1], [0, 1], ls=":", c="grey", lw=.8)
    ax[2].set_xlabel("false-positive rate")
    ax[2].set_ylabel("true-positive rate")
    ax[2].legend(fontsize=7, loc="lower right")
    ax[2].set_title("(c) Flagging the worst-quartile LUDB leads", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "deq" / "deq_iterations_difficulty.png")
    plt.close(fig)
    print("written")


if __name__ == "__main__":
    main()
