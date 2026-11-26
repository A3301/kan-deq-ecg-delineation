"""Phase 4: visualise learned univariate KAN functions of the U-Net-KAN classification head.

The head maps the 8 decoder channels (per time step) to 4 class logits through phi_{j,i}. For each class j we plot
phi_{j,i}(x) for all inputs i, and summarise each input channel's importance as the range (max-min) of phi over the
observed input distribution. We also relate each input channel to the waves by its mean activation inside
P / QRS / T / background regions of LUDB internal-test signals (fold-0 model).
Outputs: figures/kan/*.png, results/kan_interpretation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models as M  # noqa: E402
import splits  # noqa: E402
from paths import FIGURES, RESULTS, ROOT  # noqa: E402

CLS = ["background", "P", "QRS", "T"]


def main(fold=0):
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{chosen['unet_kan']}_unet_kan_f{fold}_s1.pt",
                    weights_only=False)
    m = M.build("unet_kan", **ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    head = m.head.kan
    feats = {}
    m.head.register_forward_hook(lambda mod, inp, out: feats.__setitem__("h", inp[0].detach()))
    sp = splits.load()
    test = [s for s in data.get("ludb") if s.subject in set(sp["test"])][:240]
    H, Y = [], []
    with torch.no_grad():
        for s in test:
            x = torch.from_numpy(np.pad(s.x, (0, 2512 - len(s.x))))[None, None]
            m(x)
            h = head.norm(feats["h"][0].T[:len(s.x)])   # normalised inputs to the spline layer [L, C]
            H.append(h.numpy())
            Y.append(s.y)
    H = np.concatenate(H)
    Y = np.concatenate(Y)
    keep = Y >= 0
    H, Y = H[keep], Y[keep]
    C = H.shape[1]
    lo, hi = np.percentile(H, 1), np.percentile(H, 99)
    xs = torch.linspace(float(lo), float(hi), 200)
    out_dir = FIGURES / "kan"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    importance = np.zeros((4, C))
    mb = np.stack([H[Y == c].mean(0) for c in range(4)])
    p_ch = int(np.argmax(mb[1] - np.max(mb[[0, 2, 3]], axis=0)))       # most P-selective input channel
    for j in range(4):
        for i in range(C):
            qlo, qhi = np.percentile(H[:, i], [1, 99])
            xi = torch.linspace(float(qlo), float(qhi), 200)               # each function over its own input range
            ph = head.univariate(i, j, xi).numpy()
            importance[j, i] = float(ph.max() - ph.min())
            axs[j].plot(xi.numpy(), ph, lw=2.8 if i == p_ch else 1.0, color="k" if i == p_ch else None,
                        alpha=1 if i == p_ch else .75, label=f"ch{i}" + (" (P-selective)" if i == p_ch else ""))
        axs[j].set_title(rf"{CLS[j]} logit: $\varphi_{{{CLS[j]},i}}(x_i)$", fontsize=10)
        axs[j].set_xlabel("normalised input $x_i$")
    axs[0].set_ylabel("contribution to logit")
    axs[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "unet_kan_head_functions.png", dpi=150)
    plt.close(fig)
    # channel-wave association
    mean_by_cls = np.stack([H[Y == c].mean(0) for c in range(4)])           # [4, C]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    im = ax[0].imshow(importance, aspect="auto", cmap="viridis")
    ax[0].set_yticks(range(4), CLS)
    ax[0].set_xlabel("input channel")
    ax[0].set_title("range of phi (importance)")
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(mean_by_cls, aspect="auto", cmap="coolwarm")
    ax[1].set_yticks(range(4), CLS)
    ax[1].set_xlabel("input channel")
    ax[1].set_title("mean normalised input by true wave")
    plt.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out_dir / "unet_kan_head_importance.png", dpi=150)
    plt.close(fig)
    # top channel per class + monotonicity of its function
    summ = {}
    for j in range(4):
        i = int(np.argmax(importance[j]))
        ph = head.univariate(i, j, xs).numpy()
        d = np.diff(ph)
        summ[CLS[j]] = {"top_channel": i, "importance": float(importance[j, i]),
                        "monotone_increasing_frac": float((d > 0).mean()),
                        "channel_mean_by_true_wave": dict(zip(CLS, mean_by_cls[:, i].round(3).tolist()))}
    res = {"fold": fold, "n_samples": int(len(Y)), "importance": importance.round(4).tolist(), "summary": summ}
    (RESULTS / "kan_interpretation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
