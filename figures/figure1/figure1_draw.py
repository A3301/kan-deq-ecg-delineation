# -*- coding: utf-8 -*-
"""Figure 1: the three proposed architectures, drawn from the model definitions (src/models.py), with real data
from the stored weights: learned KAN head functions, the DEQ solver residuals and the five-fold ensemble outputs on
one LUDB internal-test lead (all written by figure1_data.py). Every number in the drawing comes from the code or
from those files.
usage: python figure1_draw.py <dir with figure1_data.*> <Figure_2_Spline_Evaluations.csv> <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse
from matplotlib.lines import Line2D
import pandas as pd

D, SPL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.load(open(D / "figure1_data.json"))
z = np.load(D / "figure1_data.npz")
spl = pd.read_csv(SPL)

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})
INK, MUTED, SKIP = "#1d2b3a", "#4a5868", "#8fa3b8"
CONV, CONV_E = "#dce8f6", "#2f6db5"          # ordinary convolution blocks
KAN, KAN_E = "#d3efe6", "#0f8a6d"            # KAN layers
DEQ, DEQ_E = "#fde7c8", "#d9822b"            # equilibrium core
CLS_COL = {1: "#b45fbf", 2: "#e4572e", 3: "#2a9fd6"}   # P, QRS, T
CLS_NAME = {1: "P", 2: "QRS", 3: "T"}

W, H = 7.48, 9.0
PH = 0.192                    # height of a diagram panel (figure fraction)
YR = (-2.0, 42.0)             # data range of a diagram panel
fig = plt.figure(figsize=(W, H))
ASP = (W / 100) / (H * PH / (YR[1] - YR[0]))   # inches per x unit / inches per y unit


# ----------------------------------------------------------------------------------------------- helpers
def box(ax, x, y, w, h, fc, ec, title="", sub="", tsize=6.4, ssize=5.4, lw=0.9, glyph=False, zo=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                zorder=zo, mutation_aspect=ASP))
    if glyph:                                   # small learned-curve icon for KAN layers
        t = np.linspace(0, 1, 40)
        gy = y + h - 1.5 if title else y + h * 0.5
        ax.plot(x + w * (0.2 + 0.6 * t), gy + 0.75 * np.sin(2 * np.pi * (1.3 * t + 0.1)) * (0.4 + t),
                color=ec, lw=0.8, alpha=0.8, zorder=zo + 1, solid_capstyle="round")
    ty = y + h / 2
    if title:
        ty = (y + h - (3.6 if glyph else 2.0)) if sub else (y + h / 2 - (0.8 if glyph else 0))
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK,
                zorder=zo + 2)
    if sub:
        sy = y + (ty - 1.4 - y) / 2 if title else y + h / 2
        ax.text(x + w / 2, sy, sub, ha="center", va="center", fontsize=ssize, color=INK, zorder=zo + 2,
                linespacing=1.15)


def arrow(ax, p, q, color=INK, lw=0.8, rad=0.0, ms=6, zo=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zo, shrinkA=0, shrinkB=0))


def shape_label(ax, x, y, c, n, color=MUTED):
    ax.text(x, y, f"{c}×{n}", ha="center", va="top", fontsize=5.1, color=color)


def blk_size(c, n):
    return 2.6 + 3.2 * np.sqrt(c / 64), 3.2 + 6.6 * np.sqrt(n / 1024)     # width ~ channels, height ~ length


def skip(ax, a, b, yy):
    ax.plot([a[0] + a[2] / 2, a[0] + a[2] / 2, b[0] + b[2] / 2], [a[1] + a[3], yy, yy], color=SKIP, lw=0.7,
            ls=(0, (3, 2)), zorder=1)
    arrow(ax, (b[0] + b[2] / 2, yy), (b[0] + b[2] / 2, b[1] + b[3]), color=SKIP, lw=0.7, ms=5, zo=1)


def panel(top, letter, name, extra):
    ax = fig.add_axes([0.0, top - PH, 1.0, PH])
    ax.set_xlim(0, 100); ax.set_ylim(*YR); ax.axis("off")
    ax.text(1.0, YR[1] - 0.6, letter, fontsize=10, fontweight="bold", color=INK, va="top")
    ax.text(4.4, YR[1] - 0.75, name, fontsize=8.4, fontweight="bold", color=INK, va="top")
    ax.text(4.4, YR[1] - 4.6, extra, fontsize=6.0, color=MUTED, va="top")
    return ax


def small_axes(rect, letter, ttl):
    ax = fig.add_axes(rect)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, labelsize=5.6, pad=1.5)
    fig.text(rect[0] - 0.055, rect[1] + rect[3] + 0.028, letter, fontsize=10, fontweight="bold", color=INK,
             va="bottom")
    fig.text(rect[0] - 0.022, rect[1] + rect[3] + 0.028, ttl, fontsize=6.4, color=INK, va="bottom",
             fontweight="bold")
    return ax


rec = meta["record"]
P = meta["params"]; AG = meta["agreement_with_reference"]

# ================================================================================ (A) U-Net-KAN
axA = panel(0.995, "A", "U-Net-KAN",
            f"{P['unet_kan']:,} parameters · convolutional U-Net (kernel 9) · two residual point-wise KAN layers in "
            "the bottleneck · point-wise KAN head · cubic B-splines, grid 8")
enc = [(8, 1024), (16, 512), (32, 256), (64, 128), (64, 64)]
dec = [(64, 128), (32, 256), (16, 512), (8, 1024)]
ylev = [21.0, 15.5, 10.0, 4.5, 0.0]
xe = [3.0, 10.0, 17.0, 24.0, 31.0]
geo = {}
for i, (c, n) in enumerate(enc):
    w, h = blk_size(c, n); y = ylev[i] + (9.8 - h) / 2 if i < 4 else 0.0
    box(axA, xe[i], y, w, h, CONV, CONV_E); geo[("e", i)] = (xe[i], y, w, h)
    shape_label(axA, xe[i] + w / 2, y - 0.4, c, n)
kx = [40.0, 48.5]
for j, x in enumerate(kx):
    box(axA, x, -0.6, 6.8, 8.0, KAN, KAN_E, f"KAN {j + 1}", "64→64", glyph=True, tsize=6.0, ssize=5.2)
e4 = geo[("e", 4)]
arrow(axA, (e4[0] + e4[2], 3.0), (kx[0], 3.0))
arrow(axA, (kx[0] + 6.8, 3.0), (kx[1], 3.0))
axA.add_patch(FancyArrowPatch((37.6, 5.0), (57.3, 4.3), arrowstyle="-|>", mutation_scale=6, color=KAN_E, lw=0.8,
                              connectionstyle="arc3,rad=-0.4", zorder=2))
axA.add_patch(Ellipse((57.3, 3.0), 1.8, 1.8 * ASP, fc="white", ec=KAN_E, lw=0.8, zorder=4))
axA.text(57.3, 3.05, "+", ha="center", va="center", fontsize=7, color=KAN_E, zorder=5, fontweight="bold")
arrow(axA, (kx[1] + 6.8, 3.0), (56.4, 3.0))
axA.text(47.6, 8.6, r"residual: $z + \mathrm{KAN}_2(\mathrm{KAN}_1(z))$", ha="center", fontsize=5.4, color=KAN_E,
         bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
xd = [60.5, 67.5, 74.5, 81.5]
for i, (c, n) in enumerate(dec):
    lev = 3 - i
    w, h = blk_size(c, n); y = ylev[lev] + (9.8 - h) / 2
    box(axA, xd[i], y, w, h, CONV, CONV_E); geo[("d", lev)] = (xd[i], y, w, h)
    shape_label(axA, xd[i] + w / 2, y - 0.4, c, n)
d3 = geo[("d", 3)]
arrow(axA, (58.2, 3.0), (xd[0] + 0.5, d3[1] + 0.4))
for i in range(4):
    a = geo[("e", i)]; b = geo[("e", i + 1)]
    arrow(axA, (a[0] + a[2] * 0.75, a[1]), (b[0] + 0.3, b[1] + b[3] * 0.85))
for lev in [3, 2, 1]:
    a = geo[("d", lev)]; b = geo[("d", lev - 1)]
    arrow(axA, (a[0] + a[2] * 0.6, a[1] + a[3]), (b[0] + 0.2, b[1] + b[3] * 0.2))
for lev in range(4):
    a = geo[("e", lev)]; b = geo[("d", lev)]
    skip(axA, a, b, a[1] + a[3] + 0.9)
axA.text(9.0, 3.0, "max-pool ×2", fontsize=5.2, color=MUTED, ha="left")
axA.text(68.5, 3.0, "transposed conv ×2", fontsize=5.2, color=MUTED, ha="left")
d0 = geo[("d", 0)]
box(axA, 89.5, ylev[0] + 1.0, 8.6, 8.0, KAN, KAN_E, "KAN head", "8→4", glyph=True, tsize=6.0, ssize=5.2)
arrow(axA, (d0[0] + d0[2], ylev[0] + 5.0), (89.5, ylev[0] + 5.0))
axA.text(93.8, ylev[0] + 0.4, "LayerNorm, then\nφ = SiLU + B-spline\nper input–class pair", ha="center",
         va="top", fontsize=5.0, color=KAN_E, linespacing=1.15)
axA.text(93.8, ylev[0] + 9.6, "4×1024 logits", ha="center", fontsize=5.1, color=MUTED)

# ================================================================================ (B) ConvKAN
axB = panel(0.79, "B", "ConvKAN",
            f"{P['convkan']:,} parameters · every layer is a KAN over the flattened receptive field "
            r"($k \times C_\mathrm{in} \rightarrow C_\mathrm{out}$) · Gaussian RBF basis, grid 6 · average pooling")
encB = [(8, 1024), (16, 256), (32, 128), (32, 64)]
decB = [(32, 128), (16, 256), (8, 1024)]
ylB = [21.0, 14.0, 7.0, 0.0]
xeB = [3.0, 13.0, 23.0, 33.5]
gB = {}
for i, (c, n) in enumerate(encB):
    w, h = blk_size(c, n); y = ylB[i] + (9.8 - h) / 2 if i < 3 else 0.0
    box(axB, xeB[i], y, w, h, KAN, KAN_E, glyph=True); gB[("e", i)] = (xeB[i], y, w, h)
    shape_label(axB, xeB[i] + w / 2, y - 0.4, c, n, KAN_E)
xdB = [55.0, 66.0, 77.0]
for i, (c, n) in enumerate(decB):
    lev = 2 - i
    w, h = blk_size(c, n); y = ylB[lev] + (9.8 - h) / 2
    box(axB, xdB[i], y, w, h, KAN, KAN_E, glyph=True); gB[("d", lev)] = (xdB[i], y, w, h)
    shape_label(axB, xdB[i] + w / 2, y - 0.4, c, n, KAN_E)
for i in range(3):
    a = gB[("e", i)]; b = gB[("e", i + 1)]
    arrow(axB, (a[0] + a[2] * 0.75, a[1]), (b[0] + 0.3, b[1] + b[3] * 0.85))
axB.text(3.0, 1.2, "down: average pooling ×4, ×2, ×2", fontsize=5.0, color=MUTED, ha="left")
b3 = gB[("e", 3)]; d2 = gB[("d", 2)]
arrow(axB, (b3[0] + b3[2], b3[1] + b3[3] * 0.6), (d2[0], d2[1] + d2[3] * 0.3))
for lev in [2, 1]:
    a = gB[("d", lev)]; b = gB[("d", lev - 1)]
    arrow(axB, (a[0] + a[2] * 0.6, a[1] + a[3]), (b[0] + 0.2, b[1] + b[3] * 0.2))
axB.text(63.0, 1.2, "up: linear interpolation, concatenate skip", fontsize=5.0, color=MUTED, ha="left")
for lev in range(3):
    skip(axB, gB[("e", lev)], gB[("d", lev)], gB[("e", lev)][1] + gB[("e", lev)][3] + 0.9)
axB.text(43.0, 26.6, "every block: KANConv, k = 5", fontsize=5.8, color=KAN_E, ha="center", fontweight="bold")
axB.text(43.0, 19.4, r"each output channel: $\sum_i\,[\,w_i\,\mathrm{SiLU}(u_i)+\sum_{g=1}^{6} c_{ig}\,"
         r"e^{-((u_i-\mu_g)/\sigma)^2}\,]\,+\,b$", fontsize=5.4, color=KAN_E, ha="center", va="center")
d0 = gB[("d", 0)]
box(axB, 87.5, ylB[0] + 1.0, 11.6, 8.0, KAN, KAN_E, "KANConv head", "k = 1, 8→4", glyph=True, tsize=5.9,
    ssize=5.2)
arrow(axB, (d0[0] + d0[2], ylB[0] + 5.0), (87.5, ylB[0] + 5.0))
axB.text(93.3, ylB[0] + 9.6, "4×1024 logits", ha="center", fontsize=5.1, color=MUTED)

# ================================================================================ (C) DEQ
axC = panel(0.585, "C", "Deep equilibrium model (DEQ)",
            f"{P['deq']:,} parameters · one weight-tied block solved to a fixed point by Anderson acceleration · "
            "implicit differentiation in training")
yc = 8.5
box(axC, 2.0, yc, 8.4, 9.2, CONV, CONV_E, "Stem", "conv block\n1→16", tsize=6.1, ssize=5.2)
shape_label(axC, 6.2, yc - 0.4, 16, 1024)
box(axC, 13.2, yc, 8.6, 9.2, CONV, CONV_E, "Down", "pool, 16→32\npool, 32→48", tsize=6.1, ssize=5.2)
shape_label(axC, 17.5, yc - 0.4, 48, 256)
box(axC, 24.6, yc, 7.8, 9.2, CONV, CONV_E, "Inject", "1×1 conv\n48→48", tsize=6.0, ssize=5.2)
arrow(axC, (10.4, yc + 4.6), (13.2, yc + 4.6)); arrow(axC, (21.8, yc + 4.6), (24.6, yc + 4.6))
cx0, cw = 35.5, 29.0
box(axC, cx0, yc - 5.4, cw, 18.6, DEQ, DEQ_E, lw=1.1)
axC.text(cx0 + cw / 2, yc + 10.6, r"$z^{*} = f_{\theta}(z^{*},\,h)$", ha="center", fontsize=8.2, color=INK)
box(axC, cx0 + 2.4, yc - 0.6, cw - 4.8, 7.8, "white", DEQ_E, lw=0.8, zo=4)
axC.text(cx0 + cw / 2, yc + 5.1, r"tied block $f_{\theta}(z, h)$, 48 channels", ha="center", fontsize=5.9,
         fontweight="bold", color=INK, zorder=6)
axC.text(cx0 + cw / 2, yc + 1.6, "conv k9 → ReLU → GN → conv k9 (+h) → GN\n→ (+z) → ReLU → GN; weight-normalised",
         ha="center", va="center", fontsize=4.9, color=INK, zorder=6, linespacing=1.3)
axC.add_patch(FancyArrowPatch((cx0 + cw - 2.6, yc + 7.2), (cx0 + 2.6, yc + 7.2), arrowstyle="-|>",
                              mutation_scale=6, color=DEQ_E, lw=0.9, connectionstyle="arc3,rad=0.5", zorder=5))
axC.text(cx0 + cw / 2, yc - 3.2, r"Anderson (memory 5) · tolerance $10^{-3}$ · ≤ 30 iterations" + "\n"
         "backward: adjoint fixed point (≤ 30), no stored trajectory", ha="center", va="center", fontsize=4.9,
         color="#8a4b0f", linespacing=1.3)
arrow(axC, (32.4, yc + 4.6), (cx0, yc + 4.6))
axC.text(33.9, yc + 5.4, "h", fontsize=6.4, color=INK, ha="center", style="italic")
box(axC, 67.5, yc, 8.6, 9.2, CONV, CONV_E, "Up", "×4 upsample\n1×1, 48→16", tsize=6.1, ssize=5.2)
shape_label(axC, 71.8, yc - 0.4, 16, 1024)
box(axC, 79.0, yc, 8.4, 9.2, CONV, CONV_E, "Refine", "concat\n32→16", tsize=6.1, ssize=5.2)
box(axC, 90.3, yc, 7.8, 9.2, CONV, CONV_E, "Head", "1×1 conv\n16→4", tsize=6.1, ssize=5.2)
arrow(axC, (cx0 + cw, yc + 4.6), (67.5, yc + 4.6)); arrow(axC, (76.1, yc + 4.6), (79.0, yc + 4.6))
arrow(axC, (87.4, yc + 4.6), (90.3, yc + 4.6))
axC.text(94.2, yc - 0.4, "4×1024\nlogits", fontsize=5.1, color=MUTED, ha="center", va="top", linespacing=1.1)
axC.plot([6.2, 6.2, 83.2], [yc + 9.2, 29.5, 29.5], color=SKIP, lw=0.7, ls=(0, (3, 2)), zorder=1)
arrow(axC, (83.2, 29.5), (83.2, yc + 9.2), color=SKIP, lw=0.7, ms=5, zo=1)
axC.text(44.0, 30.2, "full-rate skip (16 channels)", fontsize=5.1, color="#7d90a5", ha="center")
axC.text(50.0, -0.6, "controls: the same block unrolled 8 or 18 times with full backpropagation "
         "(103,348 parameters); two untied blocks (145,300 parameters)", ha="center", fontsize=5.0, color=MUTED)

# ================================================================== (D–F) mechanisms read from the stored weights
row_y, row_h = 0.235, 0.105
axD = small_axes([0.075, row_y, 0.245, row_h], "D", "U-Net-KAN head, P-selective channel")
for lg, col, nm in [("P wave", CLS_COL[1], "P"), ("QRS", CLS_COL[2], "QRS"), ("T wave", CLS_COL[3], "T"),
                    ("Background", "#9aa5b1", "background")]:
    d = spl[(spl.channel == 2) & (spl.logit == lg)].sort_values("x")
    axD.plot(d.x, d.contribution, color=col, lw=1.3 if nm == "P" else 0.9)
    axD.text(d.x.iloc[-1] + 0.1, d.contribution.iloc[-1], nm, color=col, fontsize=5.3, va="center")
axD.axhline(0, color="#c3ccd5", lw=0.5, zorder=0)
axD.set_xlabel("normalised input (channel 2)", fontsize=5.8, labelpad=1)
axD.set_ylabel("contribution φ to logit", fontsize=5.8, labelpad=1)
axD.set_xlim(-1.2, 6.3)

axE = small_axes([0.405, row_y, 0.245, row_h], "E", "ConvKAN head: basis and learned φ")
xs = z["convkan_xs"]; rb = meta["convkan_rbf"]
for c in rb["centers"]:
    b = np.exp(-((xs - c) * rb["inv_w"]) ** 2)
    axE.fill_between(xs, 0, b, color=KAN_E, alpha=0.08, lw=0)
    axE.plot(xs, b, color=KAN_E, lw=0.6, alpha=0.7)
axE.set_ylim(0, 1.08); axE.set_xlim(-2, 2)
axE.set_xlabel("normalised input", fontsize=5.8, labelpad=1)
axE.set_ylabel("RBF basis (grid 6)", fontsize=5.8, labelpad=1, color=KAN_E)
ax2 = axE.twinx()
phi = z["convkan_phi_P"]
for i in range(phi.shape[0]):
    ax2.plot(xs, phi[i], color=CLS_COL[1], lw=0.8, alpha=0.8)
ax2.spines[["top"]].set_visible(False)
ax2.tick_params(length=2, width=0.5, labelsize=5.6, pad=1.5, colors=CLS_COL[1])
ax2.set_ylabel("φ for the P logit (8 inputs)", fontsize=5.8, labelpad=1, color=CLS_COL[1])

axF = small_axes([0.735, row_y, 0.245, row_h], "F", "DEQ solver on the lead in G")
ra = np.array(meta["deq_anderson_rel_res"]); rp = np.array(meta["deq_plain_rel_res"])
axF.semilogy(np.arange(1, 1 + len(rp)), rp, "-", color="#8a96a3", lw=0.9, label="plain iteration")
axF.semilogy(np.arange(3, 3 + len(ra)), ra, "-o", color=DEQ_E, lw=1.1, ms=2.2, label="Anderson")
axF.axhline(1e-3, color=INK, lw=0.5, ls=(0, (2, 2)))
axF.text(30.5, 1.3e-3, "tolerance", fontsize=5.2, ha="right", va="bottom", color=INK)
axF.axvline(30, color="#c3ccd5", lw=0.6)
axF.text(29.6, 0.55, "cap", fontsize=5.2, ha="right", color=MUTED)
axF.set_xlim(0, 31); axF.set_ylim(3e-4, 1.6)
axF.set_xlabel("iteration", fontsize=5.8, labelpad=1)
axF.set_ylabel("relative residual", fontsize=5.8, labelpad=1)
axF.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=1.4, borderaxespad=0.2)
n_it = 2 + len(ra)
axF.annotate(f"{n_it} iterations", (n_it, ra[-1]), (n_it + 3.5, 0.02), fontsize=5.3, color=DEQ_E,
             arrowprops=dict(arrowstyle="-", color=DEQ_E, lw=0.6))

# ================================================================== (G) real outputs on one internal-test lead
axG = fig.add_axes([0.085, 0.045, 0.85, 0.115])
fig.text(0.02, 0.188, "G", fontsize=10, fontweight="bold", color=INK, va="bottom")
fig.text(0.053, 0.188, f"Five-fold ensemble outputs on one internal-test lead (LUDB record {rec}, lead II, "
         f"{meta['rhythm'].lower()}; first 4.1 s of the annotated span)", fontsize=6.4, fontweight="bold",
         color=INK, va="bottom")
fs = meta["fs"]; s0 = meta["span"][0]; n = 1024
x = z["x"][s0:s0 + n]; t = np.arange(n) / fs
lo, hi = float(x.min()), float(x.max()); rng = hi - lo
rows = [("reference", z["y"], None), ("U-Net-KAN", z["mask_unet_kan"], AG["unet_kan"]),
        ("ConvKAN", z["mask_convkan"], AG["convkan"]), ("DEQ", z["mask_deq"], AG["deq"])]
bar_h, gap = 0.13 * rng, 0.05 * rng
top = lo - 0.08 * rng
for r, (nm, arr, ag) in enumerate(rows):
    y1 = top - r * (bar_h + gap); y0 = y1 - bar_h
    a = arr[s0:s0 + n]
    k = 0
    while k < n:
        j = k
        while j < n and a[j] == a[k]:
            j += 1
        if a[k] in CLS_COL:
            axG.add_patch(Rectangle((t[k], y0), (j - k) / fs, bar_h, fc=CLS_COL[a[k]], ec="none",
                                    alpha=0.95 if r == 0 else 0.75, zorder=2))
            if r == 0:
                axG.axvspan(t[k], t[j - 1], ymin=0, ymax=1, color=CLS_COL[a[k]], alpha=0.07, lw=0, zorder=0)
        k = j
    axG.text(-0.03, (y0 + y1) / 2, nm, ha="right", va="center", fontsize=5.6, color=INK,
             fontweight="bold" if r == 0 else "normal")
    if ag is not None:
        axG.text(n / fs + 0.03, (y0 + y1) / 2, f"{100 * ag:.1f}%", ha="left", va="center", fontsize=5.4,
                 color=MUTED)
axG.text(n / fs + 0.03, top + 0.06 * rng, "agreement", ha="left", va="bottom", fontsize=5.0, color=MUTED)
axG.plot(t, x, color=INK, lw=0.7, zorder=3)
axG.set_xlim(0, n / fs); axG.set_ylim(top - 4 * (bar_h + gap) + gap, hi + 0.05 * rng)
axG.set_yticks([]); axG.spines[["left", "top", "right"]].set_visible(False)
axG.set_xticks(np.arange(0, 4.5, 0.5)); axG.tick_params(length=2, width=0.5, labelsize=5.6, pad=1.5)
axG.set_xlabel("time (s)", fontsize=5.8, labelpad=1)

# ------------------------------------------------------------------------------------------ shared legend
lg = fig.add_axes([0.0, 0.0, 1.0, 0.016]); lg.axis("off"); lg.set_xlim(0, 100); lg.set_ylim(0, 1)
x0 = 3.0
for fc, ec, nm in [(CONV, CONV_E, "convolution block"), (KAN, KAN_E, "KAN layer"), (DEQ, DEQ_E, "equilibrium core")]:
    lg.add_patch(FancyBboxPatch((x0, 0.15), 2.2, 0.7, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=ec,
                                lw=0.8))
    lg.text(x0 + 3.0, 0.5, nm, va="center", fontsize=5.9, color=INK); x0 += 14.5
lg.add_line(Line2D([x0, x0 + 2.6], [0.5, 0.5], color=SKIP, lw=0.8, ls=(0, (3, 2))))
lg.text(x0 + 3.4, 0.5, "skip connection", va="center", fontsize=5.9, color=INK); x0 += 13.5
for k in [1, 2, 3]:
    lg.add_patch(Rectangle((x0, 0.15), 2.2, 0.7, fc=CLS_COL[k], alpha=0.8, ec="none"))
    lg.text(x0 + 3.0, 0.5, CLS_NAME[k], va="center", fontsize=5.9, color=INK); x0 += 7.0 if k != 2 else 8.5
lg.text(x0 + 0.5, 0.5, "C×L: channels × samples, 4.1-s crop", va="center", fontsize=5.5, color=MUTED)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "Figure_1.pdf")
fig.savefig(OUT / "Figure_1.png", dpi=300)
print("saved", OUT, "| DEQ iterations on the lead:", n_it)
