# -*- coding: utf-8 -*-
"""Data for Figure 1 (architecture schematic), taken from the model definitions and the stored fold weights.

Writes figure1_data.npz + figure1_data.json:
  - one LUDB internal-test lead II (first sinus-rhythm test participant), filtered and standardised as in training;
  - reference 4-class mask and the five-fold ensemble masks of U-Net-KAN, ConvKAN and DEQ (argmax of mean softmax);
  - DEQ (fold 0) Anderson relative residual at every iteration on that lead;
  - tensor shapes of every stage for a 1024-sample crop, and parameter counts.
usage: python figure1_data.py <project root (contains src/, configs/, experiments/)> <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
import types
sys.modules.setdefault("wfdb", types.ModuleType("wfdb"))  # only the cached LUDB pickle is read; wfdb is not needed
import data, splits, models as M  # noqa: E402

torch.set_num_threads(4)
chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())


def load(arch, fold):
    ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{chosen[arch]}_{arch}_f{fold}_s1.pt", weights_only=False)
    m = M.build(ck["model"], **ck["kw"]); m.load_state_dict(ck["state"]); m.eval()
    return m


sp = splits.load()
test = set(sp["test"])
leads = [s for s in data.get("ludb") if s.subject in test and s.lead == "ii"]
sinus = sorted([s for s in leads if s.meta["rhythm"].lower().startswith("sinus rhythm")], key=lambda s: int(s.subject))
s = sinus[0]
L = len(s.x); Lp = int(np.ceil(L / 16) * 16)
x = torch.from_numpy(np.pad(s.x, (0, Lp - L)).astype(np.float32))[None, None]

masks, params = {}, {}
for arch in ["unet_kan", "convkan", "deq"]:
    prob = 0
    with torch.no_grad():
        for f in range(5):
            m = load(arch, f)
            prob = prob + torch.softmax(m(x), 1)[0, :, :L].numpy() / 5
    masks[arch] = prob.argmax(0).astype(np.int8)
    params[arch] = M.n_params(m)

# DEQ solver trajectory on this lead (fold 0): residual after each Anderson iteration (deterministic re-runs)
deq = load("deq", 0)
with torch.no_grad():
    u = deq.inj(deq.down(deq.stem(x)))
    z0 = torch.zeros_like(u)
    res = []
    for k in range(3, 31):
        _, info = M.anderson(lambda zz: deq.f(zz, u), z0, max_iter=k, tol=1e-3)
        res.append(float(info["rel_res"][0]))
        if res[-1] < 1e-3:
            break
    # plain fixed-point iteration with the same tied block, for comparison
    z, plain = z0, []
    for k in range(30):
        zn = deq.f(z, u)
        plain.append(float(((zn - z).norm() / (1e-5 + zn.norm()))))
        z = zn

# stage shapes for a 1024-sample crop
def shp(o):
    return tuple((o[0] if isinstance(o, tuple) else o).shape[1:])


shapes = {}
xc = torch.zeros(1, 1, 1024)
for arch in ["unet_kan", "convkan", "deq"]:
    m = load(arch, 0); rec = []
    hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: rec.append((n, shp(o))))
             for n, mod in m.named_children()]
    if arch == "unet_kan":
        hooks += [b.register_forward_hook(lambda mod, i, o, n=f"enc{j}": rec.append((n, shp(o))))
                  for j, b in enumerate(m.enc.blocks)]
        hooks += [b.register_forward_hook(lambda mod, i, o, n=f"dec{j}": rec.append((n, shp(o))))
                  for j, b in enumerate(m.dec.blocks)]
    if arch == "convkan":
        hooks += [b.register_forward_hook(lambda mod, i, o, n=f"down{j}": rec.append((n, shp(o))))
                  for j, b in enumerate(m.down)]
        hooks += [b.register_forward_hook(lambda mod, i, o, n=f"up{j}": rec.append((n, shp(o))))
                  for j, b in enumerate(m.up)]
    with torch.no_grad():
        m(xc)
    for h in hooks:
        h.remove()
    shapes[arch] = rec


# ConvKAN head (fold 0): learned univariate functions phi_{P,i} of its 1x1 KAN convolution (8 inputs -> 4 logits)
ck = load("convkan", 0).head.kan
xs = torch.linspace(-2, 2, 201)
phi_p = torch.stack([ck.univariate(i, 1, xs) for i in range(ck.din)]).numpy()
rng = phi_p.max(1) - phi_p.min(1)

np.savez_compressed(OUT / "figure1_data.npz", x=s.x.astype(np.float32), y=s.y, **{f"mask_{k}": v for k, v in masks.items()},
         convkan_xs=xs.numpy(), convkan_phi_P=phi_p)
meta = {"record": s.record, "subject": s.subject, "lead": s.lead, "rhythm": s.meta["rhythm"], "fs": data.FS,
        "span": [int(v) for v in s.span], "chosen_lr": {k: chosen[k] for k in masks}, "params": params,
        "deq_anderson_rel_res": res, "deq_plain_rel_res": plain, "shapes_1024": shapes, "convkan_phi_P_range": rng.round(4).tolist(),
        "convkan_rbf": {"grid": int(ck.grid), "centers": ck.centers.tolist(), "inv_w": float(ck.inv_w)},
        "agreement_with_reference": {k: float((v[s.y >= 0] == s.y[s.y >= 0]).mean()) for k, v in masks.items()}}
json.dump(meta, open(OUT / "figure1_data.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ["record", "rhythm", "span", "params", "agreement_with_reference"]}))
print("anderson", [round(r, 5) for r in res]); print("plain", [round(r, 4) for r in plain[:12]])
for a, r in shapes.items():
    print(a, r)
