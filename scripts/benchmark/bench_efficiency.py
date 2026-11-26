"""Phase 5 efficiency: params, FLOPs (10 s record), CPU latency (1 and 6 threads, median of 50), and peak training
memory measured in a fresh subprocess per model (Windows peak working set; batch 32 x 1024, 5 steps).

Usage: python src/bench_efficiency.py [--models ...]      -> results/efficiency.json
Run only when no training is running (latency is sensitive to CPU contention).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ["unet", "cnn_bilstm", "transformer", "unet_kan", "convkan", "explicit_stack", "deq_unrolled", "deq",
          "deq_phantom"]

MEM_CHILD = r"""
import sys, json, os, psutil, torch, torch.nn.functional as F
sys.path.insert(0, r'%s')
import models as M
torch.manual_seed(0); torch.set_num_threads(6)
m = M.build('%s'); m.train()
opt = torch.optim.AdamW(m.parameters(), 1e-3)
x = torch.randn(32, 1, 1024); y = torch.randint(0, 4, (32, 1024))
p = psutil.Process(os.getpid())
import gc; gc.collect()
base = p.memory_info().rss
peak = base
for _ in range(5):
    out = m(x); loss = F.cross_entropy(out, y); opt.zero_grad(); loss.backward(); opt.step()
    peak = max(peak, p.memory_info().rss)
mi = p.memory_info()
print(json.dumps({"rss_base_mb": base/2**20, "rss_peak_sampled_mb": peak/2**20,
                  "peak_wset_mb": getattr(mi, 'peak_wset', peak)/2**20,
                  "train_mem_increase_mb": (getattr(mi, 'peak_wset', peak) - base)/2**20}))
"""


import functools  # noqa: E402


@functools.lru_cache(maxsize=None)
def real_input(L=2512):
    """10-s lead II of the first LUDB internal-test subject (DEQ iteration count depends on the input)."""
    import numpy as np
    import torch
    import data
    import splits
    sp = splits.load()
    s = [s for s in data.get("ludb") if s.subject == sp["test"][0] and s.lead == "ii"][0]
    return torch.from_numpy(np.pad(s.x, (0, L - len(s.x))))[None, None]


def flops(model, L=2512):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = real_input(L)
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops()


def latency(model, threads, L=2512, reps=50):
    import torch
    torch.set_num_threads(threads)
    x = real_input(L)
    model.eval()
    ts = []
    with torch.no_grad():
        for _ in range(5):
            model(x)
        for _ in range(reps):
            t = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t)
    ts.sort()
    return 1000 * ts[len(ts) // 2]


def main():
    import json as _j

    import torch

    import models as M
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--from_checkpoints", action="store_true", help="use fold-0 trained weights (DEQ iters!)")
    a = ap.parse_args()
    chosen = {}
    cfile = ROOT / "configs" / "chosen_lr.json"
    if cfile.exists():
        chosen = _j.loads(cfile.read_text())
    res = {}
    for nm in a.models.split(","):
        m = M.build(nm)
        if a.from_checkpoints and nm in chosen:
            ck = torch.load(ROOT / "experiments" / "checkpoints" / f"lr{chosen[nm]}_{nm}_f0_s1.pt", weights_only=False)
            m.load_state_dict(ck["state"])
        r = {"params": M.n_params(m)}
        r["gflops_10s"] = flops(m) / 1e9
        r["latency_ms_1thr"] = latency(m, 1)
        r["latency_ms_6thr"] = latency(m, 6)
        if hasattr(m, "last_info") and m.last_info:
            r["deq_iters_last"] = int(m.last_info.get("n_iter", 0))
        out = subprocess.run([sys.executable, "-c", MEM_CHILD % (str(ROOT / "src"), nm)], capture_output=True,
                             text=True, cwd=ROOT)
        try:
            r.update(_j.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:  # noqa: BLE001
            r["mem_error"] = out.stderr[-500:]
        res[nm] = r
        print(nm, _j.dumps(r), flush=True)
    tag = "_trained" if a.from_checkpoints else ""
    (ROOT / "results" / f"efficiency{tag}.json").write_text(_j.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
