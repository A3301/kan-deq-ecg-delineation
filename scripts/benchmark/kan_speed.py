"""speed/memory of KAN layer implementations (analysis plan): own B-spline (efficient-KAN formulation),
own RBF (FastKAN formulation), and pykan's KANLayer if installed. Layer 64->64 applied to 32x1024 tokens
(the U-Net-KAN bottleneck shape is 32x64 tokens; 1024 tokens stress-tests it). 6 threads, median of 20.
Output: results/kan_speed.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import KANLinear  # noqa: E402
from paths import RESULTS  # noqa: E402

torch.set_num_threads(6)
torch.manual_seed(0)
x = torch.randn(32 * 1024, 64)


def bench(layer, reps=20):
    fw, fb = [], []
    for i in range(reps + 3):
        t = time.perf_counter()
        y = layer(x)
        t1 = time.perf_counter()
        y.sum().backward()
        t2 = time.perf_counter()
        if i >= 3:
            fw.append(t1 - t)
            fb.append(t2 - t)
    fw.sort()
    fb.sort()
    return {"params": sum(p.numel() for p in layer.parameters()), "fwd_ms": 1000 * fw[len(fw) // 2],
            "fwd_bwd_ms": 1000 * fb[len(fb) // 2]}


res = {"own_bspline_G8_k3": bench(KANLinear(64, 64, grid=8, order=3, basis="bspline")),
       "own_rbf_G8": bench(KANLinear(64, 64, grid=8, basis="rbf")),
       "mlp_linear_baseline": bench(torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.SiLU()))}
try:
    from kan.KANLayer import KANLayer  # pykan
    layer = KANLayer(in_dim=64, out_dim=64, num=8, k=3)

    class W(torch.nn.Module):
        def __init__(self, l):
            super().__init__()
            self.l = l

        def forward(self, z):
            return self.l(z)[0]
    res["pykan_KANLayer_G8_k3"] = bench(W(layer), reps=5)
    import kan
    res["pykan_version"] = getattr(kan, "__version__", "unknown")
except Exception as e:  # noqa: BLE001
    res["pykan"] = f"not benchmarked: {type(e).__name__}: {e}"
(RESULTS / "kan_speed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
