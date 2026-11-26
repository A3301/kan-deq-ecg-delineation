"""DEQ latency distribution over real inputs (additional analysis; run on an idle CPU).

The primary benchmark reports DEQ latency on one LUDB lead (15 Anderson iterations). Because DEQ cost depends on the
input, this script measures single-thread latency and iteration count for every LUDB internal-test lead (41 x 12,
10 s at 250 Hz) with the fold-0 DEQ, and for comparison the fold-0 DEQ-unrolled, U-Net, CNN-BiLSTM and U-Net-KAN on
the same leads. Output: results/deq_latency.json, results/tables/deq_latency.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import splits  # noqa: E402
from final_eval import load_models  # noqa: E402
from paths import RESULTS  # noqa: E402

torch.set_num_threads(1)


def main():
    test = set(splits.load()["test"])
    leads = [s for s in data.get("ludb") if s.subject in test]
    xs = [torch.from_numpy(np.pad(s.x.astype(np.float32), (0, 2512 - len(s.x))))[None, None] for s in leads]
    out = {"n_leads": len(xs)}
    for arch in ["deq", "deq_unrolled", "unet", "cnn_bilstm", "unet_kan"]:
        m = load_models(arch)[0][0]
        m.eval()
        lat, its = [], []
        with torch.no_grad():
            for x in xs[:3]:
                m(x)                                   # warm-up
            for x in xs:
                t = time.perf_counter()
                m(x)
                lat.append(1000 * (time.perf_counter() - t))
                if hasattr(m, "last_info"):
                    its.append(int(m.last_info.get("n_iter", 0)))
        lat = np.array(lat)
        r = {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "max_ms": float(lat.max()),
             "mean_ms": float(lat.mean())}
        if its:
            its = np.array(its)
            r.update({"iters_median": float(np.median(its)), "iters_p95": float(np.percentile(its, 95)),
                      "iters_max": int(its.max()), "iters_min": int(its.min()), "frac_at_cap": float((its >= 30).mean())})
        out[arch] = r
        print(arch, {k: round(v, 2) for k, v in r.items()}, flush=True)
    (RESULTS / "deq_latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    rows = ["### Single-thread CPU latency over all LUDB internal-test leads (10 s, 250 Hz; fold-0 model)", "",
            f"n = {out['n_leads']} leads", "", "| Model | median (ms) | 95th pct (ms) | max (ms) | DEQ iterations median / p95 / max |",
            "|---|---|---|---|---|"]
    for a in ["unet", "cnn_bilstm", "unet_kan", "deq_unrolled", "deq"]:
        r = out[a]
        it = f"{r['iters_median']:.0f} / {r['iters_p95']:.0f} / {r['iters_max']}" if "iters_median" in r else "–"
        rows.append(f"| {a} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['max_ms']:.1f} | {it} |")
    (RESULTS / "tables" / "deq_latency.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
