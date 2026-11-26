"""training-memory increase of the DEQ and its unrolled twins (8 and 18
steps), measured with the same child-process procedure as bench_efficiency.py (Windows peak working set; batch
32 x 1024, 5 steps, 6 threads), three repetitions per model in fresh processes.
Output: results/unroll_memory.json
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bench_efficiency import MEM_CHILD  # noqa: E402

VARIANTS = [("deq", "deq", "{}"), ("deq_unrolled (8 steps)", "deq_unrolled", "{}"),
            ("deq_unrolled (18 steps)", "deq_unrolled", '{"unroll": 18}')]


def child(model, kw):
    code = MEM_CHILD % (str(ROOT / "src"), model)
    code = code.replace(f"m = M.build('{model}')", f"m = M.build('{model}', **{kw})")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])["train_mem_increase_mb"]


def main():
    res = {}
    for lab, model, kw in VARIANTS:
        vals = [child(model, kw) for _ in range(3)]
        res[lab] = {"runs_mb": vals, "median_mb": statistics.median(vals)}
        print(lab, [round(v, 1) for v in vals], flush=True)
    (ROOT / "results" / "unroll_memory.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
