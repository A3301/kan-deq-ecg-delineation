"""Reproduce the per-record Windows CPU timing using the original fold-zero models."""
import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import struct
import sys
import time
import winreg

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
pd.options.future.infer_string = False
import wfdb

HERE = Path(__file__).resolve().parent
ORDER_SEED = 20261125
LABELS = ["DEQ", "Unrolled8", "Unrolled18"]
N_REPEATS = 5


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def perf_core():
    """Choose the least occupied physical core in the highest Windows efficiency class."""
    lib = ctypes.WinDLL("Kernel32")
    n = wintypes.DWORD()
    lib.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
    buf = ctypes.create_string_buffer(n.value)
    if not lib.GetSystemCpuSetInformation(buf, n.value, ctypes.byref(n), None, 0):
        raise ctypes.WinError()
    data = buf.raw
    cores = {}
    i = 0
    while i < len(data):
        size, entry = struct.unpack_from("<II", data, i)
        if entry == 0:
            group = struct.unpack_from("<H", data, i+12)[0]
            logical, core = data[i+14:i+16]
            eff_class = data[i+18]
            if group == 0 and logical in psutil.Process().cpu_affinity():
                cores.setdefault((eff_class, core), []).append(logical)
        i += size
    cpu = psutil.cpu_percent(interval=2, percpu=True)
    max_class = max(c for c, _ in cores)
    chosen = min((ids for (c, k), ids in cores.items() if c == max_class),
                 key=lambda ids: (sum(cpu[i] for i in ids)/len(ids), min(ids)))
    # Use the same first SMT sibling on the chosen core across the whole run.
    logical = min(chosen)
    psutil.Process().cpu_affinity([logical])
    return dict(logical_cpu=logical, physical_core_siblings=chosen,
                windows_efficiency_class=max_class, pre_run_logical_busy_pc=cpu)


def read_signal(folder, record):
    raw, meta = wfdb.rdsamp(str(folder/record))
    lead = [x.lower() for x in meta["sig_name"]].index("ii")
    assert raw.shape == (5000, 12) and meta["fs"] == 500
    x = resample_poly(raw[:, lead], 1, 2)
    x = np.nan_to_num(x.astype("float64"))
    x = sosfiltfilt(butter(2, (.5, 40), btype="bandpass", fs=250, output="sos"), x)
    q75, q25 = np.percentile(x, [75,25])
    scale = (q75-q25)/1.349
    if scale < 1e-8:
        scale = np.std(x)+1e-8
    x = ((x-np.median(x))/scale).astype("float32")
    return torch.from_numpy(np.pad(x, (0, 12)))[None,None]


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", type=Path, default=HERE/"resources")
    ap.add_argument("--ludb", type=Path, default=HERE.parent/"kan_gelu"/"LUDB")
    ap.add_argument("--splits", type=Path, default=HERE.parent/"kan_gelu"/"splits.json")
    ap.add_argument("--output", type=Path, default=HERE)
    a=ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(a.resources.resolve()))
    import models as M
    assert Path(M.__file__).resolve() == (a.resources/"models.py").resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cpu_key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cpu_key) as k:
        product = winreg.QueryValueEx(k,"ProcessorNameString")[0].strip()
    chosen = json.loads((a.resources/"chosen_lr.json").read_text())
    weights = [f"lr{chosen['deq']}_deq_f0_s1.pt", f"lr{chosen['deq_unrolled']}_deq_unrolled_f0_s1.pt",
               "s5u18_deq_unrolled_f0_s1.pt"]
    models = {}
    hashes = {}
    for label, file, unroll in zip(LABELS, weights, (None,8,18)):
        m = M.build("deq") if unroll is None else M.build("deq_unrolled", unroll=unroll)
        ck = torch.load(a.resources/file, map_location="cpu", weights_only=True)
        m.load_state_dict(ck["state"], strict=True)
        models[label] = m.eval()
        hashes[file] = digest(a.resources/file)
    counts = {label:M.n_params(m) for label,m in models.items()}
    assert len(set(counts.values())) == 1
    ids = json.loads(a.splits.read_text())["test"]
    assert len(ids)==len(set(ids))==41
    xs = {r:read_signal(a.ludb,r) for r in ids}
    assert all(x.shape==(1,1,2512) and torch.isfinite(x).all() for x in xs.values())
    hashes.update({"models.py":digest(a.resources/"models.py"),"splits.json":digest(a.splits),
                   "chosen_lr.json":digest(a.resources/"chosen_lr.json")})
    affinity = perf_core()
    rng = random.Random(ORDER_SEED)
    log=[];load=[]
    start = datetime.now().astimezone()
    gc.disable()
    with torch.no_grad():
        for i in range(6):
            labels=LABELS.copy()
            rng.shuffle(labels)
            for name in labels:
                y = models[name](xs[ids[i]])
                assert tuple(y.shape)==(1,4,2512) and torch.isfinite(y).all()
        p = psutil.Process()
        _=p.cpu_percent(None);_=psutil.cpu_percent(None)
        for k in range(N_REPEATS):
            ordered_ids = ids.copy()
            rng.shuffle(ordered_ids)
            for rec in ordered_ids:
                labels = LABELS.copy();rng.shuffle(labels)
                for name in labels:
                    m = models[name]; x = xs[rec]
                    t0 = time.perf_counter_ns()
                    y = m(x)
                    elapsed = (time.perf_counter_ns()-t0)/1e6
                    log.append(dict(round=k+1, order=len(log)+1, record=rec, model=name, latency_ms=elapsed,
                                    iterations=int(m.last_info["n_iter"]), final_residual=float(m.last_info["rel_res"].item())))
            load.append(dict(round=k+1,host_busy_pc=psutil.cpu_percent(None),process_busy_pc_one_core=p.cpu_percent(None)))
    gc.enable()
    stop = datetime.now().astimezone()
    df = pd.DataFrame(log)
    assert (df.groupby(["record","model"]).size()==N_REPEATS).all()
    assert (df.groupby(["record","model"]).iterations.nunique()==1).all()
    df.to_csv(a.output/"latency_raw.csv", index=False)
    pd.DataFrame(load).to_csv(a.output/"cpu_load.csv", index=False)
    rec = df.groupby(["record","model"])["latency_ms"].median().unstack()
    rec.to_csv(a.output/"record_medians.csv")
    stats={}
    for name in LABELS:
        times=rec[name]
        stats[name] = dict(median_ms=float(times.median()), p95_ms=float(times.quantile(.95)),
                           min_ms=float(times.min()),max_ms=float(times.max()))
    it=df[df.model=="DEQ"].groupby("record").iterations.first()
    stats["DEQ_iterations"] = dict(median=float(it.median()),p95=float(it.quantile(.95)),max=int(it.max()),min=int(it.min()))
    meta = dict(cpu=product, physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(True),
                ram_gib=psutil.virtual_memory().total/2**30,os=platform.platform(),
                python=sys.version,torch=torch.__version__,numpy=np.__version__,wfdb=wfdb.__version__,scipy=scipy.__version__,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                mkldnn=torch.backends.mkldnn.enabled,affinity=affinity,parameters=counts,rng_seed=ORDER_SEED,
                start=start.isoformat(),end=stop.isoformat(),warmup_per_model=6,rounds=N_REPEATS,n_calls=len(df),
                load=load,hashes=hashes,metrics=stats,
                metric_definition="median of 5 repeat forward calls per record, then median and linear 95th percentile across 41 distinct records",
                model_scope="default fold-0 seed-1 eval() forward under no_grad, CPU float32; preloaded; batch=1; DEQ original solver; original explicit twin additional diagnostic residual block evaluation included")
    (a.output/"profile_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"stats":stats,"load":load,"affinity":affinity["logical_cpu"],"checkpoints":weights,"params":counts},indent=2), flush=True)

if __name__ == "__main__":
    run()
