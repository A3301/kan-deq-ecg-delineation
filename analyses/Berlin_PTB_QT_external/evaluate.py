"""Evaluate exactly the six frozen RDB fits. See protocol.txt for the locked design."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import scipy
import torch
from model import UNetKAN, merge, predict_proba, runs
from prepare_clips import ROOT, SOURCE, digest

RESULTS = ROOT / "results"

def read_models():
    models = {}
    for head in ("KAN", "GELU"):
        for seed in (1, 2, 3):
            path = ROOT / "models" / f"{head}_{seed}.pt"
            selection = json.loads(path.with_suffix(".json").read_text())
            assert selection["steps"] == 2400 and selection["weight_sha256"] == digest(path)
            assert selection["protocol"] == digest(ROOT/"models"/"training_protocol.txt")
            net = UNetKAN(where=["head"]) if head == "KAN" else UNetKAN(where=[],mlp_head=29)
            net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
            models[head,seed] = net.eval()
    return models

def retrieve_beat(qrs, t_waves, ref_q, clip_len=2500):
    if not qrs:
        return None, None, "QRS unavailable"
    i = min(range(len(qrs)),key=lambda k:(abs(4*qrs[k][0]-ref_q),qrs[k][0]))
    on, off = qrs[i]
    if abs(4*on-ref_q) > 150:
        return None, None, "QRS outside 150 ms"
    stop = qrs[i+1][0] if i+1<len(qrs) else clip_len
    t_end = next((end for start,end in t_waves if off<start<=end<stop),None)
    if t_end is None or t_end == clip_len-1:
        return 4*on, None, "T unavailable"
    return 4*on, 4*t_end, "returned"

def main():
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    nets = read_models()
    metadata = pd.read_csv(SOURCE/"cohort.csv").set_index("id")
    reference = pd.read_csv(SOURCE/"manual_reference.csv").set_index("id")
    sos = butter(2,[.5,40],fs=250,btype="bandpass",output="sos")
    rows, event_rows = [], []
    with np.load(SOURCE/"ptb_lead_ii_10s.npz",allow_pickle=False) as data:
        ids, waveforms = data["id"].tolist(), data["digital"]
    assert set(ids) == set(reference.index) == set(metadata.index) and len(ids)==len(set(ids))==549
    with torch.inference_mode():
        for index, (rid,raw) in enumerate(zip(ids,waveforms),start=1):
            meta = metadata.loc[rid]
            assert hashlib.sha256(raw.tobytes()).hexdigest() == meta.clip_int16_sha256
            signal = (raw.astype(float)-meta.baseline)/meta.gain
            filtered = sosfiltfilt(sos,resample_poly(signal,1,4))
            low,high = np.quantile(filtered,[.25,.75])
            iqr = (high-low)/1.349
            x = ((filtered-np.median(filtered))/(iqr if iqr>=1e-8 else filtered.std()+1e-8)).astype(np.float32)
            assert len(x)==2500 and np.isfinite(x).all()
            for (head,seed),net in nets.items():
                classes = predict_proba([net],x).argmax(axis=0)
                qrs = [(a,b) for a,b in merge(runs(classes,2)) if b-a+1>=10]
                tw = [(a,b) for a,b in merge(runs(classes,3)) if b-a+1>=15]
                event_rows.append(dict(id=rid,head=head,seed=seed,qrs=json.dumps(qrs),t=json.dumps(tw)))
                q,t = map(int,reference.loc[rid,["Q","T"]])
                if not 0<q<t<10000:
                    continue
                q_hat,t_hat,status = retrieve_beat(qrs,tw,q)
                rows.append(dict(id=rid,patient=meta.patient,head=head,seed=seed,reference_q_ms=q,reference_t_ms=t,
                                 predicted_q_ms=q_hat,predicted_t_ms=t_hat,status=status))
            if index%100==0 or index==len(ids):
                print(f"Evaluated {index}/{len(ids)} waveforms",flush=True)
    assert len(rows)==548*6 and len(event_rows)==549*6
    pd.DataFrame(rows).to_csv(RESULTS/"per_record.csv",index=False)
    pd.DataFrame(event_rows).to_csv(RESULTS/"predicted_runs_at_250Hz.csv",index=False)
    provenance = {p.relative_to(ROOT).as_posix():digest(p) for p in
                  [Path(__file__),ROOT/"model.py",ROOT/"protocol.txt",SOURCE/"manual_reference.csv",SOURCE/"cohort.csv",
                   SOURCE/"ptb_lead_ii_10s.npz",*sorted((ROOT/"models").glob("*.pt"))]}
    provenance["software"] = dict(numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,torch=str(torch.__version__))
    (RESULTS/"inference_manifest.json").write_text(json.dumps(provenance,indent=2))

if __name__ == "__main__":
    main()
