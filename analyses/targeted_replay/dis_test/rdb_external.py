"""Reproduce frozen lead-II evaluation. Run --help for paths or stages."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
from frozen_model import HOP, WIN, UNetKAN, merge, predict_proba, runs


HERE = Path(__file__).resolve().parent
POINTS = [f"{w}_{p}" for w in ("P", "QRS", "T") for p in ("on", "off")]
HEADS = {"KAN": "headonly", "GELU": "mlphead"}
SAMPLES = 5000


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_out(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def references(raw):
    df = pd.read_csv(io.BytesIO(raw), header=0, names=["type", "start", "end"])
    vals = {key: [] for key in POINTS}
    qc = dict(annotation_rows=len(df), invalid_interval=0, zero_length=0,
              outside=0, censored_boundary=0, duplicate_mark=0)
    for wave, onset, offset in df.itertuples(index=False, name=None):
        if wave not in (0, 1, 2) or not np.isfinite([onset, offset]).all() or offset < onset:
            qc["invalid_interval"] += 1
            continue
        if onset == offset:
            qc["zero_length"] += 1
            continue
        if onset < 0 or offset > SAMPLES - 1:
            qc["outside"] += 1
            continue
        for side, pos, censored in ((0, onset, onset == 0), (1, offset, offset == SAMPLES-1)):
            if censored:
                qc["censored_boundary"] += 1
            else:
                vals[POINTS[2 * int(wave) + side]].append(float(pos) * 2)
    for key, v in vals.items():
        qc["duplicate_mark"] += len(v) - len(set(v))
        vals[key] = sorted(set(v))
    return vals, qc


def prepare(args):
    mapping = pd.read_excel(args.mapping).sort_values("rdb_record_id")
    sources, waves, ids, ecgs, roster = set(), set(), [], [], []
    refs = {}
    sos = butter(2, [0.5, 40.0], fs=250, btype="bandpass", output="sos")
    with ZipFile(args.rdb) as archive:
        members = {n.split("/")[-1].split(".")[0] for n in archive.namelist() if n.startswith("dat_csv/") and n.endswith(".csv")}
        assert members == set(mapping.rdb_record_id) and len(members) == 2399
        for row in mapping.itertuples(index=False):
            rid, source = row.rdb_record_id, row.chapman_record_id
            stat = dict(record=rid, source=source, rhythm=row.chapman_rhythm, released_rhythm=row.rdb_rhythm, status="included")
            roster.append(stat)
            if source in sources:
                stat["status"] = "duplicate_source"
                continue
            sources.add(source)
            signals = pd.read_csv(io.BytesIO(archive.read(f"dat_csv/{rid}.csv")), header=None, usecols=[1]).iloc[:, 0].to_numpy(dtype=np.float64)
            digest = hashlib.sha256(signals.astype("<f8").tobytes()).hexdigest()
            stat["lead_II_sha256"] = digest
            if digest in waves:
                stat["status"] = "duplicate_lead_II_signal"
                continue
            waves.add(digest)
            if len(signals) != SAMPLES or not np.isfinite(signals).all() or np.ptp(signals) == 0:
                stat["status"] = "invalid_waveform"
                continue
            raw = archive.read(f"ann_txt/{rid}.ii.txt")
            labels, qc = references(raw)
            stat.update(qc)
            flat = [v for group in labels.values() for v in group]
            if not flat:
                stat["status"] = "no_reference"
                continue
            # Byte comparisons characterize apparent copies in the supplied release.
            stat["distinct_lead_annotation_files"] = len({archive.read(f"ann_txt/{rid}.{lead}.txt")
                for lead in ("i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6")})
            stat["score_start_ms"] = max(0.0, min(flat) - 150.0)
            stat["score_end_ms"] = min(2.0*(SAMPLES-1), max(flat) + 150.0)
            for k, v in labels.items():
                stat[k] = len(v)
            filt = sosfiltfilt(sos, resample_poly(signals, 1, 2))
            q25, q75 = np.percentile(filt, [25, 75])
            scale = (q75 - q25) / 1.349
            if scale < 1e-8:
                scale = filt.std() + 1e-8
            ecgs.append(((filt - np.median(filt)) / scale).astype("float32"))
            ids.append(rid)
            refs[rid] = labels
    pd.DataFrame(roster).to_csv(args.out / "cohort.csv", index=False)
    json_out(args.out / "references_ms.json", refs)
    json_out(args.out / "signal_order.json", ids)
    np.save(args.out / "signals.npy", np.stack(ecgs), allow_pickle=False)
    json_out(args.out / "input_sha256.json", {p.name: sha(p) for p in [args.rdb, args.mapping, HERE/"protocol.txt", HERE/"frozen_model.py"]})
    print("Prepared", len(ids), "source ECGs", flush=True)


def models(arm, seed, folder):
    result = []
    hashes = {}
    for fold in range(5):
        path = folder / f"abl-{HEADS[arm]}_unet_kan_f{fold}_s{seed}.pt"
        ck = torch.load(path, weights_only=True, map_location="cpu")
        assert ck["model"] == "unet_kan" and ck["args"]["fold"] == fold and ck["args"]["seed"] == seed
        kw = {"where": ["head"]} if arm == "KAN" else {"where": [], "mlp_head": 29}
        assert ck["kw"] == kw
        assert not ck["args"].get("xdb_qtdb", 0) and not ck["args"].get("multi", 0)
        m = UNetKAN(**kw)
        m.load_state_dict(ck["state"], strict=True)
        result.append(m.eval())
        hashes[path.name] = sha(path)
    return result, hashes


def batched_probabilities(nets, signals):
    # Each RDB signal is exactly 2500 samples after resampling. Start positions
    # [0,452] match the original shared sliding-window function.
    n = signals.shape[-1]
    assert WIN < n < WIN + HOP
    starts = [0, n - WIN]
    x = np.stack([signals[:, start:start+WIN] for start in starts], axis=1)
    batch = torch.from_numpy(x.reshape(-1, 1, WIN))
    probs = (sum(m(batch).softmax(dim=1) for m in nets)/len(nets)).numpy().reshape(len(signals),2,4,WIN)
    taper = np.hanning(WIN).astype("float32") + 0.001
    total = np.zeros((len(signals), 4, n), dtype="float32")
    weight = np.zeros(n, dtype="float32")
    for j, start in enumerate(starts):
        total[:,:,start:start+WIN] += probs[:,j] * taper
        weight[start:start+WIN] += taper
    return total/weight


def infer(args, arms=HEADS, checkpoint_loader=models):
    signals = np.load(args.out / "signals.npy", mmap_mode="r")
    for arm in arms:
        for seed in (1, 2):
            path = args.out / f"masks_{arm}_{seed}.npy"
            if path.is_file() and (args.out/f"checks_{arm}_{seed}.json").is_file():
                continue
            nets, hashes = checkpoint_loader(arm, seed, args.weights)
            pred = np.empty(signals.shape, dtype="uint8")
            clock = time.monotonic()
            with torch.inference_mode():
                for i in range(0, len(signals), args.batch):
                    sig = signals[i:i+args.batch]
                    proba = batched_probabilities(nets, sig)
                    if i == 0:
                        ref = predict_proba(nets, sig[0])
                        error = float(np.max(np.abs(proba[0] - ref)))
                        discord = int(np.sum(proba[0].argmax(0) != ref.argmax(0)))
                        assert error < 0.0003 and discord == 0, (error, discord)
                    pred[i:i+len(sig)] = proba.argmax(axis=1)
                    if i and i % (args.batch*50) == 0:
                        print(arm, seed, i, round(time.monotonic()-clock), flush=True)
            np.save(path, pred, allow_pickle=False)
            json_out(args.out/f"checks_{arm}_{seed}.json",dict(weights=hashes,mask_sha256=sha(path),
                single_vs_batch_max_probability_error=error,single_vs_batch_label_difference=discord,
                parameters=sum(p.numel() for p in nets[0].parameters()),record_count=len(pred),torch=str(torch.__version__)))
            print("Finished", arm, seed, round(time.monotonic()-clock), flush=True)


def detections(mask):
    pts = []
    for cls, minlen in ((1,10), (2,10), (3,15)):
        intervals = [(a,b) for a,b in merge(runs(mask, cls)) if b-a+1 >= minlen]
        pts.extend([np.array([4.*a for a,b in intervals if a>0]),
                    np.array([4.*b for a,b in intervals if b<len(mask)-1])])
    return pts


def pair_errors(ref, det, tol):
    dist = abs(ref[:,None] - det[None,:])
    ii, jj = np.nonzero(dist <= tol)
    rused, dused, errors = set(), set(), []
    for k in np.argsort(dist[ii,jj], kind="stable"):
        i, j = ii[k], jj[k]
        if i not in rused and j not in dused:
            rused.add(i)
            dused.add(j)
            errors.append(det[j]-ref[i])
    return np.array(errors)


def score(args, arms=HEADS, filename="record_scores.csv"):
    ids = json.loads((args.out/"signal_order.json").read_text())
    refs = json.loads((args.out/"references_ms.json").read_text())
    meta = pd.read_csv(args.out/"cohort.csv").set_index("record")
    with (args.out/filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record","rhythm","arm","seed","tolerance_ms","endpoint","tp","fp","fn","abs_error_sum_ms","error_sum_ms"])
        for arm in arms:
            for seed in (1,2):
                masks = np.load(args.out/f"masks_{arm}_{seed}.npy", mmap_mode="r")
                for rid, mask in zip(ids, masks, strict=True):
                    row = meta.loc[rid]
                    for key, pred in zip(POINTS, detections(mask), strict=True):
                        det = pred[(pred >= row.score_start_ms) & (pred <= row.score_end_ms)]
                        ref = np.array(refs[rid][key])
                        for tol in (150, 40):
                            err = pair_errors(ref, det, tol)
                            tp = len(err)
                            writer.writerow([rid,row.rhythm,arm,seed,tol,key,tp,len(det)-tp,len(ref)-tp,np.abs(err).sum(),err.sum()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdb", type=Path, required=True, help="dataset.zip from RDB v4")
    parser.add_argument("--mapping", type=Path, required=True, help="published RDB-to-Chapman mapping workbook")
    parser.add_argument("--weights", type=Path, required=True, help="20 original checkpoints")
    parser.add_argument("--out", type=Path, default=HERE/"results")
    parser.add_argument("--stage", choices=("all","prepare","infer","score"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage in ("all","prepare"): prepare(args)
    if args.stage in ("all","infer"): infer(args)
    if args.stage in ("all","score"): score(args)
