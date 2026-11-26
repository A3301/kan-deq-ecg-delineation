from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, re, hashlib, csv, json
import numpy as np
import wfdb

HERE=Path(__file__).resolve().parent
raw=HERE/"raw"
raw.mkdir(exist_ok=True)
base="https://physionet.org/files/qtdb/1.0.0/"
r=requests.get(base,timeout=30);r.raise_for_status()
ids=sorted(set(re.findall(r'(sel\w+)\.q2c',r.text)))
assert len(ids)==11
def get(path):
    dest=raw/path
    if not dest.exists():
        r=requests.get(base+path,timeout=30);r.raise_for_status();dest.write_bytes(r.content)
    return (path,hashlib.sha256(dest.read_bytes()).hexdigest())
with ThreadPoolExecutor(max_workers=5) as pool:
    sums=dict(pool.map(get,[id+"."+ext for id in ids for ext in ["q1c","q2c","hea"]]))
(HERE/"sources.json").write_text(json.dumps({"url":base,"protocol_sha256":hashlib.sha256((HERE/"protocol.txt").read_bytes()).hexdigest(),"files":sums},indent=2))
QRS=set("NLRBAaJSVrFejnE/fQ?")
def extract(id,reader):
    a=wfdb.rdann(str(raw/id),reader);v=[]
    for i,s in enumerate(a.symbol):
        if s in QRS and i>0 and i+1<len(a.symbol) and a.symbol[i-1]=="(" and a.symbol[i+1]==")" and a.chan[i-1]==a.chan[i]==a.chan[i+1] and a.sample[i-1]<a.sample[i]<a.sample[i+1]:
            v.append(tuple(int(x) for x in (a.sample[i-1],a.sample[i],a.sample[i+1],a.chan[i])))
    return v
def writer(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
records=[];beats=[]
for id in ids:
    fs=wfdb.rdheader(str(raw/id)).fs;u=1000/fs
    a,b=extract(id,"q1c"),extract(id,"q2c")
    pairs=sorted((abs(x[1]-y[1]),i,j) for i,x in enumerate(a) for j,y in enumerate(b) if abs(x[1]-y[1])*u<=150)
    seen1=set();seen2=set()
    for _,i,j in pairs:
        if i in seen1 or j in seen2:continue
        seen1.add(i);seen2.add(j)
        x,y=a[i],b[j]
        beats.append(dict(record=id,onset_diff_ms=(y[0]-x[0])*u,offset_diff_ms=(y[2]-x[2])*u,duration_diff_ms=((y[2]-y[0])-(x[2]-x[0]))*u,reader1_channel=x[3],reader2_channel=y[3]))
    records.append(dict(record=id,r1_complete=len(a),r2_complete=len(b),paired=len(seen1)))
writer(HERE/"records.csv",records);writer(HERE/"paired.csv",beats)
out={"n_records":len(records),"records_with_pairs":sum(v["paired"]>0 for v in records),"r1_complete":sum(v["r1_complete"] for v in records),"r2_complete":sum(v["r2_complete"] for v in records),"paired":len(beats),"different_channel":sum(v["reader1_channel"]!=v["reader2_channel"] for v in beats)}
rng=np.random.default_rng(20261125);idx=rng.integers(0,len(ids),(10000,len(ids)))
for metric in ["onset","offset","duration"]:
    col=metric+"_diff_ms";vals=np.array([x[col] for x in beats]);byrec=[np.array([x[col] for x in beats if x["record"]==id]) for id in ids]
    counts=np.array(list(map(len,byrec)))
    boot=np.array([sum(np.abs(a)) for a in byrec])[idx].sum(1)/counts[idx].sum(1)
    out[metric]={"mean_abs":np.abs(vals).mean(),"median_abs":np.median(np.abs(vals)),"signed_mean_reader2_minus_1":vals.mean(),"ci95_record_bootstrap":np.quantile(boot,[.025,.975]).tolist()}
(HERE/"summary.json").write_text(json.dumps(out,indent=2));print(out)
