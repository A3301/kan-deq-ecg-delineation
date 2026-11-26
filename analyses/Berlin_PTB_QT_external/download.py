"""Download PTBDB 1.0.0 native waveforms, verifying publisher SHA256 per file."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, re, threading, time
import requests

BASE=Path(__file__).resolve().parent
SRC=BASE/"source"
URL="https://physionet.org/files/ptbdb/1.0.0/"
DOC="https://static-content.springer.com/esm/art%3A10.1186%2F1475-925X-5-31/MediaObjects/12938_2006_174_MOESM1_ESM.doc"
sessions=threading.local()
CHECKSUMS={"reference.doc":"783d298e7090bab123b760d1cd6389ebdc32896ab96740bd81dcb7f6d2cd33a2",
           "RECORDS":"1bdcd205432b66e98389e095c8ea9f2f642f79265a1874929a9cb6de270852fe",
           "SHA256SUMS.txt":"994115d0484fe90daaff0e280ff6eeb1d8b76338f93d1687716efa3acf2cea83"}
def get(url):
    if not hasattr(sessions,"http"): sessions.http=requests.Session()
    for attempt in range(5):
        try:
            resp=sessions.http.get(url,timeout=(15,120)); resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if attempt==4: raise
            time.sleep(2**attempt)
def main():
    SRC.mkdir(parents=True,exist_ok=True)
    for name,url in {"reference.doc":DOC,"RECORDS":URL+"RECORDS","SHA256SUMS.txt":URL+"SHA256SUMS.txt"}.items():
        p=SRC/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=CHECKSUMS[name]:
            payload=get(url)
            assert hashlib.sha256(payload).hexdigest()==CHECKSUMS[name], name
            p.write_bytes(payload)
    refs={name.removeprefix("./"):sha for sha,name in (s.split(maxsplit=1) for s in (SRC/"SHA256SUMS.txt").read_text().splitlines() if s)}
    ids=(SRC/"RECORDS").read_text().splitlines()
    assert len(ids)==len(set(ids))==549 and all(re.fullmatch(r"patient\d{3}/[a-zA-Z0-9_]+",k) for k in ids)
    def fetch(rid):
        for ext in [".hea",".dat"]:
            name=rid+ext; p=SRC/"raw"/name;p.parent.mkdir(exist_ok=True,parents=True)
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==refs[name]:continue
            data=get(URL+name)
            if hashlib.sha256(data).hexdigest()!=refs[name]:raise ValueError(name)
            tmp=p.with_suffix(p.suffix+".part");tmp.write_bytes(data);tmp.replace(p)
        return rid
    with ThreadPoolExecutor(max_workers=5) as pool:
        for n,_ in enumerate(pool.map(fetch,ids),1):
            if n%25==0 or n==len(ids):print(f"Verified {n}/{len(ids)} header/signal pairs",flush=True)
if __name__=="__main__":main()
