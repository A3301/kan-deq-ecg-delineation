from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,threading,time
import pandas as pd,requests

ROOT=Path(__file__).resolve().parent
PTB="https://physionet.org/files/ptb-xl/1.0.3/"
PLUS="https://physionet.org/files/ptb-xl-plus/1.0.1/"
W=ROOT/"source";RAW=W/"ptbxl";MARK=W/"ecgdeli"
for path in (RAW,MARK):path.mkdir(exist_ok=True)
thread_local=threading.local()
def read(url):
    if not hasattr(thread_local,"s"):thread_local.s=requests.Session()
    for attempt in range(4):
        try:
            r=thread_local.s.get(url,timeout=(12,60));r.raise_for_status();return r.content
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(1.25**attempt)
def get_index(base,stem):
    path=W/(stem+"_SHA256SUMS.txt")
    if not path.exists():path.write_bytes(read(base+"SHA256SUMS.txt"))
    return {line.split(None,1)[1].strip().lstrip("*").removeprefix("./"):line.split(None,1)[0] for line in path.read_text().splitlines()}
def acquire(base,index,key,dest):
    if key not in index:return (key,"not in publisher digest list")
    if dest.exists():b=dest.read_bytes()
    else:b=read(base+key)
    h=hashlib.sha256(b).hexdigest()
    if not h==index[key]:raise ValueError(key)
    if not dest.exists():dest.write_bytes(b)
    return (key,h)
def main():
    roster=pd.read_csv(ROOT/"roster.csv")
    with ThreadPoolExecutor(max_workers=2) as pool:
        indices=list(pool.map(lambda p:get_index(*p),[(PTB,"ptbxl"),(PLUS,"ecgdeli")]))
    pi,di=indices
    print('manifests',len(pi),len(di),flush=True)
    pm=W/"ptbxl_database_official.csv";acquire(PTB,pi,"ptbxl_database.csv",pm)
    db=pd.read_csv(pm).set_index("ecg_id")
    for row in roster.itertuples():
        assert int(db.loc[row.ecg_id,"patient_id"])==row.patient_id and db.loc[row.ecg_id,"filename_hr"]==row.filename_hr
    jobs=[]
    for row in roster.itertuples():
        # Verification records are downloaded but stay unused unless a map is locked.
        g=f"{int(row.ecg_id//1000)*1000:05d}"
        mark=f"{row.ecg_id:05d}_points_global.atr"
        jobs.append((PLUS,di,f"fiducial_points/ecgdeli/{g}/{mark}",MARK/mark))
        if row.split=="calibration":
            for ext in ("hea","dat"):
                name=row.filename_hr+"."+ext
                jobs.append((PTB,pi,name,RAW/Path(name).name))
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows=list(pool.map(lambda j:acquire(*j),jobs))
    (W/"official_retrieval.json").write_text(json.dumps(rows,indent=1),encoding="utf8")
    print("downloaded or verified",len(rows),"entries, missing",sum(x[1].startswith("not in") for x in rows),flush=True)
if __name__=="__main__":main()
