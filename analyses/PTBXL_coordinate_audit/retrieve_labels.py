from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,hashlib,sys
import requests

ROOT=Path(__file__).resolve().parent
COMMIT="c9d0577dc2bb82a6738e6d5dce1451aae4b09904"
REPO="figureli/ptb-xl-ecg-delineation"
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}"
where=ROOT/"source";where.mkdir(exist_ok=True)
def run(meta):
    name=meta["path"];dest=where/name
    if not dest.exists():
        path=dest.with_suffix(dest.suffix+".partial")
        with requests.get(BASE+"/"+name,stream=True,timeout=90) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for c in r.iter_content(1024*1024):f.write(c)
        data=path.read_bytes()
    else: data=dest.read_bytes()
    assert len(data)==meta["size"],name
    digest=hashlib.sha256(data).hexdigest()
    if "lfs" in meta:
        assert meta["lfs"]["oid"]==digest,name
    else:
        assert meta["oid"]==hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest(),name
    if not dest.exists():path.replace(dest)
    return {"name":name,"bytes":len(data),"sha256":digest}

r=requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{COMMIT}",params={"recursive":"true","expand":"true"},timeout=30);r.raise_for_status()
tree=r.json();(where/"pinned_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf8")
with ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(run,[x for x in tree if x["path"]!=".gitattributes"]))
(where/"verified_files.json").write_text(json.dumps(rows,indent=2),encoding="utf8");print(rows,flush=True)
