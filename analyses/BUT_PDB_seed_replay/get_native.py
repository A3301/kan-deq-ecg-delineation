"""Restore the native release subset by pinned PhysioNet checksums."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from urllib.request import urlopen
from time import sleep

BASE="https://physionet.org/files/but-pdb/1.0.0/"
PIN="b7231f90c5c167dec5a151e099e0b9732a90b1ee48c2a0c8d0ebcb563187850c"
OUT=Path(__file__).resolve().parent/"source"/"native"

def download(name,sha):
    path=OUT/name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha:
        return
    for attempt in range(4):
        try:
            with urlopen(BASE+name,timeout=45) as response:
                content=response.read()
            if hashlib.sha256(content).hexdigest()!=sha:
                raise ValueError("Checksum mismatch: "+name)
            temp=path.with_suffix(path.suffix+".part")
            temp.write_bytes(content)
            temp.replace(path)
            return
        except OSError:
            if attempt==3:raise
            sleep(2**attempt)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    download("SHA256SUMS.txt",PIN)
    manifest=dict((name,digest) for digest,name in
                  (l.split(maxsplit=1) for l in (OUT/"SHA256SUMS.txt").read_text().splitlines() if l.strip()))
    names=[f"{i:02d}.{ext}" for i in range(1,51) for ext in ("hea","dat","pwave")]+["README.txt","LICENSE.txt"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name:download(name,manifest[name]),names))
    print("Verified the 153 release files used by this replay.")

if __name__=="__main__":
    main()
