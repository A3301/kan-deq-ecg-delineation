"""Extract just the 60 locked folds from the accompanying training ZIP by verified SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent

def main(bundle,target):
    target.mkdir(exist_ok=True,parents=True)
    met=json.loads((ROOT/"results"/"weights.json").read_text())
    expected={name:h for m in met.values() for name,h in m.items() if name.endswith(".pt")}
    assert len(expected)==60
    with ZipFile(bundle) as z:
        for name,sha in expected.items():
            assert name==Path(name).name
            payload=z.read("weights/"+name)
            assert hashlib.sha256(payload).hexdigest()==sha,name
            (target/name).write_bytes(payload)
    print("Validated all 60 immutable fold weights.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("bundle",type=Path)
    ap.add_argument("--target",type=Path,default=ROOT/"weights")
    a=ap.parse_args()
    main(a.bundle,a.target)
