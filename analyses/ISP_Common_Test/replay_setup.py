"""Get the unmodified public ISP release and select frozen weights from accompanying archives."""
import argparse
from hashlib import sha256, md5
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent
URL="https://zenodo.org/records/14679837/files/isp_delineation_dataset.zip?download=1"
MD5="a070d5f7d5ff88ce823dbedc7412ee28"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rdb-validation",type=Path,required=True,help="RDB_external_validation.zip (LUDB-trained weights)")
    ap.add_argument("--reverse-training",type=Path,required=True,help="Reverse_source_training.zip (RDB-trained weights)")
    ap.add_argument("--isp",type=Path,help="Existing official ISP ZIP, otherwise downloaded to --out")
    ap.add_argument("--out",type=Path,default=ROOT/"inputs")
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    if args.isp:
        assert md5(args.isp.read_bytes()).hexdigest()==MD5
        print("Verified ISP release",args.isp)
    else:
        target=args.out/"isp_delineation_dataset.zip"
        if not target.exists():
            with urlopen(URL,timeout=120) as resp:
                b=resp.read(42_000_001)
            assert len(b)==41_016_366 and md5(b).hexdigest()==MD5
            target.write_bytes(b)
        assert md5(target.read_bytes()).hexdigest()==MD5
        print("Verified ISP release",target)
    meta=json.loads((ROOT/"results"/"inference_audit.json").read_text(encoding="utf-8"))
    for source,archive,prefix in (("RDB",args.reverse_training,"models/"),("LUDB",args.rdb_validation,"weights/")):
        dest=args.out/source
        dest.mkdir(exist_ok=True)
        with ZipFile(archive) as z:
            manifest={name:digest for digest,name in (x.split(maxsplit=1) for x in z.read("SHA256SUMS.txt").decode().splitlines())}
            files={name:val for m in meta["models"] if m["source"]==source for name,val in m["models"].items()}
            for name,expected in files.items():
                assert Path(name).name==name
                names=[name,name.replace(".pt",".json")] if source=="RDB" else [name]
                for f in names:
                    b=z.read(prefix+f)
                    digest=sha256(b).hexdigest()
                    assert digest==manifest[prefix+f]
                    if f==name: assert digest==expected
                    target=dest/f
                    if target.exists() and target.read_bytes()!=b:raise FileExistsError(target)
                    target.write_bytes(b)
        print("Verified",len(files),source,"trained checkpoints")

if __name__=="__main__":
    main()
