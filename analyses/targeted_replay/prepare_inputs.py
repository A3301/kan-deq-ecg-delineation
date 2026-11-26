"""Prepare LUDB 1.0.1 and the 30 channel-intervention weights from their two source ZIPs."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
PREFIX = "lobachevsky-university-electrocardiography-database-1.0.1/"
SOURCE = "https://physionet.org/static/published-projects/ludb/" + PREFIX[:-1] + ".zip"
OFFICIAL = "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f"


def manifest(raw):
    rows={}
    for line in raw.decode("utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        rows[name.strip()] = digest
    return rows


def save_member(z, member, path, expected, cap):
    info=z.getinfo(member)
    if info.file_size>cap:
        raise ValueError(f"Unexpected member size: {member}")
    b=z.read(info)
    if sha256(b).hexdigest()!=expected:
        raise ValueError(f"SHA-256 mismatch: {member}")
    if path.exists():
        if path.read_bytes()!=b:
            raise FileExistsError(f"Existing file differs: {path}")
    else:
        path.write_bytes(b)


def ludb(archive, dest):
    official=(ROOT/"LUDB_official_SHA256SUMS.txt").read_bytes()
    if sha256(official).hexdigest()!=OFFICIAL:
        raise ValueError("PhysioNet release manifest changed")
    entries=manifest(official)
    if archive is None:
        with urlopen(SOURCE,timeout=120) as resp:
            archive=BytesIO(resp.read(25_000_001))
        if archive.getbuffer().nbytes>25_000_000:
            raise ValueError("Unexpected LUDB archive size")
    dest.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as z:
        if z.read(PREFIX+"SHA256SUMS.txt")!=official:
            raise ValueError("The source archive has a different release manifest")
        for rec in range(1,201):
            for ext in ("dat","hea","ii"):
                file=f"{rec}.{ext}"
                member="data/"+file
                save_member(z,PREFIX+member,dest/file,entries[member],2_000_000)
    print("600 LUDB files matched the official version-1.0.1 SHA-256 manifest.")


def weights(archive, dest):
    dest.mkdir(parents=True,exist_ok=True)
    entries=manifest((ROOT/"channel_weights_SHA256SUMS.txt").read_bytes())
    assert len(entries)==30
    with ZipFile(archive) as z:
        their=manifest(z.read("SHA256SUMS.txt"))
        for file, digest in entries.items():
            if not file.isascii() or Path(file).name!=file or not file.endswith(".pt"):
                raise ValueError(file)
            member="weights/"+file
            if their[member]!=digest:
                raise ValueError(f"Accompanying archive manifest mismatch: {member}")
            save_member(z,member,dest/file,digest,8_000_000)
    print("30 verified channel weights prepared from the accompanying RDB archive.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rdb-zip",type=Path,required=True,help="RDB_external_validation.zip from the same computational package")
    ap.add_argument("--ludb-zip",type=Path,help="Optional local PhysioNet version-1.0.1 release ZIP; otherwise downloaded")
    ap.add_argument("--data",type=Path,default=ROOT/"kan_gelu"/"LUDB")
    ap.add_argument("--weights",type=Path,default=ROOT/"channel_weights")
    args=ap.parse_args()
    ludb(args.ludb_zip,args.data)
    weights(args.rdb_zip,args.weights)
