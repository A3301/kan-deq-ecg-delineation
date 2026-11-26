"""Download the two public RDB v4 files and verify their publisher-provided MD5."""
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen


FILES = [
    ("dataset.zip","54060602","15edc780ac478b3d723fcec84f23c9a0"),
    ("rdb_chapman_mapping.xlsx","67725117","9c67ed075e126a39db90bc8b07a9609b"),
]


def checksum(p):
    with p.open("rb") as src:
        return hashlib.file_digest(src,"md5").hexdigest()


if __name__=="__main__":
    dest = Path("data")
    dest.mkdir(exist_ok=True)
    for name, file_id, expected in FILES:
        target = dest/name
        if target.exists() and checksum(target)==expected:
            print("Verified",target)
            continue
        if target.exists():
            raise ValueError(f"An existing file differs from the official release: {target}")
        partial = dest/(name+".part")
        with urlopen("https://ndownloader.figshare.com/files/"+file_id,timeout=120) as src, partial.open("wb") as out:
            shutil.copyfileobj(src,out)
        if checksum(partial)!=expected:
            raise ValueError(f"Downloaded file differs from the official release: {partial}")
        partial.replace(target)
        print("Downloaded and verified",target)
