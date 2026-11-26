import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request

EXPECTED = "d03192c7361ab5deeaba3f0d46e274b9ecb0b74c9caf8964151e20f1b5a7df06"
URL = "https://physionet.org/content/ludb/get-zip/1.0.1/"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main(path):
    if path.exists():
        if sha(path) == EXPECTED:
            print("Reusing verified LUDB v1.0.1 archive.")
            return
        raise ValueError("Existing archive has a different hash.")
    path.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        candidate = Path(temp)/"source.zip"
        with urllib.request.urlopen(URL,timeout=90) as source, candidate.open("wb") as dest:
            shutil.copyfileobj(source,dest)
        if sha(candidate) != EXPECTED:
            raise ValueError("Source archive differs from the release used for this analysis.")
        candidate.replace(path)
    print("Downloaded and verified LUDB v1.0.1.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    main(p.parse_args().path)
