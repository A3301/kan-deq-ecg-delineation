"""Fetch the official RDB v4 files into ./raw/ and validate digests."""
from pathlib import Path
import hashlib, shutil, time
from urllib.request import urlopen

def main():
    destination=Path(__file__).resolve().parent/'raw';destination.mkdir(exist_ok=True)
    for name,ident,digest in [('RDB.dataset.zip',54060602,'d24b8c98635bd14c358b58fc432d82db6eb830b8597cb4793e50f43fe3df04d7'),('rdmap.xlsx',67725117,'5ad900a244a0171f0b49a5e4f402311ac5d521f8564c994f696f66a64bcf7f9a')]:
        path=destination/name
        if path.exists():
            with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,path
            continue
        partial=path.with_suffix('.partial');url=f'https://api.figshare.com/v2/file/download/{ident}?download={time.time_ns()}'
        with urlopen(url,timeout=120) as reader,partial.open('wb') as writer:shutil.copyfileobj(reader,writer)
        with partial.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
        partial.rename(path)

if __name__=='__main__': main()
