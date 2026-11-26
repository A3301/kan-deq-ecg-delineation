"""Verify every bundled native LUDB waveform and annotation against the publisher's SHA-256 file."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    native=ROOT/"native"
    source=dict((str(path),sha) for sha,path in (line.split(maxsplit=1) for line in
                (native/"SHA256SUMS.txt").read_text().splitlines() if line.strip()))
    files=[p for p in native.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt"]
    for p in files:
        ref=source[p.relative_to(native).as_posix()]
        with p.open("rb") as f:actual=hashlib.file_digest(f,"sha256").hexdigest()
        if ref!=actual:raise ValueError(f"Publisher checksum mismatch for {p}")
    assert len(files)==2802
    data_files=sum(p.parent.name=="data" for p in files)
    assert data_files==2800
    print(json.dumps(dict(waveform_header_and_annotation_files=data_files,manifest_and_license_files=2,
                          authority="PhysioNet LUDB 1.0.1",sha256_mismatches=0)))

if __name__=="__main__":main()
