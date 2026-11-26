"""Recompute publication tables from the selected ZIP members in a fresh isolated directory."""
import argparse
from pathlib import Path, PurePosixPath
import sys
import subprocess
import tempfile
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent

def verify(path):
    interim="Interim" in path.name
    tags=["ensemble","fold0","homogeneous"]+( [] if interim else ["harmonized_ensemble"])
    matched=0
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
        dest=Path(tmp).resolve()
        for name in z.namelist():
            member=PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:raise ValueError(name)
            if name in ("splits.json","analyse.py","audit_results.py") or (name.startswith("results/") and name.endswith(".npz")):
                out=(dest/member).resolve()
                if not out.is_relative_to(dest):raise ValueError(name)
                out.parent.mkdir(exist_ok=True,parents=True);out.write_bytes(z.read(name))
        for tag in tags:
            for program in ["analyse.py","audit_results.py"]:
                subprocess.run([sys.executable,"-I",str(dest/program),tag],cwd=dest,check=True,capture_output=True)
            for suffix in ("scores.csv","contrasts.csv","subject_counts.csv","qa.json","independent_replay.json"):
                name=f"results/{tag}_{suffix}"
                assert (dest/name).read_bytes()==z.read(name),name
                matched+=1
    with path.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    output=dict(archive=path.name,archive_sha256=sha,panels=tags,recomputed_tables_and_audits_identical=matched,
                source="fresh temporary directory, discrete ZIP members, separate Python subprocess")
    (ROOT/("interim_archive_clean_replay.json" if interim else "archive_clean_replay.json")).write_text(json.dumps(output,indent=2))
    print(output)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path)
    verify(p.parse_args().archive.resolve())
