"""Build and verify the shareable training-variability replay package."""
import hashlib
import argparse
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def pack(interim=False):
    destination=ROOT.parent/("ECG_Training_Variability_Interim_Replay.zip" if interim else "ECG_Training_Variability_Replay.zip")
    files=[]
    for pattern in ["*.py","protocol.txt","scheduling.txt","homogeneous_protocol.txt","harmonized_protocol.txt","README.txt","*insert.md","inventory.txt","source_archive.sha256.txt",
                    "chosen_lr.json","splits.json","validation.json","source/*.py","source/*.csv","weights/*.pt","logs/*.json",
                    "source/scoring_amendment.txt","homogeneous_weights/*.pt","homogeneous_logs/*.json","results/*","cpu_replay/*.npz","cpu_harmonized/*.npz","native/*","native/data/*"]:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    files=sorted(set(files))
    if interim:
        files=[p for p in files if "harmonized" not in p.name.lower() and "cpu_harmonized" not in p.parts
               and (p.parent.name not in ("homogeneous_weights","homogeneous_logs") or "_f0_" in p.name)]
    assert len(list((ROOT/"weights").glob("*.pt")))==120 and len(list((ROOT/"logs").glob("*.json")))==60
    assert (ROOT/"results/ensemble_independent_replay.json") in files
    assert (ROOT/"results/homogeneous_checkpoint_manifest.csv") in files
    assert len([p for p in files if p.parent.name=="homogeneous_weights"])==(12 if interim else 60)
    assert (ROOT/"results/cpu_gpu_inference_audit.json") in files
    if not interim:
        assert (ROOT/"results/harmonized_ensemble_checkpoint_manifest.csv") in files
        assert (ROOT/"results/harmonized_cpu_gpu_inference_audit.json") in files
    manifest=[]
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:
            data=p.read_bytes()
            name=p.relative_to(ROOT).as_posix()
            z.writestr(name,data)
            manifest.append(hashlib.sha256(data).hexdigest()+"  "+name)
        z.writestr("SHA256SUMS.txt","\n".join(manifest)+"\n")
        z.writestr("PACKAGE_SCOPE.txt", ("Interim release: 120 diverse-origin five-fold members and the 24-member same-GPU single-fold sensitivity; 132 unique weight files." if interim else "Complete release: 120 homogeneous GPU-trained fold weights and 60 archived CPU weights, giving 180 unique files across all panels."))
    with zipfile.ZipFile(destination,"r") as z:
        assert z.testzip() is None
        for ln in z.read("SHA256SUMS.txt").decode().splitlines():
            sha,entry=ln.split("  ",1)
            assert sha==hashlib.sha256(z.read(entry)).hexdigest()
    with destination.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    (ROOT/("interim_zip.sha256.txt" if interim else "replay_zip.sha256.txt")).write_text(digest+"  "+destination.name+"\n")
    print(destination,destination.stat().st_size,digest)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--interim",action="store_true")
    pack(ap.parse_args().interim)
