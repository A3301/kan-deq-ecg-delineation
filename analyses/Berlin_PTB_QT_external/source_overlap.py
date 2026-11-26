"""Optional cross-source waveform screen; aligned duplicates only, not identity linkage."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
if hasattr(pd.options.future, "infer_string"):
    pd.options.future.infer_string=False
import wfdb
from prepare_clips import ROOT, SOURCE, digest

FILTER=butter(2,[.5,40],fs=250,btype="bandpass",output="sos")
def filter_250(signal,fs):
    factor=Fraction(250,int(fs))
    result=sosfiltfilt(FILTER,resample_poly(signal,factor.numerator,factor.denominator))
    assert len(result)==2500
    return result

def extract_wfdb(path):
    r=wfdb.rdrecord(str(path),channel_names=["ii"])
    assert r.p_signal.shape[1]==1
    return filter_250(r.p_signal[:,0],r.fs)

def normalise(data):
    values=resample_poly(np.stack(data).astype(float),1,5,axis=1)[:,25:-25].copy()
    values-=values.mean(axis=1,keepdims=True)
    return values/np.linalg.norm(values,axis=1,keepdims=True)

def main(rdb,isp,ludb):
    with np.load(SOURCE/"ptb_lead_ii_10s.npz") as d:
        berlin=normalise([filter_250(x.astype(float),1000) for x in d["digital"]])
    assert len(berlin)==549
    # Pearson normalization removes the native offset/gain and the source-model z scaling.
    sets={"RDB":np.load(rdb)}
    assert hashlib.md5(isp.read_bytes()).hexdigest()=="a070d5f7d5ff88ce823dbedc7412ee28"
    with ZipFile(isp) as z, TemporaryDirectory() as tmp:
        bucket=Path(tmp)
        extracted=[]
        for header in sorted(p for p in z.namelist() if p.endswith(".hea")):
            data=header[:-4]+".dat"
            target=bucket/Path(header).name
            target.write_bytes(z.read(header))
            target.with_suffix(".dat").write_bytes(z.read(data))
            extracted.append(extract_wfdb(target.with_suffix("")))
        sets["ISP"]=extracted
    sets["LUDB"]=[extract_wfdb(p.with_suffix("")) for p in sorted(ludb.glob("*.hea"))]
    assert [len(x) for x in sets.values()]==[2398,475,200]
    results={}
    for name,data in sets.items():
        similarity=np.abs(berlin@normalise(data).T)
        assert np.isfinite(similarity).all()
        results[name]=dict(reference_records=len(data),ptb_clips=len(berlin),
                           pairs_examined=int(similarity.size),max_abs_aligned_r=float(similarity.max()),
                           pairs_r_ge_0_98=int((similarity>=.98).sum()))
    results["note"]="Aligned central 9-s absolute Pearson correlation after common filtering and 50-Hz resampling. Different segments, staggered traces or different visits may not match. Public records do not permit identity linkage."
    results["rdb_array_sha256"]=digest(rdb)
    (ROOT/"results"/"source_overlap.json").write_text(json.dumps(results,indent=2))
    print(results)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ["rdb","isp","ludb"]:p.add_argument(arg,type=Path)
    args=p.parse_args()
    main(args.rdb,args.isp,args.ludb)
