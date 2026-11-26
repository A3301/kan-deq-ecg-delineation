"""Verify ID-level coverage and create the frozen patient partition without loading any models."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
W=ROOT/"source";OUT=ROOT/"audit";OUT.mkdir(exist_ok=True)
meta=pd.read_csv(W/"ptbxl_database.csv").set_index("ecg_id");rows=[];groups={}
for gr in ["AF","others"]:
    labels=np.load(W/(gr+".npy"),mmap_mode="r",allow_pickle=False)
    names=(W/(gr+".txt")).read_text().splitlines()
    assert labels.shape==(12*len(names),1200)
    lab=labels.reshape(len(names),12,1200)
    vals,counts=np.unique(lab,return_counts=True)
    groups[gr]=dict(subject_records=len(names),shape=list(labels.shape),dtype=str(labels.dtype),distinct_values=vals.tolist(),
                    counts=counts.tolist(),all_twelve_rows_identical=int(np.all(lab==lab[:,:1,:],axis=(1,2)).sum()))
    for i,name in enumerate(names):
        rid=int(name.split("_")[0]);r=meta.loc[rid]
        assert name==Path(r.filename_hr).name
        rows.append((rid,int(r.patient_id),gr,i,r.filename_hr))
d=pd.DataFrame(rows,columns=["ecg_id","patient_id","stratum","array_position","filename_hr"])
assert d.ecg_id.is_unique
ids=sorted(d.patient_id.unique(),key=lambda n:hashlib.sha256(("PTBXL-reference-audit-1|"+str(n)).encode("ascii")).hexdigest())
d["split"]=np.where(d.patient_id.isin(ids[:150]),"calibration","verification");d.to_csv(ROOT/"roster.csv",index=False)
audit=dict(groups=groups,records=len(d),patients=d.patient_id.nunique(),calibration_records=int(d.split.eq("calibration").sum()),calibration_patients=150,
           reserved_records=int(d.split.eq("verification").sum()),reserved_patients=len(ids)-150)
(OUT/"release_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
print(audit)
