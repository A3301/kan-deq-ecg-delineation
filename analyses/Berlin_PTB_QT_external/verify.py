"""Independent recount of the primary endpoint and one patient-bootstrap interval."""
import csv
import json
from pathlib import Path
import numpy as np
from extract_reference import reference_bytes
from prepare_clips import ROOT, SOURCE, digest

def main():
    out=ROOT/"results"
    assert digest(ROOT/"protocol.txt")=="f000640816ba7b07191bbf2d25e8e1bfa54ceb2b74b5ffffbaa9574ad7b9cdfc"
    assert reference_bytes()==(SOURCE/"manual_reference.csv").read_bytes()
    with (out/"per_record.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    groups={}
    returned={}
    for r in records:
        group=groups.setdefault(r["patient"],{})
        model=(r["head"],int(r["seed"]))
        group.setdefault(model,[])
        success=False
        if r["predicted_t_ms"]:
            qt=int(float(r["predicted_t_ms"])-float(r["predicted_q_ms"]))
            success=abs(qt-(int(r["reference_t_ms"])-int(r["reference_q_ms"])))<=40
        group[model].append(int(success))
        returned.setdefault(r["id"],[]).append(bool(r["predicted_t_ms"]))
    assert len(groups)==289 and len(records)==548*6
    assert sum(all(v) for v in returned.values())==547
    rng=np.random.default_rng(20261125)
    keys=sorted(groups)
    counts=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=10000)
    delta=np.array([sum(sum(v) for (h,_),v in groups[k].items() if h=="KAN")-
                    sum(sum(v) for (h,_),v in groups[k].items() if h=="GELU") for k in keys])
    denominators=np.array([sum(len(v) for (h,_),v in groups[k].items() if h=="KAN") for k in keys])
    estimate=100*sum(delta)/sum(denominators)
    low,high=np.quantile(100*(counts@delta)/(counts@denominators),[.025,.975])
    with (out/"summary.csv").open(newline="") as f:
        row=next(r for r in csv.DictReader(f) if r["metric"]=="QT_success_40ms")
    np.testing.assert_allclose([estimate,low,high],[float(row[x]) for x in ["difference","difference_lo","difference_hi"]],atol=1e-10)
    print(f"Independent all-record and patient-cluster recount passed, difference {estimate:.6f} pp [{low:.6f}, {high:.6f}].")
    meta=json.loads((out/"inference_manifest.json").read_text())
    for path,sha in meta.items():
        if path!="software":
            assert digest(ROOT/path)==sha, path
    print("Verified input and model digests.")

if __name__=="__main__":
    main()
