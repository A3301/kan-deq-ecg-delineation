"""Independent confidence-interval replay using subject frequency vectors and the exported long CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
POINTS=["P onset","P offset","QRS onset","QRS offset","T onset","T offset"]

def rows(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def audit(tag):
    root=HERE/"results"
    subjects=sorted(json.loads((HERE/"splits.json").read_text())["test"],key=int)
    counts=rows(root/f"{tag}_subject_counts.csv")
    cube={}
    for r in counts:
        key=(r["architecture"],int(r["tolerance_ms"]))
        if key not in cube:cube[key]=np.full((41,3,6,3),-1)
        p=subjects.index(r["subject"]);s=int(r["seed"])-1;q=POINTS.index(r["landmark"])
        assert (cube[key][p,s,q]==-1).all()
        cube[key][p,s,q]=[int(r[name]) for name in ["tp","fn","fp"]]
    ids=np.random.default_rng(412026).integers(0,41,(10000,41))
    freq=np.stack([np.bincount(ids_,minlength=41) for ids_ in ids])
    weights=np.vstack([np.ones(41),freq])
    by_arch={}
    for key,c in cube.items():
        assert (c>=0).all()
        grouped=(weights @ c.reshape(41,-1)).reshape(-1,3,6,3)
        tp,fn,fp=np.moveaxis(grouped,-1,0);by_arch[key]=(100*tp/(tp+.5*(fn+fp))).mean(axis=-1)
    checked=0
    for r in rows(root/f"{tag}_scores.csv"):
        key=(r["architecture"],int(r["tolerance_ms"]))
        for s in (1,2,3):
            np.testing.assert_allclose(by_arch[key][0,s-1],float(r[f"seed_{s}"]),rtol=0,atol=1e-10)
            checked+=1
    for r in rows(root/f"{tag}_contrasts.csv"):
        a=(r["model_a"],int(r["tolerance_ms"]));b=(r["model_b"],int(r["tolerance_ms"]))
        d=by_arch[a]-by_arch[b]
        vals=d.mean(axis=1) if r["level"]=="mean_of_three" else d[:,int(r["level"][-1])-1]
        np.testing.assert_allclose([vals[0],*np.quantile(vals[1:],[.025,.975])],
                                   [float(r[k]) for k in ["difference_pp","lower_95","upper_95"]],
                                   rtol=0,atol=1e-10);checked+=3
    assert checked==192
    (root/f"{tag}_independent_replay.json").write_text(json.dumps(dict(checked_numerical_results=checked,
        maximum_accepted_arithmetic_difference_pp=1e-10, tests_passed=True, rows=len(counts)),indent=2))
    print("Independent long-table replay passed",tag,checked)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("tag",choices=["fold0","ensemble","homogeneous","harmonized_ensemble"]);args=p.parse_args();audit(args.tag)
