"""Fixed ISP lead-II external test; see protocol.txt before using the output."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
from fractions import Fraction
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
if hasattr(pd.options.future,"infer_string"):
    pd.options.future.infer_string=False
import wfdb
from frozen_model import UNetKAN, predict_proba
from rdb_external import batched_probabilities, detections, models, POINTS

ROOT=Path(__file__).resolve().parent
PROTOCOL="1c01759b3e25896dd199d89f6f93bf900599d8abbbc4e966af6cce433530e2aa"
SOURCE_MD5="a070d5f7d5ff88ce823dbedc7412ee28"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(triples,n,fs):
    marks={k:[] for k in POINTS}
    qc=dict(bad_interval=0,zero_interval=0,censored_boundary=0,duplicate_mark=0)
    for cls,start,end in triples:
        if cls not in (0,1,2) or end<start:
            qc["bad_interval"]+=1
            continue
        if end==start:
            qc["zero_interval"]+=1
            continue
        for offset,sample in enumerate((start,end)):
            if sample<=0 or sample>=n-1:
                qc["censored_boundary"]+=1
                continue
            marks[POINTS[2*cls+offset]].append(1000*sample/fs)
    for name,values in marks.items():
        clean=sorted(set(values))
        qc["duplicate_mark"]+=len(values)-len(clean)
        marks[name]=np.array(clean,dtype=float)
    return marks,qc


def prepare(archive,out):
    with archive.open("rb") as f:
        assert hashlib.file_digest(f,"md5").hexdigest()==SOURCE_MD5
    records=[]; roster=[]; seen={}
    filt=butter(2,[.5,40],btype="bandpass",fs=250,output="sos")
    with TemporaryDirectory() as td,ZipFile(archive) as z:
        for split in ("train","test"):
            prefix="isp_delineation_dataset/"+split
            table=list(csv.DictReader(io.StringIO(z.read(prefix+"_isp_delineation_data.csv").decode("utf-8-sig"))))
            for row in sorted(table,key=lambda x:int(x["file_name"])):
                name=row["file_name"]
                assert name.isdigit()
                rid=split+"/"+name
                hdr=z.read(prefix+"_data/"+name+".hea");raw=z.read(prefix+"_data/"+name+".dat")
                h=hashlib.sha256(raw).hexdigest()
                rowmeta=dict(record=rid,partition=split,file_name=name,raw_sha256=h,header_sha256=hashlib.sha256(hdr).hexdigest(),
                             status="included")
                roster.append(rowmeta)
                if h in seen:
                    rowmeta["status"]="duplicate of "+seen[h]
                    continue
                p=Path(td)/name
                p.with_suffix(".hea").write_bytes(hdr);p.with_suffix(".dat").write_bytes(raw)
                sig,meta=wfdb.rdsamp(str(p))
                fs=int(meta["fs"]); assert fs==meta["fs"]
                names=[x.lower() for x in meta["sig_name"]];assert names.count("ii")==1
                x=sig[:,names.index("ii")]
                marks,qc=reference(ast.literal_eval(row["target"]),len(x),fs)
                obs=np.concatenate(list(marks.values()))
                rowmeta.update(native_fs=fs,native_samples=len(x),lead_index_zero_based=names.index("ii"),**qc)
                if not len(obs) or not np.isfinite(x).all():
                    rowmeta["status"]="missing reference or nonfinite input"
                    continue
                seen[h]=rid
                start,end=float(obs.min()),float(obs.max())
                gaps=np.diff(marks["QRS_on"])
                rowmeta.update(span_start_ms=start,span_end_ms=end,span_ms=end-start,
                               qrs_gaps_gt_2s=int((gaps>2000).sum()),
                               longest_qrs_gap_ms=float(gaps.max()) if len(gaps) else 0,
                               **{k:len(v) for k,v in marks.items()})
                factor=Fraction(250,fs)
                x=sosfiltfilt(filt,resample_poly(x,factor.numerator,factor.denominator))
                q25,q75=np.percentile(x,[25,75]);scale=(q75-q25)/1.349
                if scale<1e-8: scale=x.std()+1e-8
                x=((x-np.median(x))/scale).astype(np.float32)
                assert len(x)==2500
                records.append((rid,x,marks,(start,end)))
    pd.DataFrame(roster).to_csv(out/"cohort.csv",index=False)
    return records


def score(ref,det,tol,span):
    distances=np.abs(ref[:,None]-det[None,:])
    ii,jj=np.nonzero(distances<=tol)
    used_i=set();used_j=set()
    for k in np.argsort(distances[ii,jj],kind="stable"):
        i,j=ii[k],jj[k]
        if i not in used_i and j not in used_j:
            used_i.add(i);used_j.add(j)
    fp=sum(j not in used_j and span[0]<=v<=span[1] for j,v in enumerate(det))
    return len(used_i),len(ref)-len(used_i),int(fp)


def get_net(source,head,seed,rdb,ludb):
    if source=="LUDB":
        return models(head,seed,ludb)
    p=rdb/f"{head}_{seed}.pt";md=json.loads(p.with_suffix(".json").read_text())
    assert md["head"]==head and md["seed"]==seed and md["steps"]==2400 and md["weight_sha256"]==file_hash(p)
    net=UNetKAN(where=["head"]) if head=="KAN" else UNetKAN(where=[],mlp_head=29)
    net.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    return [net.eval()],{p.name:md["weight_sha256"]}


def run(archive,ludb,rdb,out):
    assert file_hash(ROOT/"protocol.txt")==PROTOCOL
    out.mkdir(parents=True,exist_ok=True)
    records=prepare(archive,out)
    print("Eligible records",len(records),flush=True)
    X=np.stack([r[1] for r in records]);rows=[];audit=[]
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with torch.inference_mode():
        for source,nseed in (("RDB",3),("LUDB",2)):
            for seed in range(1,nseed+1):
                for head in ("KAN","GELU"):
                    nets,hashes=get_net(source,head,seed,rdb,ludb)
                    masks=[]
                    for i in range(0,len(X),8):
                        p=batched_probabilities(nets,X[i:i+8])
                        m=p.argmax(axis=1)
                        if i==0:
                            single=predict_proba(nets,X[0]);drift=float(np.max(np.abs(single-p[0])))
                            disagreement=int(np.sum(single.argmax(axis=0)!=m[0]))
                            assert drift<.0003 and disagreement==0
                        masks.extend(m)
                    for (rid,x,ref,span),mask in zip(records,masks,strict=True):
                        for mark,det in zip(POINTS,detections(mask),strict=True):
                            for tol in (150,40):
                                tp,fn,fp=score(ref[mark],det,tol,span)
                                rows.append((source,head,seed,rid,tol,mark,tp,fn,fp))
                    audit.append(dict(source=source,head=head,seed=seed,n_folds=len(nets),n=len(masks),
                                 models=hashes,first_record_single_batch_max_error=drift,first_record_single_batch_disagreement=disagreement))
                    print(source,seed,head,len(masks),flush=True)
    df=pd.DataFrame(rows,columns=["source","head","seed","record","tol_ms","landmark","tp","fn","fp"])
    assert df.groupby(["record","landmark"])[["tp","fn"]].apply(lambda d:(d.tp+d.fn).nunique()).max()==1
    df.to_csv(out/"event_counts.csv",index=False)
    (out/"inference_audit.json").write_text(json.dumps(dict(protocol=PROTOCOL,reference_zip_md5=SOURCE_MD5,
                    torch=str(torch.__version__),numpy=np.__version__,wfdb=wfdb.__version__,models=audit),indent=2),encoding="utf-8")
    summarize(df).to_csv(out/"comparison.csv",index=False)


def summarize(df):
    ids=sorted(df.record.unique());rng=np.random.default_rng(20261125)
    b=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=10000)
    weights=np.vstack([np.ones(len(ids),dtype=int),b]);rows=[]
    for (source,tol),g in df.groupby(["source","tol_ms"],sort=False):
        vals={}
        seeds=sorted(g.seed.unique())
        for (head,seed),d in g.groupby(["head","seed"]):
            index=pd.MultiIndex.from_product([ids,POINTS],names=["record","landmark"])
            a=d.set_index(["record","landmark"]).reindex(index)[["tp","fn","fp"]]
            assert not a.isna().any().any() and len(a)==len(d)
            counts=(weights@a.to_numpy().reshape(len(ids),-1)).reshape(len(weights),len(POINTS),3)
            tp,fn,fp=np.moveaxis(counts,-1,0)
            f=200*tp/(2*tp+fn+fp)
            vals[head,seed]=np.column_stack([f,f.mean(1)])
        for seed in [*seeds,"mean"]:
            a,b=[np.mean([vals[h,s] for s in seeds],axis=0) if seed=="mean" else vals[h,seed] for h in ("KAN","GELU")]
            delta=a-b;lo,hi=np.quantile(delta[1:],[.025,.975],axis=0)
            for i,k in enumerate([*POINTS,"macro"]):
                rows.append((source,tol,seed,k,a[0,i],b[0,i],delta[0,i],lo[i],hi[i]))
    return pd.DataFrame(rows,columns=["source","tol_ms","seed","landmark","KAN_F1","GELU_F1","difference_pp","lower","upper"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isp",type=Path,required=True)
    parser.add_argument("--ludb-weights",type=Path,required=True)
    parser.add_argument("--rdb-models",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"recomputed")
    args=parser.parse_args()
    run(args.isp,args.ludb_weights,args.rdb_models,args.out)
