"""Lead-II KAN and GELU channel intervention on fixed LUDB splits; see protocol."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt
import torch
# WFDB 4.3.0 expects the pandas 2 string-array behaviour for its label table.
pd.options.future.infer_string=False
import wfdb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dis_test"))
from frozen_model import WIN, HOP, runs, merge, match
from rdb_external import models
from rdb_stage1 import checkpoint_loader

def lead(folder, ident):
    path=str(folder/ident)
    sig,fields=wfdb.rdsamp(path)
    assert fields["fs"]==500 and len(sig)==5000
    ii=[n.lower() for n in fields["sig_name"]].index("ii")
    samples=resample_poly(sig[:,ii],1,2)
    samples=sosfiltfilt(butter(2,[.5,40],btype="bandpass",fs=250,output="sos"),samples)
    scale=(np.quantile(samples,.75)-np.quantile(samples,.25))/1.349
    if scale<1e-8: scale=np.std(samples)+1e-8
    x=((samples-np.median(samples))/scale).astype("float32")
    ann=wfdb.rdann(path,"ii")
    q=np.rint(ann.sample/2).astype(int); syms=ann.symbol
    span=(q.min(),q.max()+1) if len(q) else (0,0)
    y=np.full(len(x),-1); y[slice(*span)]=0; ref=[]
    for i,s in enumerate(syms):
        if s in ("(",")","u"): continue
        on=q[i-1] if i and syms[i-1]=="(" else None
        off=q[i+1] if i+1<len(q) and syms[i+1]==")" else None
        if s=="p" and on is not None: ref.append(on)
        if on is None or off is None: y[max(0,q[i]-25):q[i]+25]=-1
        else: y[on:off+1]={"p":1,"t":3}.get(s,2)
    return x,y,np.array(ref,dtype=int),span

def fold_windows(m,x):
    starts=list(range(0,len(x)-WIN+1,HOP))
    if starts[-1]!=len(x)-WIN: starts.append(len(x)-WIN)
    win=torch.from_numpy(np.stack([x[i:i+WIN] for i in starts]))[:,None,:]
    z,skip=m.enc(win)
    if "bottleneck" in m.where: z=z+m.kb(z)
    return m.dec(z,skip),starts

def join(windows,starts,n):
    tap=np.hanning(WIN).astype("float32")+.001
    acc=np.zeros((windows.shape[1],n),dtype="float32");den=np.zeros(n,dtype="float32")
    for a,start in zip(windows,starts):
        acc[:,start:start+WIN]+=a*tap;den[start:start+WIN]+=tap
    return acc/den

def pick(m,val,kind):
    raw=[];labels=[]
    for x,y,_,_ in val:
        f,starts=fold_windows(m,x); raw.append(join(f.numpy(),starts,len(x)).T); labels.append(y)
    y=np.concatenate(labels); r=np.concatenate(raw)[y>=0];y=y[y>=0]
    norm=m.head.norm if kind=="GELU" else m.head.kan.norm
    a=norm(torch.from_numpy(r)).numpy()
    means=np.stack([a[y==i].mean(axis=0) for i in range(4)])
    i=int(np.argmax(means[1]-means[[0,2,3]].max(axis=0)))
    energy=np.mean(r*r,axis=0);j=min((c for c in range(len(energy)) if c!=i),key=lambda c:abs(energy[c]-energy[i]))
    return i,j,r.mean(axis=0)

def score(m,record,ids,means):
    x,_,ref,span=record
    f,starts=fold_windows(m,x)
    for condition,c in zip(("intact","selected","energy"),(None,*ids)):
        g=f.clone()
        if c is not None: g[:,c,:]=float(means[c])
        mask=join(torch.softmax(m.head(g),dim=1).numpy(),starts,len(x)).argmax(axis=0)
        det=np.array([a for a,b in merge(runs(mask,1)) if b-a+1>=10],dtype=int)
        r,matched,fn,_=match(ref,det,38)
        un=np.setdiff1d(det,matched);fp=np.sum((un>=span[0]) & (un<span[1]))
        yield condition,len(r),fn,int(fp)

def summarize(df):
    results=[]; prior={}
    keys=["head","seed","condition","fold","record"]
    indexed=df.set_index(keys).sort_index()
    ids=sorted(df.record.unique())
    rng=np.random.default_rng(20261125)
    wt=np.vstack([np.ones(len(ids),dtype=int),rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=10000)])
    def add(head,seed,metric,r):
        lo,hi=np.quantile(r[1:],[.025,.975])
        results.append(dict(head=head,seed=seed,metric=metric,estimate=r[0],lower=lo,upper=hi))
    for seed in (1,2):
        # Keep the archived CSV's KAN, GELU order for a byte-for-byte replay.
        for head in df["head"].drop_duplicates():
            vals={}
            for cond in ("intact","selected","energy"):
                sub=indexed.loc[(head,seed,cond),["tp","fn","fp"]].reindex(pd.MultiIndex.from_product([range(5),ids],names=["fold","record"]))
                assert sub.notna().all().all()
                agg=np.einsum("br,frc->bfc",wt,sub.to_numpy().reshape(5,len(ids),3),optimize=True)
                tp,fn,fp=np.moveaxis(agg,-1,0)
                vals[cond]=(200*tp/(2*tp+fn+fp)).mean(axis=1)
            prior[head]=vals["selected"]-vals["energy"]
            add(head,seed,"intact",vals["intact"])
            add(head,seed,"selected_minus_intact",vals["selected"]-vals["intact"])
            add(head,seed,"selected_minus_energy",prior[head])
        if {"KAN","GELU"} <= set(df["head"]):
            add("KAN minus GELU",seed,"difference_in_selected_minus_energy",prior["KAN"]-prior["GELU"])
    return pd.DataFrame(results)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for param in ("data","weights","splits","out"): ap.add_argument("--"+param,type=Path,required=True)
    ap.add_argument("--hybrid-only",action="store_true")
    a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    s=json.loads(a.splits.read_text());test=s["test"]
    ids=set(test)|{k for fold in s["folds"] for k in fold}
    assert len(ids)==200 and len(test)==41 and all(not set(f)&set(test) for f in s["folds"])
    records={k:lead(a.data,k) for k in ids}; rows=[]; picks=[]
    for seed in (1,2):
        for head in (("UNetKAN",) if a.hybrid_only else ("KAN","GELU")):
            nets,_=(checkpoint_loader if a.hybrid_only else models)(head,seed,a.weights)
            for fold,m in enumerate(nets):
                i,j,means=pick(m,[records[k] for k in s["folds"][fold]],head)
                picks.append(dict(head=head,seed=seed,fold=fold,selected=i,energy=j))
                for k in test:
                    for cond,tp,fn,fp in score(m,records[k],(i,j),means):
                        rows.append(dict(head=head,seed=seed,fold=fold,record=k,condition=cond,tp=tp,fn=fn,fp=fp))
                print(head,seed,fold,flush=True)
    df=pd.DataFrame(rows);df.to_csv(a.out/"intervention_counts.csv",index=False)
    pd.DataFrame(picks).to_csv(a.out/"selected_channels.csv",index=False)
    summarize(df).to_csv(a.out/"intervention_estimates.csv",index=False,float_format="%.8f")
if __name__=="__main__": main()
