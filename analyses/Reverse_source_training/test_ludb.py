"""Score frozen reverse-source fits only after all six RDB selections are final."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
if hasattr(pd.options.future,'infer_string'):pd.options.future.infer_string=False
import wfdb
from model import merge, runs, predict_proba
from train import create

ROOT=Path(__file__).resolve().parent
POINTS=('P_on','P_off','QRS_on','QRS_off','T_on','T_off')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_ludb(data,original_split):
    split=json.loads(original_split.read_text());ids=split['test'];assert len(ids)==len(set(ids))==41
    assert all(set(ids).isdisjoint(f) for f in split['folds']);sos=butter(2,[.5,40],fs=250,btype='bandpass',output='sos');subjects=[];manifest={}
    for rid in ids:
        path=str(data/rid);raw,meta=wfdb.rdsamp(path);assert meta['fs']==500 and len(raw)==5000
        li=[k.lower() for k in meta['sig_name']].index('ii');x=sosfiltfilt(sos,resample_poly(raw[:,li],1,2));q25,q75=np.quantile(x,[.25,.75]);scale=(q75-q25)/1.349;scale=scale if scale>=1e-8 else x.std()+1e-8
        ann=wfdb.rdann(path,'ii');symbols=ann.symbol;times=ann.sample.astype(int)*2
        assert (np.diff(times)>=0).all() and set(symbols)<=set(['(',')','p','N','t','u']),set(symbols)
        ref={k:[] for k in POINTS};delta=0
        for j,s in enumerate(symbols):
            if s not in ['p','N','t']:continue
            kind={'p':'P','N':'QRS','t':'T'}[s]
            if j>0 and symbols[j-1]=='(':ref[kind+'_on'].append(int(times[j-1]))
            if j+1<len(times) and symbols[j+1]==')':ref[kind+'_off'].append(int(times[j+1]))
        subjects.append((rid,((x-np.median(x))/scale).astype(np.float32),ref,(times.min(),times.max())))
        manifest[rid]={ext:sha(data/(rid+'.'+ext)) for ext in ['hea','dat','ii']}
    return subjects,manifest

def score(ref,pred,tol,span):
    ref=np.asarray(ref,dtype=int);pred=np.asarray(pred,dtype=int);dist=np.abs(ref[:,None]-pred[None,:]);ix,jx=np.where(dist<=tol);order=np.argsort(dist[ix,jx],kind='stable');used_i=set();used_j=set()
    for n in order:
        if ix[n] not in used_i and jx[n] not in used_j:used_i.add(ix[n]);used_j.add(jx[n])
    fp=sum(j not in used_j and span[0]<=val<=span[1] for j,val in enumerate(pred));return len(used_i),len(ref)-len(used_i),int(fp)

def read_model(head,seed):
    weight=ROOT/'models'/f'{head}_{seed}.pt';stats=json.loads(weight.with_suffix('.json').read_text());assert stats['steps']==2400 and stats['protocol']==sha(ROOT/'protocol.txt') and stats['weight_sha256']==sha(weight)
    net=create(head);net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));return net.eval()

def summarize(df):
    rng=np.random.default_rng(20261125);order=sorted(df.record.unique());weights=rng.multinomial(len(order),np.full(len(order),1/len(order)),size=10000);weights=np.vstack([np.ones(len(order),dtype=int),weights]);out=[]
    for tol,subset in df.groupby('tol_ms',sort=False):
        values={}
        for (head,seed),g in subset.groupby(['head','seed']):
            shares=[]
            for key in POINTS:
                t=g[g.landmark==key].set_index('record').loc[order,['tp','fn','fp']].values;cnt=weights@t
                f=200*cnt[:,0]/(2*cnt[:,0]+cnt[:,1]+cnt[:,2]);assert np.isfinite(f).all();shares.append(f)
            values[head,seed]=np.array(shares+[np.mean(shares,axis=0)])
        for seed in [1,2,3,'mean']:
            a,b=[np.mean([values[h,s] for s in [1,2,3]],axis=0) if seed=='mean' else values[h,seed] for h in ['KAN','GELU']]
            for i,mark in enumerate([*POINTS,'macro']):
                diff=a[i]-b[i];lo,hi=np.quantile(diff[1:],[.025,.975]);out.append(dict(tol_ms=tol,landmark=mark,seed=seed,KAN_F1=a[i,0],GELU_F1=b[i,0],difference_pp=diff[0],lower=lo,upper=hi))
    return pd.DataFrame(out)

def main(data,splits):
    torch.set_num_threads(2);out=ROOT/'results';out.mkdir(exist_ok=True)
    nets={(arm,s):read_model(arm,s) for s in [1,2,3] for arm in ['KAN','GELU']}
    subjects,manifest=load_ludb(data,splits);rows=[];refs=[]
    with torch.inference_mode():
        for rid,x,ref,span in subjects:
            refs.append(dict(record=rid,label_start_ms=int(span[0]),label_end_ms=int(span[1]),**{k:len(v) for k,v in ref.items()}))
            for (head,seed),net in nets.items():
                labels=predict_proba([net],x).argmax(0)
                for wi,cat in enumerate(['P','QRS','T'],1):
                    valid=[(i,j) for i,j in merge(runs(labels,wi)) if j-i+1 >= (15 if cat=='T' else 10)]
                    for side,name in enumerate(['on','off']):
                        mark=f'{cat}_{name}';pred=[v[side]*4 for v in valid]
                        for tol in [150,40]:
                            tp,fn,fp=score(ref[mark],pred,tol,span);rows.append(dict(record=rid,head=head,seed=seed,tol_ms=tol,landmark=mark,tp=tp,fn=fn,fp=fp))
    df=pd.DataFrame(rows);assert len(df)==41*6*2*6 and (df.assign(n=lambda t:t.tp+t.fn).groupby(['record','landmark']).n.nunique()==1).all()
    df.to_csv(out/'ludb_event_counts.csv',index=False);pd.DataFrame(refs).to_csv(out/'ludb_reference_counts.csv',index=False);summarize(df).to_csv(out/'reverse_comparisons.csv',index=False)
    (out/'ludb_manifest.json').write_text(json.dumps(dict(subject_files=manifest,split_sha256=sha(splits),analysis_code=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.txt'),torch=str(torch.__version__),wfdb=wfdb.__version__),indent=2),encoding='utf8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);ap.add_argument('splits',type=Path);a=ap.parse_args();main(a.data,a.splits)
