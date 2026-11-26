"""Prepare RDB training inputs. This program has no LUDB dependency."""
import argparse, csv, hashlib, io, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt

ROOT=Path(__file__).resolve().parent
def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(rdb,mapping):
    out=ROOT/'training';out.mkdir(exist_ok=True)
    assert digest(rdb)=='d24b8c98635bd14c358b58fc432d82db6eb830b8597cb4793e50f43fe3df04d7'
    assert digest(mapping)=='5ad900a244a0171f0b49a5e4f402311ac5d521f8564c994f696f66a64bcf7f9a'
    frame=pd.read_excel(mapping).sort_values('rdb_record_id');records=[];waves=[];masks=[];seen=set();sources=set();excluded=[]
    sos=butter(2,[.5,40],fs=250,btype='bandpass',output='sos')
    examples=[]
    with ZipFile(rdb) as z:
        for row in frame.itertuples(index=False):
            rid,src=row.rdb_record_id,row.chapman_record_id
            if src in sources:excluded.append([rid,src,'repeated source']);continue
            sources.add(src)
            raw=pd.read_csv(io.BytesIO(z.read(f'dat_csv/{rid}.csv')),header=None,usecols=[1]).iloc[:,0].to_numpy(dtype='float64')
            assert len(raw)==5000 and np.isfinite(raw).all() and np.ptp(raw)>0
            h=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest()
            if h in seen:excluded.append([rid,src,'identical lead II']);continue
            seen.add(h)
            x=sosfiltfilt(sos,resample_poly(raw,1,2));lo,hi=np.quantile(x,[.25,.75]);sc=(hi-lo)/1.349;sc=sc if sc>=1e-8 else x.std()+1e-8
            raw_ann=list(csv.DictReader(io.StringIO(z.read(f'ann_txt/{rid}.ii.txt').decode('utf-8-sig'))));assert raw_ann
            flags=np.zeros((3,2500),dtype=bool);ints=set();invalid=0;duplicate=0;censored=0
            for e in raw_ann:
                typ,on,off=[float(e[s]) for s in ['TYPE','START','END']]
                if typ not in [0,1,2] or not 0<=on<off<=4999: invalid+=1;continue
                k=int(typ)
                if (k,on,off) in ints:duplicate+=1;continue
                ints.add((k,on,off));censored+=int(on==0)+int(off==4999)
                flags[k,int(np.ceil(on/2)):int(np.floor(off/2))+1]=True
            assert ints,rid
            span_lo=min(int(np.ceil(on/2)) for k,on,off in ints);span_hi=max(int(np.floor(off/2)) for k,on,off in ints)
            label=np.full(2500,-100,dtype=np.int8);label[span_lo:span_hi+1]=0
            for k in range(3):label[flags[k]]=k+1
            conflicts=flags.sum(axis=0)>1;label[conflicts]=-100
            records.append(dict(record=rid,source=src,rhythm=row.chapman_rhythm,signal_hash=h,valid_intervals=len(ints),p_intervals=sum(k==0 for k,on,off in ints),invalid=invalid,duplicate=duplicate,censored_ends=censored,conflicting_samples=int(conflicts.sum()),ignored_samples=int((label<0).sum()),first_sample=span_lo,last_sample=span_hi))
            if len(examples)<8 and row.chapman_rhythm not in {r['rhythm'] for r in examples}:
                examples.append(dict(record=rid,rhythm=row.chapman_rhythm,first_six_native_intervals=sorted(ints,key=lambda v:v[1])[:6],native_last=max(off for _,_,off in ints),valid_250Hz_range=[span_lo,span_hi]))
            waves.append(((x-np.median(x))/sc).astype('float32'));masks.append(label)
    df=pd.DataFrame(records);assert len(df)==2398 and df.source.is_unique and df.signal_hash.is_unique
    rng=np.random.default_rng(20261125);df['partition']='train'
    for group,block in df.groupby('rhythm',sort=True):
        ids=rng.permutation(block.sort_values('record').index);df.loc[ids[:round(.20*len(ids))],'partition']='validation'
    np.save(out/'rdb_x.npy',np.stack(waves),allow_pickle=False);np.save(out/'rdb_y.npy',np.stack(masks),allow_pickle=False)
    df.to_csv(out/'rdb_split.csv',index=False);(out/'source_qc.json').write_text(json.dumps(dict(excluded=excluded,examples=examples,source_archive=digest(rdb),source_map=digest(mapping),signal_array=digest(out/'rdb_x.npy'),label_array=digest(out/'rdb_y.npy'),roster_hash=digest(out/'rdb_split.csv'),protocol=digest(ROOT/'protocol.txt')),indent=2),encoding='utf8');print(df.groupby(['rhythm','partition']).size());print(df[['invalid','censored_ends','conflicting_samples','ignored_samples']].sum())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('rdb',type=Path);ap.add_argument('mapping',type=Path);a=ap.parse_args();prepare(a.rdb,a.mapping)
