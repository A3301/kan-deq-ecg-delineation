"""Independent checks of numerical summaries, representative interval masks and waveform overlap."""
import argparse, csv, hashlib, io, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from test_ludb import load_ludb

ROOT=Path(__file__).resolve().parent
def main(rdb,ludb,split):
    train=ROOT/'training';results=ROOT/'results';a=pd.read_csv(train/'rdb_split.csv');y=np.load(train/'rdb_y.npy',mmap_mode='r');x=np.load(train/'rdb_x.npy',mmap_mode='r')
    # Map original intervals directly in the native sample coordinate; this is
    # independent of the integer ceil/floor slice construction in preparation.
    inspect=['AF0001','AT0003','AF0043','AF0053',a.loc[a.rhythm.eq('SVT'),'record'].iloc[-1]];examples=[]
    with ZipFile(rdb) as z:
        for rid in inspect:
            pos=a.index[a.record==rid][0]; native=np.arange(2500)*2
            raw=np.array([[float(r[k]) for k in ['TYPE','START','END']] for r in csv.DictReader(io.StringIO(z.read(f'ann_txt/{rid}.ii.txt').decode('utf8')))])
            intervals=raw[(raw[:,1]<raw[:,2])&(raw[:,1]>=0)&(raw[:,2]<5000)&np.isin(raw[:,0],[0,1,2])];inside=(native>=intervals[:,1,None])&(native<=intervals[:,2,None]);wave=np.array([inside[intervals[:,0]==k].any(0) for k in range(3)])
            test=np.where((native>=intervals[:,1].min())&(native<=intervals[:,2].max()),0,-100)
            for k in range(3):test[wave[k]]=k+1
            test[wave.sum(0)>1]=-100;np.testing.assert_array_equal(test,y[pos]);examples.append(dict(record=rid,rhythm=a.loc[pos,'rhythm'],raw_rows=len(raw),valid_intervals=len(intervals),fractional_native_coordinate_count=int(((intervals[:,1:]%1)!=0).sum()),covered_at_start=int(test[0]),covered_at_end=int(test[-1]),first_valid_interval=sorted(intervals.tolist(),key=lambda v:v[1])[0],last_valid_interval=sorted(intervals.tolist(),key=lambda v:v[2])[-1],counts_ignored_bg_P_QRS_T=[int((test==cls).sum()) for cls in [-100,0,1,2,3]]))
    subjects,_=load_ludb(ludb,split)
    def fingerprint(v):return hashlib.sha256(np.round(v,4).astype('<f4').tobytes()).hexdigest()
    rkeys={fingerprint(sig) for sig in x};overlap=[rid for rid,sig,_,_ in subjects if fingerprint(sig) in rkeys];assert not overlap
    counts=pd.read_csv(results/'ludb_event_counts.csv');rawsum=counts.groupby(['tol_ms','head','seed','landmark'])[['tp','fn','fp']].sum();rawsum['f']=200*rawsum.tp/(2*rawsum.tp+rawsum.fn+rawsum.fp);averages=rawsum.f.groupby(['tol_ms','head']).mean();report=pd.read_csv(results/'reverse_comparisons.csv').query("landmark=='macro' and seed=='mean'")
    for row in report.itertuples(index=False):np.testing.assert_allclose([row.KAN_F1,row.GELU_F1],[averages.loc[row.tol_ms,'KAN'],averages.loc[row.tol_ms,'GELU']],atol=1e-12)
    checks=[]
    for s in [1,2,3]:
        meta=[json.loads((ROOT/'models'/f'{h}_{s}.json').read_text()) for h in ['KAN','GELU']];assert meta[0]['backbone_initial_sha256']==meta[1]['backbone_initial_sha256'] and meta[0]['minibatch_sha256']==meta[1]['minibatch_sha256']
        for m in meta:
            assert min(m['checkpoints'],key=lambda x:x['validation_CE'])['step']==m['best_step'] and m['steps']==2400
            checks.append({k:m[k] for k in ['seed','head','best_step','best_validation_CE','parameters','minibatch_sha256','backbone_initial_sha256','weight_sha256']})
    pd.DataFrame(checks).to_csv(results/'selected_checkpoints.csv',index=False)
    (results/'independent_checks.json').write_text(json.dumps(dict(source_mask_spot_checks=examples,source_count=len(rkeys),external_count=len(subjects),exact_rounded_waveform_matches=overlap,checked_paired_initialisation_and_crops=True,checked_minimum_validation_CE_all_six=True,checked_event_summaries=True),indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rdb',type=Path);p.add_argument('ludb',type=Path);p.add_argument('split',type=Path);a=p.parse_args();main(a.rdb,a.ludb,a.split)
