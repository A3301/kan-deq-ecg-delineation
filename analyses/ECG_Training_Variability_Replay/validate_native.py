"""Independent CPU check against archived CPU trainer and reference evaluator on sampled records."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import train_panel as panel

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"source"))
import data
import evaluate
import train

def main():
    data.LUDB=HERE/"native";data.LUDB_DATA=data.LUDB/"data"
    samples=data.load_ludb()
    x,rows=panel.metadata();sp=panel.SPLITS
    selections=[(0,2),(1,3),(4,2)]
    for fold,seed in selections:
        train_ids={s for i,ids in enumerate(sp["folds"]) if i!=fold for s in ids}
        train_samples=[s for s in samples if s.subject in train_ids and s.span[1]>s.span[0]]
        base=HERE/"batch_cache"/f"f{fold}_s{seed}"
        xx=np.load(base.with_suffix(".x.npy"),mmap_mode="r")
        yy=np.load(base.with_suffix(".y.npy"),mmap_mode="r")
        for i,(xb,yb) in enumerate(train.batches(train_samples,32,np.random.default_rng(seed),3)):
            np.testing.assert_array_equal(xb[:,0].numpy(),xx[i]);np.testing.assert_array_equal(yb.numpy(),yy[i])
    test_ids=sorted(sp["test"],key=int)
    selected=[next(i for i,s in enumerate(samples) if s.subject==record and s.lead==lead) for record,lead
              in zip(test_ids[:3],["i","v1","iii"])]
    panel.DEVICE="cpu";torch.set_num_threads(2)
    max_diff={};checks=0
    for model in panel.FAMILIES:
        cp=torch.load(panel.output_path(model,0,1), weights_only=True,map_location="cpu")
        net=panel.models.build(model);net.load_state_dict(cp["state"], strict=True);net.eval()
        probs=panel.predict(net,x[selected])
        original_prob=np.stack([evaluate.predict_proba([net],samples[i].x) for i in selected])
        max_diff[model]=float(np.abs(probs-original_prob).max())
        np.testing.assert_allclose(probs,original_prob,atol=7e-5,rtol=1e-4)
        current=panel.counts_for(probs,[rows[i] for i in selected],[150,40])
        for k,tol in enumerate([150,40]):
            for q,idx in enumerate(selected):
                original=evaluate.ludb_eval([net],[samples[idx]],tol)
                for j,point in enumerate(panel.POINTS):
                    observed=current[k,q,j]
                    expected=[original.c[point][col] for col in ("tp","fn","fp")]
                    np.testing.assert_array_equal(observed,expected);checks+=1
    res=dict(sampling_checks=len(selections)*3*32, native_records=len(samples), participants=len(set(s.subject for s in samples)),
             independent_check_records=[str(samples[i].record)+"/"+samples[i].lead for i in selected],
             matched_fiducial_count_triplets=checks, orig_vs_panel_max_cpu_probability_difference=max_diff,
             probabilistic_tolerance_absolute=7e-5)
    (HERE/"validation.json").write_text(json.dumps(res,indent=2))
    print(res)

if __name__=="__main__":main()
