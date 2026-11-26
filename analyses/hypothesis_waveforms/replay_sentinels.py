"""Run the four originally selected five-fold models on the two negative-reference segments.

Requires NumPy, SciPy, WFDB and CPU PyTorch. Outputs alone suffice for reanalyse.py.
"""
from fractions import Fraction
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
# WFDB 4.3 expects NumPy-backed strings in its import-time annotation tables.
if hasattr(pd.options.future, 'infer_string'):
    pd.options.future.infer_string = False
import wfdb

from inference_definitions import CNNBiLSTM, ConvKANNet, DEQSeg, UNet1D, UNetKAN, merge, peak_of, predict_proba, runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-weights', type=Path, help='Optional directory for the plain, matched heads and full U-Net-KAN secondary ensemble weights at both seeds')
    args=parser.parse_args()
    folder = Path(__file__).resolve().parent / 'inputs'
    selections = json.loads((folder / 'chosen_lr.json').read_text())
    models = [(m,m,net,folder/'weights',f'lr{selections[m]}_{m}',1) for m,net in {'cnn_bilstm':CNNBiLSTM, 'unet_kan':UNetKAN, 'convkan':ConvKANNet, 'deq':DEQSeg}.items()]
    if args.head_weights:
        models = [(label, model, net, args.head_weights, prefix,seed) for label,model,net,prefix in [('plain','unet',UNet1D,'lr1e-3_unet'),('head_KAN','unet_kan',UNetKAN,'abl-headonly_unet_kan'),('head_GELU','unet_kan',UNetKAN,'abl-mlphead_unet_kan'),('full_KAN','unet_kan',UNetKAN,'lr3e-3_unet_kan')] for seed in [1,2]]
    torch.set_num_threads(3)
    filters = butter(2, [.5, 40], btype='bandpass', fs=250, output='sos')
    rows = []
    for label, model, architecture, directory, prefix, seed in models:
        ensemble = []
        for fold in range(5):
            weight = directory / f'{prefix}_f{fold}_s{seed}.pt'
            checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
            assert checkpoint['model'] == model
            net = architecture(**checkpoint['kw'])
            net.load_state_dict(checkpoint['state'],strict=True)
            ensemble.append(net.eval())
        for case, source in [('08','MIT-BIH Arrhythmia Database:203'),('48','Long Term AF Database:62')]:
            path = str(folder/'but_headers'/case)
            sig, meta = wfdb.rdsamp(path)
            ref = wfdb.rdann(path, 'pwave')
            assert list(ref.sample) == [-1] and ref.symbol == ['N']
            frac = Fraction(250, int(meta['fs']))
            traces = resample_poly(sig, frac.numerator, frac.denominator, axis=0)
            for j, t in enumerate(traces.T):
                filt = sosfiltfilt(filters, np.nan_to_num(t.astype(np.float64)))
                q25, q75 = np.percentile(filt,[25,75]); scale=(q75-q25)/1.349
                if scale < 1e-8:
                    scale = np.std(filt)+1e-8
                x=((filt-np.median(filt))/scale).astype(np.float32)
                with torch.inference_mode():
                    mask = predict_proba(ensemble,x).argmax(axis=0)
                peaks = [peak_of(x,i,k) for i,k in merge(runs(mask,1)) if k-i+1 >= 10]
                rows.append(dict(model=label,seed=seed,segment=case,source=source,lead=j,original_rate=meta['fs'],samples_250Hz=len(x),valid_references=0,predicted_P=len(peaks),peak_samples=' '.join(map(str,peaks))))
    filename='head_sentinel_predictions.csv' if args.head_weights else 'sentinel_predictions.csv'
    pd.DataFrame(rows).to_csv(folder/filename,index=False)


if __name__ == '__main__':
    main()
