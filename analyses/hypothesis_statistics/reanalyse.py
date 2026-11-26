"""Reanalyse saved predictions. Run `python reanalyse.py [--extract-counts]`.

Requires numpy, pandas, scipy, scikit-learn. No model code or weights are loaded.
The optional extraction accepts only primitive-object tables from the specified archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
MODELS = ['cnn_bilstm', 'unet_kan', 'convkan', 'deq']
N_BOOT = 10_000


class PrimitivesOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('No executable pickle globals are permitted.')


def extract_counts():
    allowed = set('APPEND APPENDS BINFLOAT BINGET BININT1 BININT2 EMPTY_DICT EMPTY_LIST FRAME LONG_BINGET MARK MEMOIZE PROTO SETITEM SETITEMS SHORT_BINUNICODE STOP'.split())
    rows = []
    for model in MODELS:
        data = (INPUT / f'{model}_records.pkl').read_bytes()
        if set(op.name for op, _, _ in pickletools.genops(data)) - allowed:
            raise ValueError('Unexpected pickle opcode')
        subject = PrimitivesOnly(io.BytesIO(data)).load()['butpdb_each_lead']['subj']['P_peak']
        rows += [{'model': model, 'source': key, **{c: v[c] for c in ['tp', 'fn', 'fp']}} for key, v in sorted(subject.items())]
    pd.DataFrame(rows).to_csv(INPUT / 'but_pooled_counts.csv', index=False)


def pct_f1(c):
    t, n, p = np.moveaxis(c, -1, 0)
    den = 2 * t + n + p
    return np.divide(200.0 * t, den, out=np.full(np.shape(den), np.nan), where=den > 0)


def holm(ps):
    order = np.argsort(ps)
    out = np.zeros(len(ps))
    out[order] = np.minimum(1, np.maximum.accumulate(np.array(ps)[order] * np.arange(len(ps), 0, -1)))
    return out


def h3():
    ref = json.loads((INPUT / 'primary_family.json').read_text())
    plan = (INPUT / 'analiz_plani_v1.0_frozen.md').read_bytes()
    assert hashlib.sha256(plan).hexdigest() == (INPUT / 'analiz_plani.sha256').read_text().split()[0]
    assert max(ref['cv_scores'], key=ref['cv_scores'].get) == ref['best_baseline'] == MODELS[0]
    data = pd.read_csv(INPUT / 'but_pooled_counts.csv')
    arr = {m: g.set_index('source')[['tp', 'fn', 'fp']].sort_index() for m, g in data.groupby('model')}
    keys = arr[MODELS[0]].index
    assert len(keys) == 45 and all(tbl.index.equals(keys) for tbl in arr.values())
    excluded = ['MIT-BIH Arrhythmia Database:203', 'Long Term AF Database:62']
    for m in MODELS:
        sentinel = arr[m].loc[excluded]
        assert ((sentinel.tp + sentinel.fn) == 2).all() and (sentinel.fp == 0).all()
    corrected={m:c.copy() for m,c in arr.items()}
    cnt=pd.read_csv(INPUT/'sentinel_predictions.csv')
    assert len(cnt)==16 and (cnt.valid_references==0).all()
    for (model, source), group in cnt.groupby(['model','source']):
        assert len(group)==2
        corrected[model].loc[source]=[0,0,group.predicted_P.sum()]
    rows = []
    for cohort, included, tables in [('full_corrected', keys, corrected), ('as_run', keys, arr), ('valid_P_segments', keys.drop(excluded), arr)]:
        ix = np.random.default_rng(0).integers(0, len(included), (N_BOOT, len(included)))
        cohort_rows = []
        for m in MODELS[1:]:
            a, b = (tables[x].loc[included].values for x in [m, MODELS[0]])
            fa, fb = pct_f1(a.sum(0)), pct_f1(b.sum(0))
            d = pct_f1(a[ix].sum(1)) - pct_f1(b[ix].sum(1))
            wa, wb = pct_f1(a), pct_f1(b)
            ok = np.isfinite(wa) & np.isfinite(wb)
            ps = min(1, 2 * min(np.mean(d>=0), np.mean(d<=0)))
            lo, hi = np.quantile(d, [.025, .975]); ta, na, pa = a.sum(0); tb, nb, pb = b.sum(0)
            cohort_rows.append(dict(cohort=cohort, model=m, comparator=MODELS[0], segments=48 if cohort=='valid_P_segments' else 50, source_records=len(included), tp=ta,fn=na,fp=pa,baseline_tp=tb,baseline_fn=nb,baseline_fp=pb,model_F1=fa,baseline_F1=fb,delta_pp=fa-fb,CI_low=lo,CI_high=hi,p_boot=ps,p_wilcoxon=wilcoxon(wa[ok], wb[ok]).pvalue))
        for row, p in zip(cohort_rows, holm([r['p_boot'] for r in cohort_rows])):
            row['p_holm']=p
        if cohort == 'as_run':
            for row, original in zip(cohort_rows, ref['tests']):
                np.testing.assert_allclose([row['model_F1'],row['baseline_F1'],row['CI_low'],row['CI_high'],row['p_holm']], [original['F1_A'],original['F1_B'],*original['diff_CI95'],original['p_holm']], atol=1e-8)
        rows.extend(cohort_rows)
    pd.DataFrame(rows).to_csv(ROOT / 'h3_results.csv', index=False)


def midpoint_counts(train, external):
    ordered = np.sort(train)
    # Twice the midrank ECDF numerator. Integer units preserve exact ties in the sum.
    return np.searchsorted(ordered, external, side='left') + np.searchsorted(ordered, external, side='right')


def auc_from_weights(ties, y, weights):
    positives = np.bincount(ties, weights=weights*y, minlength=ties.max()+1)
    negatives = np.bincount(ties, weights=weights*(1-y), minlength=ties.max()+1)
    denominator = positives.sum() * negatives.sum()
    if denominator == 0:
        return np.nan
    return np.dot(positives, np.cumsum(negatives)-0.5*negatives) / denominator


def difficulty():
    internal = pd.read_csv(INPUT/'deq_windows.csv', dtype={'record':str}).query("db == 'LUDB'").copy()
    external = pd.read_csv(INPUT/'deq_external_windows.csv', dtype={'record':str})
    assert len(internal) == 492 and internal.record.nunique() == 41
    assert not external[['db','record','lead']].duplicated().any() and not internal[['record','lead']].duplicated().any()
    assert internal[['entropy','iters']].notna().all().all() and external[['entropy','iters','n_missed_false']].notna().all().all()
    err = internal.mae_ms.quantile(.75); missed = internal.n_missed_false.quantile(.75)
    agg=internal.groupby('record').agg(iters=('iters','mean'), mae_ms=('mae_ms','mean'), hf_frac=('hf_frac','mean'), group=('group','first'))
    assert all(internal.groupby('record').group.nunique()==1)
    h2=[]
    for col in ['mae_ms','hf_frac']:
        value, p=spearmanr(agg.iters,agg[col],alternative='greater')
        h2.append(dict(indicator=col,n=len(agg),statistic='Spearman_rho',value=value,p_greater=p))
    path,sinus=agg.loc[agg.group!='sinus','iters'],agg.loc[agg.group=='sinus','iters']
    value,p=mannwhitneyu(path,sinus,alternative='greater')
    h2.append(dict(indicator='non_sinus_vs_sinus',n=len(agg),statistic='Mann_Whitney_U',value=value,p_greater=p,n_non_sinus=len(path),n_sinus=len(sinus),median_non_sinus=path.median(),median_sinus=sinus.median()))
    pd.DataFrame(h2).to_csv(ROOT/'h2c_record_results.csv',index=False)
    output, transfers = [], []
    for db, all_rows in external.groupby('db',sort=False):
        cohort = all_rows.dropna(subset='mae_ms').reset_index(drop=True)
        _, rec = np.unique(cohort.record, return_inverse=True)
        scores = {k:midpoint_counts(internal[k].values,cohort[k].values) for k in ['iters','entropy']}
        ranks = scores['iters'] + scores['entropy']
        _, t_entropy = np.unique(cohort.entropy,return_inverse=True)
        _, t_score = np.unique(ranks,return_inverse=True)
        _, t_mid_entropy = np.unique(scores['entropy'],return_inverse=True)
        for key in scores:
            transfers.append(dict(database=db, feature=key, internal_min=internal[key].min(),internal_max=internal[key].max(), external_n=len(scores[key]), below_internal=(cohort[key]<internal[key].min()).sum(),above_internal=(cohort[key]>internal[key].max()).sum(),internal_n=len(internal)))
        local = rankdata(cohort.iters)+rankdata(cohort.entropy)
        for label, mask in [('matched_MAE',cohort.mae_ms >= err),('matched_MAE_or_unmatched',(cohort.mae_ms>=err)|(cohort.n_missed_false>missed))]:
            y = mask.astype(float).values
            e = roc_auc_score(y,cohort.entropy); auc = roc_auc_score(y,ranks); mid = roc_auc_score(y,scores['entropy'])
            rng = np.random.default_rng(25); nrec=int(max(rec))+1
            differences = np.empty((N_BOOT,2))
            for i in range(N_BOOT):
                sampled = np.bincount(rng.integers(0,nrec,nrec),minlength=nrec)[rec]
                aa = auc_from_weights(t_score,y,sampled)
                differences[i] = [aa-auc_from_weights(t_entropy,y,sampled), aa-auc_from_weights(t_mid_entropy,y,sampled)]
            assert np.all(np.isfinite(differences))
            lo, hi = np.quantile(differences,[.025,.975],axis=0)
            np.testing.assert_allclose(auc_from_weights(t_score,y,np.ones(len(y))),auc,atol=1e-12)
            output.append(dict(database=db,endpoint=label,threshold_mae_ms=err,threshold_unmatched_strictly_greater=missed,n_initial=len(all_rows),n_evaluable=len(y),n_records=nrec,n_high=int(y.sum()),p_high=y.mean(),raw_entropy_auc=e,raw_iters_auc=roc_auc_score(y,cohort.iters),cohort_rank_sum_auc=roc_auc_score(y,local),fixed_ludb_rank_sum_auc=auc,delta_vs_raw_entropy=auc-e,low_vs_raw_entropy=lo[0],high_vs_raw_entropy=hi[0],fixed_entropy_ecdf_auc=mid,delta_vs_fixed_entropy=auc-mid,low_vs_fixed_entropy=lo[1],high_vs_fixed_entropy=hi[1]))
    pd.DataFrame(output).to_csv(ROOT/'deq_external_results.csv',index=False)
    pd.DataFrame(transfers).to_csv(ROOT/'deq_range_support.csv',index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-counts',action='store_true')
    args=parser.parse_args()
    if args.extract_counts:
        extract_counts()
    h3()
    difficulty()
