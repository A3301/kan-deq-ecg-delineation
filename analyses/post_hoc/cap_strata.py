# Post hoc: does DEQ accuracy on RDB lead II differ from its eight-step twin on records where the solver hit the cap?
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
R = sys.argv[1]
sc = pd.read_csv(f'{R}/deq_record_scores.csv'); dg = pd.read_csv(f'{R}/solver_diagnostics.csv')
EP = ['P_on', 'P_off', 'QRS_on', 'QRS_off', 'T_on', 'T_off']
print(sc.arm.unique(), sc.endpoint.unique(), sc.tolerance_ms.unique())
sc = sc[sc.endpoint.isin(EP)]
def f1(df):
    g = df.groupby('endpoint')[['tp', 'fp', 'fn']].sum()
    return float(np.mean([200 * r.tp / (2 * r.tp + r.fp + r.fn) if (2 * r.tp + r.fp + r.fn) > 0 else np.nan for r in g.itertuples()]))
for tol in (150, 40):
    s = sc[sc.tolerance_ms == tol]
    a = np.mean([f1(s[(s.arm == 'DEQ') & (s.seed == k)]) for k in (1, 2)]); b = np.mean([f1(s[(s.arm != 'DEQ') & (s.seed == k)]) for k in (1, 2)])
    print('overall', tol, round(a, 3), round(b, 3), round(a - b, 3))
d = dg[dg.arm == 'DEQ'].copy()
d['capped'] = (d.common_window_iterations >= 30) & (d.max_window_final_residual >= 1e-3)
nc = d.groupby(['record', 'seed']).capped.sum().rename('n_capped').reset_index()   # 0..5 folds
print('records x seeds by number of capped folds:', nc.n_capped.value_counts().sort_index().to_dict())
sc = sc.merge(nc, on=['record', 'seed'])
sc['stratum'] = np.where(sc.n_capped == 0, 'none', np.where(sc.n_capped == 5, 'all5', 'some'))
rng = np.random.default_rng(2026)
for tol in (150, 40):
    for st in ['none', 'some', 'all5']:
        s = sc[(sc.tolerance_ms == tol) & (sc.stratum == st)]
        recs = s.record.unique()
        res = []
        for k in (1, 2):
            sk = s[s.seed == k]
            res.append((f1(sk[sk.arm == 'DEQ']), f1(sk[sk.arm != 'DEQ'])))
        a, b = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])
        # paired bootstrap over records (both seeds and arms kept together)
        idx = {r: g for r, g in s.groupby('record')}
        boots = []
        for _ in range(1000):
            pick = rng.choice(recs, len(recs))
            bs = pd.concat([idx[r] for r in pick])
            boots.append(np.mean([f1(bs[(bs.arm == 'DEQ') & (bs.seed == k)]) - f1(bs[(bs.arm != 'DEQ') & (bs.seed == k)]) for k in (1, 2)]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rh = s.drop_duplicates(['record']).rhythm.value_counts().to_dict()
        print(f'tol {tol} stratum {st}: record-seed pairs {s.drop_duplicates(["record","seed"]).shape[0]}, records {len(recs)}, '
              f'DEQ {a:.3f} twin {b:.3f} diff {a-b:+.3f} [{lo:+.3f}, {hi:+.3f}] rhythms {rh}')
