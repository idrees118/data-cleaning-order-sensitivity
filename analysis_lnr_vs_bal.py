import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

T = Path(__file__).parent / 'results' / 'tables'
SEP = '→'


def bh(p):
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    n = len(p)
    idx = np.argsort(p)
    adj = np.minimum.accumulate((p[idx] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[idx] = np.minimum(adj, 1.0)
    return out


df = pd.read_csv(T / 'all_results.csv')
df = df[df['balanced_accuracy'].notna()]

rows = []
for (ds, clf), g in df.groupby(['dataset', 'classifier']):
    acc = {(r.seed, r.cleaning_order): r.balanced_accuracy for r in g.itertuples()}
    diffs, per_seed = [], {}
    for (seed, order), v in acc.items():
        ops = order.split(SEP)
        i, j = ops.index('LNR'), ops.index('BAL')
        if i < j:
            sw = ops.copy()
            sw[i], sw[j] = 'BAL', 'LNR'
            partner = acc.get((seed, SEP.join(sw)))
            if partner is None:
                continue
            d = v - partner
            diffs.append(d)
            per_seed.setdefault(seed, []).append(d)
    if not diffs:
        continue
    diffs = np.array(diffs)
    seed_means = np.array([np.mean(x) for x in per_seed.values()])
    try:
        p_wil = stats.wilcoxon(diffs).pvalue
    except ValueError:
        p_wil = np.nan
    sd = seed_means.std(ddof=1) if len(seed_means) > 1 else np.nan
    p_t = stats.ttest_1samp(seed_means, 0).pvalue if np.isfinite(sd) and sd > 0 else np.nan
    rows.append({
        'dataset': ds, 'classifier': clf, 'n_pairs': len(diffs),
        'mean_diff_pp': 100 * diffs.mean(),
        'pct_pairs_lnr_first_better': 100 * (diffs > 0).mean(),
        'p_wilcoxon_pooled': p_wil,
        'p_ttest_seed_means': p_t,
    })

out = pd.DataFrame(rows)
out['p_ttest_bh'] = bh(out['p_ttest_seed_means'])
out.to_csv(T / 'lnr_vs_bal_paired.csv', index=False)
print('Positive mean_diff_pp = label noise correction BEFORE SMOTE is better')
print(out.round(4).to_string(index=False))

if 'smote_skipped' in df.columns:
    rf = df[df['classifier'] == 'random_forest'].copy()
    rf['smote_skipped'] = rf['smote_skipped'].astype(float)
    skip = rf.groupby(['dataset', 'cleaning_order'])['smote_skipped'].mean().unstack('cleaning_order')
    skip.to_csv(T / 'smote_skip_by_order.csv')
    print('\nShare of runs where SMOTE was skipped, saved to smote_skip_by_order.csv')
    print(skip.mean(axis=1).round(3).to_string())
