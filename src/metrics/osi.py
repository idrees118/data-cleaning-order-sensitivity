import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import wilcoxon as _wilcoxon


def compute_osi(values_by_order):
    means = {o: np.mean(v) for o, v in values_by_order.items()}
    best = max(means.values())
    worst = min(means.values())
    return (best - worst) / best if best > 0 else 0.0


def compute_osi_from_df(df, order_col='cleaning_order', metric_col='balanced_accuracy'):
    df = df[~df['failed']].copy() if 'failed' in df.columns else df.copy()
    grouped = df.groupby(order_col)[metric_col].agg(list).to_dict()
    means = {o: np.mean(v) for o, v in grouped.items()}
    best_order = max(means, key=means.get)
    worst_order = min(means, key=means.get)
    best_mean = means[best_order]
    worst_mean = means[worst_order]
    osi = (best_mean - worst_mean) / best_mean if best_mean > 0 else 0.0
    return {'osi': osi, 'osi_pct': osi * 100, 'best_order': best_order,
            'best_mean': best_mean, 'worst_order': worst_order,
            'worst_mean': worst_mean, 'gap_pp': (best_mean - worst_mean) * 100}


def paired_ttest(a, b):
    n = min(len(a), len(b))
    a, b = np.array(a[:n]), np.array(b[:n])
    diffs = a - b
    t, p = stats.ttest_rel(a, b)
    d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0
    return {'t': t, 'p': p, 'cohens_d': d, 'mean_diff': np.mean(diffs)}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    greater = sum(1 for x in a for y in b if x > y)
    less    = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (len(a) * len(b))


def benjamini_hochberg(p_values, alpha=0.05):
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adjusted = np.minimum(1.0, sorted_p * n / (np.arange(1, n + 1)))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


def bootstrap_osi_ci(values_by_order, n_boot=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    orders = list(values_by_order.keys())
    n_seeds = len(next(iter(values_by_order.values())))
    boot_osi = []
    for _ in range(n_boot):
        idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        resampled = {o: np.array(values_by_order[o])[idx] for o in orders}
        boot_osi.append(compute_osi(resampled))
    lo = (1 - ci) / 2
    return float(np.percentile(boot_osi, lo * 100)), float(np.percentile(boot_osi, (1 - lo) * 100))


def significance_report(df, dataset, classifier,
                         order_col='cleaning_order', metric_col='balanced_accuracy'):
    subset = df[(df['dataset'] == dataset) & (df['classifier'] == classifier)].copy()
    if 'failed' in subset.columns:
        subset = subset[~subset['failed']]
    if len(subset) == 0:
        return {'dataset': dataset, 'classifier': classifier, 'error': 'no_data'}

    grouped = subset.groupby(order_col)[metric_col].agg(list).to_dict()
    means = {o: np.mean(v) for o, v in grouped.items()}
    best_order  = max(means, key=means.get)
    worst_order = min(means, key=means.get)
    best_vals  = np.array(grouped[best_order])
    worst_vals = np.array(grouped[worst_order])

    osi_summary = compute_osi_from_df(subset, order_col, metric_col)
    ttest       = paired_ttest(best_vals, worst_vals)
    delta       = cliffs_delta(best_vals, worst_vals)
    ci_lo, ci_hi = bootstrap_osi_ci(grouped)

    try:
        _, w_p = _wilcoxon(best_vals, worst_vals)
    except Exception:
        w_p = float('nan')

    return {
        'dataset':      dataset,
        'classifier':   classifier,
        'osi_pct':      osi_summary['osi_pct'],
        'best_order':   best_order,
        'best_mean':    means[best_order],
        'worst_order':  worst_order,
        'worst_mean':   means[worst_order],
        'gap_pp':       osi_summary['gap_pp'],
        'p_value':      ttest['p'],
        'cohens_d':     ttest['cohens_d'],
        'cliffs_delta': delta,
        'wilcoxon_p':   w_p,
        'osi_ci_lower': ci_lo * 100,
        'osi_ci_upper': ci_hi * 100,
        'n_seeds':      len(best_vals),
    }
