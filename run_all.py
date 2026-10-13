import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── project root on path ──────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_dataset, LOADERS
from src.faults.injection import FaultInjector
from src.cleaning.methods import DataCleaner
from src.models.train import train_and_evaluate
from src.metrics.osi import (compute_osi_from_df, significance_report,
                              benjamini_hochberg, bootstrap_osi_ci)

# ── experiment configuration ──────────────────────────────────────────
SEEDS       = [42, 43, 44, 45, 46]
CLASSIFIERS = ['random_forest', 'logistic_regression', 'xgboost']
OPERATIONS  = ['imputation', 'outliers', 'label_noise', 'balance']
ALL_PERMS   = list(itertools.permutations(OPERATIONS))   # 24 orderings

FAULT_CONFIG = {
    'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
    'missing':     {'rate': 0.10, 'mechanism': 'MAR'},
    'outliers':    {'rate': 0.05},
}

RESULTS_DIR = ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Core single-run function
# ─────────────────────────────────────────────────────────────────────

def run_single(X_train_corrupt: np.ndarray,
               y_train_corrupt: np.ndarray,
               X_test: np.ndarray,
               y_test: np.ndarray,
               permutation: tuple,
               classifier: str,
               seed: int) -> dict:
    """
    Apply one permutation of cleaning operations to training data,
    train classifier, evaluate on pristine test set.
    """
    cleaner = DataCleaner(random_state=seed)
    order_str = '→'.join(p[:3].upper() if p == 'imputation' else
                          p[:3].upper() if p == 'outliers' else
                          'LNR' if p == 'label_noise' else 'BAL'
                          for p in permutation)
    # Human-readable label: IMP, OUT, LNR, BAL
    labels = {'imputation': 'IMP', 'outliers': 'OUT',
               'label_noise': 'LNR', 'balance': 'BAL'}
    order_str = '→'.join(labels[p] for p in permutation)

    try:
        X_clean, y_clean, step_log = cleaner.apply_sequence(
            X_train_corrupt, y_train_corrupt, list(permutation)
        )
        result = train_and_evaluate(X_clean, y_clean, X_test, y_test,
                                    classifier, random_state=seed)
        bal = next(s for s in step_log if s['operation'] == 'balance')
        lnr = next(s for s in step_log if s['operation'] == 'label_noise')
        result['smote_skipped'] = bool(bal.get('skipped', False))
        result['smote_skip_reason'] = bal.get('reason', '')
        result['n_relabelled'] = lnr.get('n_relabelled', 0)
    except Exception as e:
        result = {'balanced_accuracy': np.nan, 'n_train': 0,
                  'n_test': len(y_test), 'failed': True, 'error': str(e)}

    result['cleaning_order'] = order_str
    result['permutation']    = str(permutation)
    result['classifier']     = classifier
    result['seed']           = seed
    return result


# ─────────────────────────────────────────────────────────────────────
# Dataset runner
# ─────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str,
                classifiers: list = CLASSIFIERS,
                seeds: list = SEEDS,
                permutations: list = ALL_PERMS,
                verbose: bool = True) -> pd.DataFrame:
    """
    Run full permutation study for one dataset.
    Returns DataFrame with all results.
    """
    data = load_dataset(dataset_name)
    X_all, y_all = data['X'], data['y']

    if verbose:
        classes, counts = np.unique(y_all, return_counts=True)
        ratio = counts.max() / counts.min()
        print(f"\n{'='*60}")
        print(f"Dataset : {dataset_name}  ({len(y_all)} samples, "
              f"{X_all.shape[1]} features, IR={ratio:.2f})")
        print(f"Runs    : {len(permutations)} perms × {len(seeds)} seeds "
              f"× {len(classifiers)} classifiers "
              f"= {len(permutations)*len(seeds)*len(classifiers)}")
        print('='*60)

    all_results = []
    total = len(seeds) * len(permutations) * len(classifiers)
    done  = 0

    for seed in seeds:
        # ── fixed train/test split per seed (test is PRISTINE) ──────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all,
            test_size=0.20,
            random_state=seed,
            stratify=y_all
        )

        # ── inject faults into training set only ────────────────────
        injector = FaultInjector(random_state=seed)
        faulted  = injector.inject_all_faults(X_tr, y_tr, FAULT_CONFIG)
        X_tr_c   = faulted['X_corrupted']
        y_tr_c   = faulted['y_corrupted']

        for perm in permutations:
            for clf in classifiers:
                row = run_single(X_tr_c, y_tr_c, X_te, y_te, perm, clf, seed)
                row['dataset'] = dataset_name
                all_results.append(row)
                done += 1
                if verbose and done % 50 == 0:
                    pct = done / total * 100
                    print(f"  {done}/{total} ({pct:.0f}%) — last: "
                          f"{row['cleaning_order']} | {clf} | "
                          f"BA={row['balanced_accuracy']:.4f}")

    df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / f"{dataset_name}_results.csv"
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────

def build_summary_tables(all_results: pd.DataFrame) -> None:
    """
    Build Table 4 (RF summary), Table 5 (OSI by classifier),
    Table 3 (significance), and Cliff's δ table.
    Applies BH correction across all dataset–classifier pairs.
    """
    datasets    = all_results['dataset'].unique()
    classifiers = all_results['classifier'].unique()

    # ── per-dataset × classifier significance reports ────────────────
    reports = []
    for ds in datasets:
        for clf in classifiers:
            rep = significance_report(all_results, ds, clf)
            reports.append(rep)

    rep_df = pd.DataFrame(reports)

    # BH correction across all comparisons
    valid = rep_df['p_value'].notna()
    p_raw = rep_df.loc[valid, 'p_value'].tolist()
    p_adj = benjamini_hochberg(p_raw)
    rep_df.loc[valid, 'p_value_bh'] = p_adj
    rep_df.loc[valid, 'significant_bh'] = rep_df.loc[valid, 'p_value_bh'] < 0.05

    rep_df.to_csv(RESULTS_DIR / 'significance_all.csv', index=False)

    # ── Table 5 style: OSI by dataset and classifier ─────────────────
    pivot = rep_df.pivot_table(index='dataset', columns='classifier',
                                values='osi_pct', aggfunc='first')
    pivot.to_csv(RESULTS_DIR / 'osi_by_classifier.csv')

    # ── Table 4 style: RF summary ────────────────────────────────────
    rf_df = rep_df[rep_df['classifier'] == 'random_forest'][
        ['dataset', 'osi_pct', 'best_mean', 'worst_mean', 'gap_pp',
         'p_value', 'p_value_bh', 'cohens_d', 'cliffs_delta',
         'osi_ci_lower', 'osi_ci_upper']
    ].copy()
    rf_df.to_csv(RESULTS_DIR / 'rf_summary.csv', index=False)

    print("\n" + "="*60)
    print("RESULTS SUMMARY (Random Forest)")
    print("="*60)
    print(rf_df[['dataset', 'osi_pct', 'best_mean', 'worst_mean',
                  'gap_pp', 'p_value_bh', 'cliffs_delta']].to_string(index=False))

    print("\n" + "="*60)
    print("OSI (%) BY CLASSIFIER")
    print("="*60)
    print(pivot.to_string())


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Permutation study runner')
    parser.add_argument('--dataset',  type=str, default=None,
                        help='Single dataset to run (default: all)')
    parser.add_argument('--fast',     action='store_true',
                        help='Smoke test: RF only, 2 seeds, 6 permutations')
    parser.add_argument('--no-xgb',  action='store_true',
                        help='Skip XGBoost (saves ~40% time)')
    args = parser.parse_args()

    datasets    = [args.dataset] if args.dataset else list(LOADERS.keys())
    classifiers = CLASSIFIERS
    seeds       = SEEDS
    perms       = ALL_PERMS

    if args.fast:
        classifiers = ['random_forest']
        seeds       = [42, 43]
        perms       = ALL_PERMS[:6]
        print("FAST MODE: RF only, 2 seeds, 6 permutations")

    if args.no_xgb and 'xgboost' in classifiers:
        classifiers = [c for c in classifiers if c != 'xgboost']

    print(f"\nRunning: {len(datasets)} datasets × {len(perms)} perms × "
          f"{len(seeds)} seeds × {len(classifiers)} classifiers")
    print(f"Total runs: {len(datasets)*len(perms)*len(seeds)*len(classifiers)}\n")

    all_dfs = []
    for ds in datasets:
        df = run_dataset(ds, classifiers=classifiers,
                         seeds=seeds, permutations=perms)
        all_dfs.append(df)

    all_results = pd.concat(all_dfs, ignore_index=True)
    all_results.to_csv(RESULTS_DIR / 'all_results.csv', index=False)

    if not args.fast:
        build_summary_tables(all_results)

    print(f"\nDone. Total rows: {len(all_results)}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()