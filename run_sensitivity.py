import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_dataset
from src.faults.injection import FaultInjector
from src.cleaning.methods import DataCleaner
from src.models.train import train_and_evaluate

SEEDS      = [42, 43, 44, 45, 46]
OPERATIONS = ['imputation', 'outliers', 'label_noise', 'balance']
ALL_PERMS  = list(itertools.permutations(OPERATIONS))

LABELS = {'imputation': 'IMP', 'outliers': 'OUT',
          'label_noise': 'LNR', 'balance': 'BAL'}

# Only two datasets, RF only, two noise levels
DATASETS    = ['pima', 'credit_default']
NOISE_RATES = [0.10, 0.20]

RESULTS_DIR = ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_sensitivity():
    all_rows = []
    
    for noise_rate in NOISE_RATES:
        fault_config = {
            'label_noise': {'rate': noise_rate, 'type': 'asymmetric'},
            'missing':     {'rate': 0.10, 'mechanism': 'MAR'},
            'outliers':    {'rate': 0.05},
        }
        print(f"\n{'='*60}")
        print(f"NOISE RATE: {int(noise_rate*100)}%")
        print(f"{'='*60}")
        
        for ds_name in DATASETS:
            data = load_dataset(ds_name)
            X_all, y_all = data['X'], data['y']
            print(f"\n  Dataset: {ds_name} ({len(y_all)} samples)")
            
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all,
                    test_size=0.20,
                    random_state=seed,
                    stratify=y_all
                )
                
                injector = FaultInjector(random_state=seed)
                faulted  = injector.inject_all_faults(X_tr, y_tr, fault_config)
                X_tr_c   = faulted['X_corrupted']
                y_tr_c   = faulted['y_corrupted']
                
                for perm in ALL_PERMS:
                    order_str = '→'.join(LABELS[p] for p in perm)
                    cleaner   = DataCleaner(random_state=seed)
                    
                    try:
                        X_clean, y_clean, _ = cleaner.apply_sequence(
                            X_tr_c, y_tr_c, list(perm)
                        )
                        result = train_and_evaluate(
                            X_clean, y_clean, X_te, y_te,
                            'random_forest', random_state=seed
                        )
                    except Exception as e:
                        result = {'balanced_accuracy': np.nan,
                                  'failed': True, 'error': str(e)}
                    
                    result['dataset']      = ds_name
                    result['noise_rate']   = noise_rate
                    result['seed']         = seed
                    result['order']        = order_str
                    result['classifier']   = 'random_forest'
                    all_rows.append(result)
                
                print(f"    seed {seed} done")
    
    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / 'sensitivity_noise.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Best and Worst Orders per Dataset × Noise Rate")
    print(f"{'='*60}")
    
    df_ok = df[df['failed'] == False].copy() if 'failed' in df.columns else df.copy()
    
    for noise_rate in NOISE_RATES:
        for ds_name in DATASETS:
            sub = df_ok[(df_ok['noise_rate'] == noise_rate) &
                        (df_ok['dataset']    == ds_name)]
            if sub.empty:
                continue
            
            means = sub.groupby('order')['balanced_accuracy'].mean()
            best_order  = means.idxmax()
            worst_order = means.idxmin()
            best_val    = means.max()
            worst_val   = means.min()
            gap         = (best_val - worst_val) * 100
            osi         = (best_val - worst_val) / best_val * 100
            
            print(f"\n  {ds_name.upper()} | noise={int(noise_rate*100)}%")
            print(f"    Best  order : {best_order}  (BA={best_val:.4f})")
            print(f"    Worst order : {worst_order}  (BA={worst_val:.4f})")
            print(f"    Gap         : {gap:.2f} pp")
            print(f"    OSI         : {osi:.2f}%")
            
            # Check LNR position in best and worst
            best_lnr_pos  = best_order.split('→').index('LNR') + 1
            worst_lnr_pos = worst_order.split('→').index('LNR') + 1
            print(f"    LNR pos in best  order: {best_lnr_pos}")
            print(f"    LNR pos in worst order: {worst_lnr_pos}")
    
    print(f"\n{'='*60}")
    print("Reference (original 15% noise from all_results.csv):")
    try:
        orig = pd.read_csv(RESULTS_DIR / 'all_results.csv')
        orig = orig[orig['classifier'] == 'random_forest']
        for ds_name in DATASETS:
            sub = orig[orig['dataset'] == ds_name]
            means = sub.groupby('cleaning_order')['balanced_accuracy'].mean()
            best_val  = means.max()
            worst_val = means.min()
            gap = (best_val - worst_val) * 100
            osi = (best_val - worst_val) / best_val * 100
            best_order  = means.idxmax()
            worst_order = means.idxmin()
            best_lnr_pos  = best_order.split('→').index('LNR') + 1
            worst_lnr_pos = worst_order.split('→').index('LNR') + 1
            print(f"\n  {ds_name.upper()} | noise=15%")
            print(f"    Best  order: {best_order}  LNR pos={best_lnr_pos}")
            print(f"    Worst order: {worst_order}  LNR pos={worst_lnr_pos}")
            print(f"    Gap: {gap:.2f} pp | OSI: {osi:.2f}%")
    except Exception as e:
        print(f"  Could not load all_results.csv: {e}")


if __name__ == '__main__':
    run_sensitivity()