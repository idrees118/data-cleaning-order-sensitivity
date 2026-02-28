"""
Injection-Rate Sensitivity Analysis
====================================

Tests how OSI varies with different fault injection rates.

This runs:
- 3-4 different fault rate configurations
- All 24 orders × 5 seeds × 5 datasets
- Total: ~1,800-2,400 experiments (12-16 hours compute)

Helps answer: "Are results sensitive to specific fault rates we chose?"
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sensitivity_analysis(
    dataset_name: str,
    fault_configs: list,
    classifier_name: str = 'random_forest'
):
    """
    Run experiments at multiple fault rates.
    
    Args:
        dataset_name: 'pima', 'adult', etc.
        fault_configs: List of fault configuration dicts
        classifier_name: 'random_forest', 'logistic_regression', 'xgboost'
    """
    
    from src.data.loaders import load_dataset
    from src.faults.injection import FaultInjector
    from src.cleaning.methods import DataCleaner
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    import xgboost as xgb
    
    # Load dataset
    data = load_dataset(dataset_name)
    X_clean, y_clean = data['X'], data['y']
    X_clean = X_clean.astype(float)
    
    # All orders to test
    from itertools import permutations
    orders = list(permutations(['imputation', 'label_noise', 'outliers', 'balance']))
    
    seeds = [42, 43, 44, 45, 46]
    
    all_results = []
    
    for config_idx, fault_config in enumerate(fault_configs):
        config_name = f"label_{int(fault_config['label_noise']['rate']*100)}_miss_{int(fault_config['missing']['rate']*100)}_out_{int(fault_config['outliers']['rate']*100)}"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"FAULT CONFIG {config_idx+1}/{len(fault_configs)}: {config_name}")
        logger.info(f"{'='*70}")
        
        for seed in seeds:
            logger.info(f"  Seed {seed}")
            
            # Split data
            X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
                X_clean, y_clean,
                test_size=0.2,
                random_state=seed,
                stratify=y_clean
            )
            
            for order_idx, order in enumerate(orders):
                # Inject faults
                injector = FaultInjector(random_state=seed)
                fault_result = injector.inject_all_faults(
                    X_train_clean.copy(),
                    y_train_clean.copy(),
                    fault_config
                )
                X_corrupted = fault_result['X_corrupted']
                y_corrupted = fault_result['y_corrupted']
                
                # Clean
                cleaner = DataCleaner(random_state=seed)
                cleaning_params = {
                    'imputation': {'method': 'knn', 'n_neighbors': 5},
                    'label_noise': {'method': 'confidence', 'threshold': 0.7, 'mode': 'correct'},
                    'outliers': {'method': 'isolation_forest', 'contamination': 0.05},
                    'balance': {'method': 'smote', 'target_ratio': 0.8}
                }
                
                X_cleaned, y_cleaned, _ = cleaner.apply_cleaning_sequence(
                    X_corrupted, y_corrupted, list(order), cleaning_params
                )
                
                # Train model
                if classifier_name == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        random_state=seed, n_jobs=-1
                    )
                elif classifier_name == 'logistic_regression':
                    model = LogisticRegression(
                        max_iter=1000, random_state=seed, n_jobs=-1
                    )
                elif classifier_name == 'xgboost':
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=10, learning_rate=0.1,
                        random_state=seed, n_jobs=-1, verbosity=0
                    )
                
                model.fit(X_cleaned, y_cleaned)
                y_pred = model.predict(X_test_clean)
                balanced_acc = balanced_accuracy_score(y_test_clean, y_pred)
                
                all_results.append({
                    'dataset': dataset_name,
                    'fault_config': config_name,
                    'label_noise_rate': fault_config['label_noise']['rate'],
                    'missing_rate': fault_config['missing']['rate'],
                    'outlier_rate': fault_config['outliers']['rate'],
                    'cleaning_order': '→'.join(order),
                    'classifier': classifier_name,
                    'seed': seed,
                    'balanced_accuracy': balanced_acc
                })
                
                if (order_idx + 1) % 6 == 0:
                    logger.info(f"    Completed {order_idx+1}/24 orders")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = Path('results/tables') / f'{dataset_name}_sensitivity_analysis.csv'
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Analyze
    analyze_sensitivity(results_df, dataset_name)
    
    return results_df


def analyze_sensitivity(results_df, dataset_name):
    """Analyze how OSI varies with fault rates."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SENSITIVITY ANALYSIS - {dataset_name}")
    logger.info(f"{'='*70}\n")
    
    configs = results_df['fault_config'].unique()
    
    osi_by_config = []
    
    for config in configs:
        subset = results_df[results_df['fault_config'] == config]
        
        # Compute OSI
        order_means = subset.groupby('cleaning_order')['balanced_accuracy'].mean()
        best = order_means.max()
        worst = order_means.min()
        osi = (best - worst) / best
        
        label_rate = subset['label_noise_rate'].iloc[0]
        miss_rate = subset['missing_rate'].iloc[0]
        out_rate = subset['outlier_rate'].iloc[0]
        
        osi_by_config.append({
            'config': config,
            'label_noise': f"{label_rate*100:.0f}%",
            'missing': f"{miss_rate*100:.0f}%",
            'outliers': f"{out_rate*100:.0f}%",
            'OSI': f"{osi*100:.2f}%",
            'osi_value': osi
        })
        
        logger.info(f"{config}:")
        logger.info(f"  OSI = {osi*100:.2f}%")
        logger.info(f"  Best:  {best:.4f}")
        logger.info(f"  Worst: {worst:.4f}")
        logger.info(f"  Gap:   {(best-worst)*100:.2f} pp\n")
    
    # Summary
    osi_df = pd.DataFrame(osi_by_config)
    
    print("\nOSI Variation Across Fault Rates:")
    print(osi_df[['label_noise', 'missing', 'outliers', 'OSI']].to_string(index=False))
    
    # Check if pattern is consistent
    osi_values = osi_df['osi_value'].values
    osi_std = np.std(osi_values)
    osi_mean = np.mean(osi_values)
    cv = (osi_std / osi_mean) * 100  # Coefficient of variation
    
    print(f"\nOSI Statistics:")
    print(f"  Mean: {osi_mean*100:.2f}%")
    print(f"  Std:  {osi_std*100:.2f}%")
    print(f"  CV:   {cv:.2f}%")
    
    if cv < 20:
        print(f"\n✓ OSI is STABLE across fault rates (CV < 20%)")
    else:
        print(f"\n⚠ OSI varies SUBSTANTIALLY across fault rates (CV > 20%)")


def run_full_sensitivity_study():
    """
    Run complete sensitivity analysis on all datasets.
    
    Tests 4 fault configurations:
    1. Low: 10% label, 5% missing, 3% outliers
    2. Medium-Low: 12% label, 8% missing, 4% outliers  
    3. Medium-High: 15% label, 10% missing, 5% outliers (CURRENT)
    4. High: 20% label, 15% missing, 7% outliers
    """
    
    fault_configs = [
        {  # Low
            'label_noise': {'rate': 0.10, 'type': 'asymmetric'},
            'missing': {'rate': 0.05, 'mechanism': 'MAR'},
            'outliers': {'rate': 0.03, 'method': 'extreme'}
        },
        {  # Medium-Low
            'label_noise': {'rate': 0.12, 'type': 'asymmetric'},
            'missing': {'rate': 0.08, 'mechanism': 'MAR'},
            'outliers': {'rate': 0.04, 'method': 'extreme'}
        },
        {  # Medium-High (CURRENT in paper)
            'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
            'missing': {'rate': 0.10, 'mechanism': 'MAR'},
            'outliers': {'rate': 0.05, 'method': 'extreme'}
        },
        {  # High
            'label_noise': {'rate': 0.20, 'type': 'asymmetric'},
            'missing': {'rate': 0.15, 'mechanism': 'MAR'},
            'outliers': {'rate': 0.07, 'method': 'extreme'}
        }
    ]
    
    datasets = ['pima', 'spambase']  # Start with 2 small datasets for testing
    
    logger.info(f"\n{'='*70}")
    logger.info(f"INJECTION-RATE SENSITIVITY STUDY")
    logger.info(f"{len(fault_configs)} fault configs × {len(datasets)} datasets")
    logger.info(f"{'='*70}\n")
    
    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*70}")
        
        try:
            run_sensitivity_analysis(dataset, fault_configs, 'random_forest')
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        
        # Quick test with 2 configs
        fault_configs = [
            {  # Medium-High (current)
                'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
                'missing': {'rate': 0.10, 'mechanism': 'MAR'},
                'outliers': {'rate': 0.05, 'method': 'extreme'}
            },
            {  # High
                'label_noise': {'rate': 0.20, 'type': 'asymmetric'},
                'missing': {'rate': 0.15, 'mechanism': 'MAR'},
                'outliers': {'rate': 0.07, 'method': 'extreme'}
            }
        ]
        
        run_sensitivity_analysis(dataset, fault_configs)
    else:
        # Run full study
        run_full_sensitivity_study()
