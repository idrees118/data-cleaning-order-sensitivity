"""
Run Additional Classifiers - FIXED VERSION
===========================================

Fixed: Converts integer columns to float before fault injection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_additional_classifier_experiments(
    dataset_name: str,
    classifier_name: str = 'logistic_regression'
):
    """
    Run experiments with a different classifier on existing cleaned data.
    
    Args:
        dataset_name: 'pima', 'adult', etc.
        classifier_name: 'logistic_regression' or 'lightgbm'
    """
    
    # Load existing results to get cleaned data info
    results_dir = Path('results/tables')
    results_file = list(results_dir.glob(f'{dataset_name}_scientific_results_*.csv'))[0]
    
    logger.info(f"Loading existing results from: {results_file}")
    existing_results = pd.read_csv(results_file)
    
    # Check what orders and seeds were run
    orders = existing_results['cleaning_order'].unique()
    seeds = existing_results['seed'].unique()
    
    logger.info(f"Found {len(orders)} orders and {len(seeds)} seeds")
    logger.info(f"Will run {classifier_name} on all combinations")
    
    # Load clean dataset
    from src.data.loaders import load_dataset
    data = load_dataset(dataset_name)
    X_clean, y_clean = data['X'], data['y']
    
    # CRITICAL FIX: Convert to float to allow NaN injection
    X_clean = X_clean.astype(float)
    
    new_results = []
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'='*60}")
        
        # Create same train/test split as original
        from sklearn.model_selection import train_test_split
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean,
            test_size=0.2,
            random_state=seed,
            stratify=y_clean
        )
        
        # For each order
        for order_idx, order_str in enumerate(orders):
            logger.info(f"Order {order_idx+1}/{len(orders)}: {order_str}")
            
            # Inject faults (same as original)
            from src.faults.injection import FaultInjector
            injector = FaultInjector(random_state=seed)
            
            # Make copies and ensure float type
            X_train_copy = X_train_clean.copy().astype(float)
            y_train_copy = y_train_clean.copy()
            
            fault_result = injector.inject_all_faults(
                X_train_copy,
                y_train_copy,
                {
                    'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
                    'missing': {'rate': 0.10, 'mechanism': 'MAR'},
                    'outliers': {'rate': 0.05, 'method': 'extreme'}
                }
            )
            X_train_corrupted = fault_result['X_corrupted']
            y_train_corrupted = fault_result['y_corrupted']
            
            # Apply cleaning in specified order
            from src.cleaning.methods import DataCleaner
            cleaner = DataCleaner(random_state=seed)
            order_list = order_str.split('→')
            
            cleaning_params = {
                'imputation': {'method': 'knn', 'n_neighbors': 5},
                'label_noise': {'method': 'confidence', 'threshold': 0.7, 'mode': 'correct'},
                'outliers': {'method': 'isolation_forest', 'contamination': 0.05},
                'balance': {'method': 'smote', 'target_ratio': 0.8}
            }
            
            X_train_cleaned, y_train_cleaned, _ = cleaner.apply_cleaning_sequence(
                X_train_corrupted,
                y_train_corrupted,
                order_list,
                cleaning_params
            )
            
            # Train NEW classifier
            if classifier_name == 'logistic_regression':
                model = LogisticRegression(
                    random_state=seed,
                    max_iter=1000,
                    n_jobs=-1
                )
            elif classifier_name == 'lightgbm':
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                raise ValueError(f"Unknown classifier: {classifier_name}")
            
            # Train and evaluate
            model.fit(X_train_cleaned, y_train_cleaned)
            y_pred = model.predict(X_test_clean)
            balanced_acc = balanced_accuracy_score(y_test_clean, y_pred)
            
            # Record result
            new_results.append({
                'dataset': dataset_name,
                'cleaning_order': order_str,
                'classifier': classifier_name,
                'seed': seed,
                'balanced_accuracy': balanced_acc
            })
            
            logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    # Save results
    results_df = pd.DataFrame(new_results)
    output_file = results_dir / f'{dataset_name}_{classifier_name}_results.csv'
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}")
    
    # Compute OSI
    compute_osi_for_classifier(results_df, dataset_name, classifier_name)
    
    return results_df


def compute_osi_for_classifier(results_df, dataset_name, classifier_name):
    """Compute OSI for the new classifier."""
    
    # Group by order and compute mean
    order_means = results_df.groupby('cleaning_order')['balanced_accuracy'].mean()
    
    best_order = order_means.idxmax()
    worst_order = order_means.idxmin()
    
    best_perf = order_means.max()
    worst_perf = order_means.min()
    
    osi = (best_perf - worst_perf) / best_perf
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OSI ANALYSIS - {classifier_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Best order:  {best_order} ({best_perf:.4f})")
    logger.info(f"Worst order: {worst_order} ({worst_perf:.4f})")
    logger.info(f"OSI: {osi:.4f} ({osi*100:.2f}%)")
    logger.info(f"Gap: {(best_perf - worst_perf):.4f} ({(best_perf - worst_perf)*100:.2f} pp)")
    
    # Compare to Random Forest
    rf_file = Path('results/tables') / f'{dataset_name}_scientific_results_*.csv'
    rf_file = list(Path('results/tables').glob(f'{dataset_name}_scientific_results_*.csv'))[0]
    rf_results = pd.read_csv(rf_file)
    
    rf_means = rf_results.groupby('cleaning_order')['balanced_accuracy'].mean()
    rf_best = rf_means.max()
    rf_worst = rf_means.min()
    rf_osi = (rf_best - rf_worst) / rf_best
    
    logger.info(f"\nComparison to Random Forest:")
    logger.info(f"  RF OSI:  {rf_osi*100:.2f}%")
    logger.info(f"  {classifier_name} OSI: {osi*100:.2f}%")
    logger.info(f"  Ratio: {osi/rf_osi:.2f}×")
    
    if osi > 0.05:
        logger.info(f"\n✓ Effect holds for {classifier_name}!")
    else:
        logger.info(f"\n⚠ Effect weaker for {classifier_name}")


def run_all_datasets_with_new_classifier(classifier_name='logistic_regression'):
    """Run all 5 datasets with new classifier."""
    
    datasets = ['pima', 'adult', 'credit_default', 'spambase', 'breast_cancer']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING ALL DATASETS WITH {classifier_name.upper()}")
    logger.info(f"{'='*70}\n")
    
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*70}\n")
        
        try:
            results = run_additional_classifier_experiments(dataset, classifier_name)
            all_results[dataset] = results
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary table
    create_classifier_comparison_table(all_results, classifier_name)


def create_classifier_comparison_table(all_results, classifier_name):
    """Create table comparing RF vs new classifier."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY: Random Forest vs {classifier_name}")
    logger.info(f"{'='*70}\n")
    
    summary_data = []
    
    for dataset, results in all_results.items():
        # Compute OSI for new classifier
        order_means = results.groupby('cleaning_order')['balanced_accuracy'].mean()
        best = order_means.max()
        worst = order_means.min()
        osi_new = (best - worst) / best
        
        # Load RF OSI
        rf_file = list(Path('results/tables').glob(f'{dataset}_scientific_results_*.csv'))[0]
        rf_results = pd.read_csv(rf_file)
        rf_means = rf_results.groupby('cleaning_order')['balanced_accuracy'].mean()
        rf_best = rf_means.max()
        rf_worst = rf_means.min()
        osi_rf = (rf_best - rf_worst) / rf_best
        
        summary_data.append({
            'Dataset': dataset,
            'RF_OSI': f"{osi_rf*100:.2f}%",
            f'{classifier_name}_OSI': f"{osi_new*100:.2f}%",
            'Ratio': f"{osi_new/osi_rf:.2f}×"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save
    output_file = Path('results/tables') / f'classifier_comparison_{classifier_name}.csv'
    summary_df.to_csv(output_file, index=False)
    logger.info(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        classifier = sys.argv[1]  # 'logistic_regression' or 'lightgbm'
    else:
        classifier = 'logistic_regression'
    
    if len(sys.argv) > 2:
        dataset = sys.argv[2]
        # Run single dataset
        run_additional_classifier_experiments(dataset, classifier)
    else:
        # Run all datasets
        run_all_datasets_with_new_classifier(classifier)
