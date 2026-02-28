"""
Run Additional Classifiers - WITH XGBOOST
==========================================

Run XGBoost on all cleaned datasets to complete classifier comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_xgboost_experiments(dataset_name: str):
    """
    Run XGBoost experiments on existing cleaned data.
    
    Args:
        dataset_name: 'pima', 'adult', etc.
    """
    
    # Load existing results
    results_dir = Path('results/tables')
    results_file = list(results_dir.glob(f'{dataset_name}_scientific_results_*.csv'))[0]
    
    logger.info(f"Loading existing results from: {results_file}")
    existing_results = pd.read_csv(results_file)
    
    orders = existing_results['cleaning_order'].unique()
    seeds = existing_results['seed'].unique()
    
    logger.info(f"Found {len(orders)} orders and {len(seeds)} seeds")
    logger.info(f"Will run XGBoost on all combinations")
    
    # Load clean dataset
    from src.data.loaders import load_dataset
    data = load_dataset(dataset_name)
    X_clean, y_clean = data['X'], data['y']
    
    # Convert to float to allow NaN injection
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
            
            # Inject faults
            from src.faults.injection import FaultInjector
            injector = FaultInjector(random_state=seed)
            
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
            
            # Apply cleaning
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
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0  # Suppress warnings
            )
            
            # Train and evaluate
            model.fit(X_train_cleaned, y_train_cleaned)
            y_pred = model.predict(X_test_clean)
            balanced_acc = balanced_accuracy_score(y_test_clean, y_pred)
            
            # Record result
            new_results.append({
                'dataset': dataset_name,
                'cleaning_order': order_str,
                'classifier': 'xgboost',
                'seed': seed,
                'balanced_accuracy': balanced_acc
            })
            
            logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    # Save results
    results_df = pd.DataFrame(new_results)
    output_file = results_dir / f'{dataset_name}_xgboost_results.csv'
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}")
    
    # Compute OSI
    compute_osi_comparison(results_df, dataset_name)
    
    return results_df


def compute_osi_comparison(results_df, dataset_name):
    """Compute OSI for XGBoost and compare to RF and LR."""
    
    # XGBoost OSI
    order_means = results_df.groupby('cleaning_order')['balanced_accuracy'].mean()
    best_order = order_means.idxmax()
    worst_order = order_means.idxmin()
    best_perf = order_means.max()
    worst_perf = order_means.min()
    xgb_osi = (best_perf - worst_perf) / best_perf
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OSI ANALYSIS - XGBoost")
    logger.info(f"{'='*60}")
    logger.info(f"Best order:  {best_order} ({best_perf:.4f})")
    logger.info(f"Worst order: {worst_order} ({worst_perf:.4f})")
    logger.info(f"XGBoost OSI: {xgb_osi:.4f} ({xgb_osi*100:.2f}%)")
    logger.info(f"Gap: {(best_perf - worst_perf):.4f} ({(best_perf - worst_perf)*100:.2f} pp)")
    
    # Compare to RF and LR
    results_dir = Path('results/tables')
    
    # RF results
    rf_file = list(results_dir.glob(f'{dataset_name}_scientific_results_*.csv'))[0]
    rf_results = pd.read_csv(rf_file)
    rf_means = rf_results.groupby('cleaning_order')['balanced_accuracy'].mean()
    rf_best = rf_means.max()
    rf_worst = rf_means.min()
    rf_osi = (rf_best - rf_worst) / rf_best
    
    # LR results
    lr_file = results_dir / f'{dataset_name}_logistic_regression_results.csv'
    if lr_file.exists():
        lr_results = pd.read_csv(lr_file)
        lr_means = lr_results.groupby('cleaning_order')['balanced_accuracy'].mean()
        lr_best = lr_means.max()
        lr_worst = lr_means.min()
        lr_osi = (lr_best - lr_worst) / lr_best
        
        logger.info(f"\nComparison Across Classifiers:")
        logger.info(f"  RF OSI:      {rf_osi*100:.2f}%")
        logger.info(f"  LR OSI:      {lr_osi*100:.2f}%")
        logger.info(f"  XGBoost OSI: {xgb_osi*100:.2f}%")
        logger.info(f"  XGB/RF ratio: {xgb_osi/rf_osi:.2f}×")
        logger.info(f"  XGB/LR ratio: {xgb_osi/lr_osi:.2f}×")
    else:
        logger.info(f"\nComparison to Random Forest:")
        logger.info(f"  RF OSI:      {rf_osi*100:.2f}%")
        logger.info(f"  XGBoost OSI: {xgb_osi*100:.2f}%")
        logger.info(f"  Ratio: {xgb_osi/rf_osi:.2f}×")
    
    if xgb_osi > 0.05:
        logger.info(f"\n✓ Effect holds for XGBoost!")
    else:
        logger.info(f"\n⚠ Effect weaker for XGBoost")


def run_all_datasets_xgboost():
    """Run all 5 datasets with XGBoost."""
    
    datasets = ['pima', 'adult', 'credit_default', 'spambase', 'breast_cancer']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING ALL DATASETS WITH XGBOOST")
    logger.info(f"{'='*70}\n")
    
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*70}\n")
        
        try:
            results = run_xgboost_experiments(dataset)
            all_results[dataset] = results
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create complete summary
    create_three_classifier_summary()


def create_three_classifier_summary():
    """Create complete comparison table for all 3 classifiers."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE SUMMARY: RF vs LR vs XGBoost")
    logger.info(f"{'='*70}\n")
    
    datasets = ['pima', 'adult', 'credit_default', 'spambase', 'breast_cancer']
    results_dir = Path('results/tables')
    
    summary_data = []
    
    for dataset in datasets:
        # RF OSI
        rf_file = list(results_dir.glob(f'{dataset}_scientific_results_*.csv'))[0]
        rf_df = pd.read_csv(rf_file)
        rf_means = rf_df.groupby('cleaning_order')['balanced_accuracy'].mean()
        rf_best = rf_means.max()
        rf_worst = rf_means.min()
        rf_osi = (rf_best - rf_worst) / rf_best
        
        # LR OSI
        lr_file = results_dir / f'{dataset}_logistic_regression_results.csv'
        if lr_file.exists():
            lr_df = pd.read_csv(lr_file)
            lr_means = lr_df.groupby('cleaning_order')['balanced_accuracy'].mean()
            lr_best = lr_means.max()
            lr_worst = lr_means.min()
            lr_osi = (lr_best - lr_worst) / lr_best
        else:
            lr_osi = None
        
        # XGBoost OSI
        xgb_file = results_dir / f'{dataset}_xgboost_results.csv'
        if xgb_file.exists():
            xgb_df = pd.read_csv(xgb_file)
            xgb_means = xgb_df.groupby('cleaning_order')['balanced_accuracy'].mean()
            xgb_best = xgb_means.max()
            xgb_worst = xgb_means.min()
            xgb_osi = (xgb_best - xgb_worst) / xgb_best
        else:
            xgb_osi = None
        
        summary_data.append({
            'Dataset': dataset,
            'RF_OSI': f"{rf_osi*100:.2f}%",
            'LR_OSI': f"{lr_osi*100:.2f}%" if lr_osi else 'N/A',
            'XGBoost_OSI': f"{xgb_osi*100:.2f}%" if xgb_osi else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save
    output_file = results_dir / 'classifier_comparison_all_three.csv'
    summary_df.to_csv(output_file, index=False)
    logger.info(f"\nComplete summary saved to: {output_file}")
    
    # Calculate means
    print("\n" + "="*70)
    print("MEAN OSI BY CLASSIFIER TYPE")
    print("="*70)
    
    imb_datasets = ['pima', 'adult', 'credit_default']
    bal_datasets = ['spambase', 'breast_cancer']
    
    for dataset_type, dataset_list in [('IMBALANCED', imb_datasets), ('BALANCED', bal_datasets)]:
        subset = summary_df[summary_df['Dataset'].isin(dataset_list)]
        
        rf_mean = np.mean([float(x.strip('%')) for x in subset['RF_OSI']])
        lr_mean = np.mean([float(x.strip('%')) for x in subset['LR_OSI'] if x != 'N/A'])
        xgb_vals = [float(x.strip('%')) for x in subset['XGBoost_OSI'] if x != 'N/A']
        xgb_mean = np.mean(xgb_vals) if xgb_vals else None
        
        print(f"\n{dataset_type} DATASETS:")
        print(f"  RF mean OSI:      {rf_mean:.2f}%")
        print(f"  LR mean OSI:      {lr_mean:.2f}%")
        if xgb_mean:
            print(f"  XGBoost mean OSI: {xgb_mean:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        # Run single dataset
        run_xgboost_experiments(dataset)
    else:
        # Run all datasets
        run_all_datasets_xgboost()
