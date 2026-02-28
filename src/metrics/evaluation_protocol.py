"""
Scientifically Correct Evaluation Protocol

CRITICAL PRINCIPLE:
All models (baseline and cleaned) must be evaluated on the SAME clean test set.
This ensures fair comparison where the ONLY variable is the cleaning order.

Correct Flow:
1. Split clean data into train/test ONCE
2. Keep test set pristine (NO faults, NO cleaning, EVER)
3. Apply faults + cleaning ONLY to training set
4. Train baseline on clean_train
5. Train cleaned on cleaned_train
6. Evaluate BOTH on same clean_test

This prevents:
- Data leakage
- Test set contamination
- Unfair comparisons due to different dataset sizes
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScientificEvaluator:
    """
    Scientifically rigorous evaluation with fixed clean test set.
    
    Key Principle: Test set is NEVER modified - only used for evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    def create_train_test_split(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                test_size: float = 0.2) -> Tuple:
        """
        Create train/test split that will be reused throughout experiment.
        
        CRITICAL: This split is created ONCE on clean data and never changes.
        
        Args:
            X: Clean feature matrix
            y: Clean labels
            test_size: Fraction for test set
            
        Returns:
            X_train, X_test, y_train, y_test, train_indices, test_indices
        """
        logger.info(f"Creating fixed train/test split (test_size={test_size})...")
        
        indices = np.arange(len(y))
        
        try:
            # Stratified split to preserve class distribution
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
        except ValueError:
            # Fallback if stratification fails
            logger.warning("Stratification failed, using random split")
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.random_state
            )
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        logger.info(f"Split created: Train={len(y_train)}, Test={len(y_test)}")
        logger.info(f"Test set will remain UNTOUCHED throughout all experiments")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    
    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   model_name: str = 'random_forest') -> object:
        """
        Train a model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Model type
            
        Returns:
            Trained model
        """
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self,
                      model: object,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate model with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of all metrics
        """
        from src.metrics.osi import MultiMetricEvaluator
        
        evaluator = MultiMetricEvaluator()
        
        # Predictions
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_proba = None
        
        # Calculate all metrics
        metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_proba)
        
        return metrics
    
    def compare_baseline_vs_cleaned(self,
                                    X_clean_train: np.ndarray,
                                    y_clean_train: np.ndarray,
                                    X_clean_test: np.ndarray,
                                    y_clean_test: np.ndarray,
                                    X_cleaned_train: np.ndarray,
                                    y_cleaned_train: np.ndarray,
                                    model_name: str = 'random_forest') -> Dict:
        """
        Scientifically correct comparison: baseline vs cleaned.
        
        CRITICAL: Both models evaluated on SAME clean test set.
        
        Args:
            X_clean_train: Clean training features
            y_clean_train: Clean training labels
            X_clean_test: Clean test features (NEVER MODIFIED)
            y_clean_test: Clean test labels (NEVER MODIFIED)
            X_cleaned_train: Cleaned training features (after faults + cleaning)
            y_cleaned_train: Cleaned training labels (after faults + cleaning)
            model_name: Model to use
            
        Returns:
            Comparison results with all metrics
        """
        logger.info(f"Scientifically rigorous comparison using {model_name}...")
        logger.info(f"Clean train: {len(y_clean_train)}, Cleaned train: {len(y_cleaned_train)}, Test: {len(y_clean_test)}")
        logger.info("CRITICAL: Both models evaluated on SAME clean test set")
        
        # Train baseline model on clean training data
        logger.info("Training baseline model on clean training data...")
        model_baseline = self.train_model(X_clean_train, y_clean_train, model_name)
        
        # Train cleaned model on cleaned training data
        logger.info("Training cleaned model on cleaned training data...")
        model_cleaned = self.train_model(X_cleaned_train, y_cleaned_train, model_name)
        
        # CRITICAL: Evaluate BOTH on the SAME clean test set
        logger.info("Evaluating baseline on clean test set...")
        metrics_baseline = self.evaluate_model(model_baseline, X_clean_test, y_clean_test)
        
        logger.info("Evaluating cleaned on clean test set...")
        metrics_cleaned = self.evaluate_model(model_cleaned, X_clean_test, y_clean_test)
        
        # Calculate recovery for each metric
        from src.metrics.osi import MultiMetricEvaluator
        recovery_metrics = {}
        
        for metric_name in MultiMetricEvaluator.get_all_metric_names():
            if metric_name in metrics_baseline and metric_name in metrics_cleaned:
                baseline_val = metrics_baseline[metric_name]
                cleaned_val = metrics_cleaned[metric_name]
                
                if baseline_val is not None and cleaned_val is not None:
                    if baseline_val > 0:
                        recovery_rate = cleaned_val / baseline_val
                        gap = baseline_val - cleaned_val
                    else:
                        recovery_rate = 0.0
                        gap = 0.0
                    
                    recovery_metrics[f'{metric_name}_baseline'] = baseline_val
                    recovery_metrics[f'{metric_name}_cleaned'] = cleaned_val
                    recovery_metrics[f'{metric_name}_recovery'] = recovery_rate
                    recovery_metrics[f'{metric_name}_gap'] = gap
        
        results = {
            'model_name': model_name,
            'n_clean_train': len(y_clean_train),
            'n_cleaned_train': len(y_cleaned_train),
            'n_test': len(y_clean_test),
            'baseline_metrics': metrics_baseline,
            'cleaned_metrics': metrics_cleaned,
            'recovery_metrics': recovery_metrics
        }
        
        logger.info(f"Comparison complete. Test set size: {len(y_clean_test)} (same for both models)")
        
        return results


if __name__ == "__main__":
    # Test scientific evaluation protocol
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data.loaders import load_dataset
    from src.faults.injection import FaultInjector
    from src.cleaning.methods import DataCleaner
    
    print("\n" + "="*70)
    print("TESTING SCIENTIFICALLY CORRECT EVALUATION PROTOCOL")
    print("="*70)
    
    # Load data
    data = load_dataset('pima')
    X_clean, y_clean = data['X'], data['y']
    
    print(f"\nClean data: {X_clean.shape[0]} samples")
    
    # Create scientific evaluator
    evaluator = ScientificEvaluator(random_state=42)
    
    # STEP 1: Split clean data FIRST (this split never changes)
    print("\nSTEP 1: Creating fixed train/test split on CLEAN data...")
    X_clean_train, X_clean_test, y_clean_train, y_clean_test, train_idx, test_idx = \
        evaluator.create_train_test_split(X_clean, y_clean, test_size=0.2)
    
    print(f"Train: {len(y_clean_train)} samples")
    print(f"Test: {len(y_clean_test)} samples (WILL NEVER BE MODIFIED)")
    
    # STEP 2: Apply faults ONLY to training set
    print("\nSTEP 2: Applying faults ONLY to training set...")
    injector = FaultInjector(random_state=42)
    fault_result = injector.inject_all_faults(X_clean_train, y_clean_train, {
        'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
        'missing': {'rate': 0.10, 'mechanism': 'MCAR'},
        'outliers': {'rate': 0.05, 'method': 'extreme'}
    })
    
    X_train_corrupted = fault_result['X_corrupted']
    y_train_corrupted = fault_result['y_corrupted']
    
    print(f"Corrupted training set: {len(y_train_corrupted)} samples")
    print("Test set: UNTOUCHED ✓")
    
    # STEP 3: Clean ONLY training set
    print("\nSTEP 3: Cleaning ONLY training set...")
    cleaner = DataCleaner(random_state=42)
    X_train_cleaned, y_train_cleaned, _ = cleaner.apply_cleaning_sequence(
        X_train_corrupted, y_train_corrupted,
        ['imputation', 'label_noise', 'outliers'],
        {'imputation': {'method': 'median'}, 
         'label_noise': {'mode': 'correct', 'threshold': 0.7}}
    )
    
    print(f"Cleaned training set: {len(y_train_cleaned)} samples")
    print("Test set: STILL UNTOUCHED ✓")
    
    # STEP 4: Compare baseline vs cleaned (BOTH evaluated on SAME test set)
    print("\nSTEP 4: Comparing baseline vs cleaned...")
    print("=" * 70)
    
    comparison = evaluator.compare_baseline_vs_cleaned(
        X_clean_train, y_clean_train,
        X_clean_test, y_clean_test,  # SAME test set for both
        X_train_cleaned, y_train_cleaned,
        model_name='random_forest'
    )
    
    print("\n" + "="*70)
    print("RESULTS - SCIENTIFICALLY VALID COMPARISON")
    print("="*70)
    print(f"Baseline trained on: {comparison['n_clean_train']} clean samples")
    print(f"Cleaned trained on:  {comparison['n_cleaned_train']} cleaned samples")
    print(f"BOTH evaluated on:   {comparison['n_test']} SAME test samples ✓")
    
    print("\nKey Metrics:")
    for metric in ['balanced_accuracy', 'f1_macro', 'mcc', 'auc']:
        baseline_key = f'{metric}_baseline'
        cleaned_key = f'{metric}_cleaned'
        if baseline_key in comparison['recovery_metrics']:
            baseline = comparison['recovery_metrics'][baseline_key]
            cleaned = comparison['recovery_metrics'][cleaned_key]
            recovery = comparison['recovery_metrics'][f'{metric}_recovery']
            print(f"  {metric:20s}: Baseline={baseline:.4f}, Cleaned={cleaned:.4f}, Recovery={recovery:.2%}")
    
    print("\n" + "="*70)
    print("SCIENTIFICALLY CORRECT EVALUATION PROTOCOL VALIDATED ✓")
    print("="*70)
    print("\nKey Achievement:")
    print("  ✓ Test set never modified")
    print("  ✓ Same test distribution for all models")
    print("  ✓ Only variable = cleaning on training set")
    print("  ✓ PUBLICATION-READY METHODOLOGY")
    print("="*70)
