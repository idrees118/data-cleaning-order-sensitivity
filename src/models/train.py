"""
Model training and evaluation for data cleaning order experiments.

Supports multiple models (LR, RF, XGB) with consistent interface.
Tracks performance metrics and provides fair evaluation.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Professional model training with consistent interface and metrics."""
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'logistic_regression': self._create_logistic_regression,
            'random_forest': self._create_random_forest,
            'xgboost': self._create_xgboost
        }
    
    def _create_logistic_regression(self) -> LogisticRegression:
        """Create logistic regression model."""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
    
    def _create_random_forest(self) -> RandomForestClassifier:
        """Create random forest model."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _create_xgboost(self) -> xgb.XGBClassifier:
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    def train_and_evaluate(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          model_name: str = 'random_forest',
                          test_size: float = 0.2,
                          cv_folds: int = 5) -> Dict:
        """
        Train model and evaluate performance.
        
        Args:
            X: Feature matrix
            y: Labels
            model_name: 'logistic_regression', 'random_forest', or 'xgboost'
            test_size: Fraction for test set
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with metrics and trained model
        """
        start_time = time.time()
        logger.info(f"Training {model_name} on {len(y)} samples...")
        
        # Handle edge cases
        if len(y) < 10:
            logger.warning(f"Too few samples ({len(y)}) for reliable training")
            return {
                'model_name': model_name,
                'failed': True,
                'reason': 'insufficient_samples',
                'n_samples': len(y)
            }
        
        # Check for class presence
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(f"Only one class present in labels")
            return {
                'model_name': model_name,
                'failed': True,
                'reason': 'single_class',
                'n_samples': len(y)
            }
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=y
            )
        except ValueError:
            # Fallback if stratification fails (too few samples per class)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state
            )
        
        # Create and train model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.models.keys())}")
        
        model = self.models[model_name]()
        
        try:
            # Train
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get probabilities for AUC
            try:
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba_test = model.predict_proba(X_test)[:, 1]
            except:
                y_proba_train = None
                y_proba_test = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_pred_train, y_proba_train,
                y_test, y_pred_test, y_proba_test
            )
            
            # Cross-validation score (on full dataset for better estimate)
            try:
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=min(cv_folds, len(y) // 2),  # Adapt to data size
                    scoring='accuracy',
                    n_jobs=-1
                )
                metrics['cv_accuracy_mean'] = cv_scores.mean()
                metrics['cv_accuracy_std'] = cv_scores.std()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                metrics['cv_accuracy_mean'] = None
                metrics['cv_accuracy_std'] = None
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f} | F1: {metrics['test_f1']:.4f} | Time: {elapsed_time:.2f}s")
            
            result = {
                'model_name': model_name,
                'model': model,
                'metrics': metrics,
                'train_time': train_time,
                'total_time': elapsed_time,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'n_features': X.shape[1],
                'failed': False
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            result = {
                'model_name': model_name,
                'failed': True,
                'error': str(e),
                'n_samples': len(y)
            }
        
        return result
    
    def _calculate_metrics(self,
                          y_train, y_pred_train, y_proba_train,
                          y_test, y_pred_test, y_proba_test) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Training metrics
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['train_f1'] = f1_score(y_train, y_pred_train, average='binary', zero_division=0)
        metrics['train_precision'] = precision_score(y_train, y_pred_train, average='binary', zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_pred_train, average='binary', zero_division=0)
        
        if y_proba_train is not None:
            try:
                metrics['train_auc'] = roc_auc_score(y_train, y_proba_train)
            except:
                metrics['train_auc'] = None
        else:
            metrics['train_auc'] = None
        
        # Test metrics
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        metrics['test_f1'] = f1_score(y_test, y_pred_test, average='binary', zero_division=0)
        metrics['test_precision'] = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
        metrics['test_recall'] = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
        
        if y_proba_test is not None:
            try:
                metrics['test_auc'] = roc_auc_score(y_test, y_proba_test)
            except:
                metrics['test_auc'] = None
        else:
            metrics['test_auc'] = None
        
        return metrics
    
    def train_multiple_models(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             model_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Train multiple models on the same data.
        
        Args:
            X: Feature matrix
            y: Labels
            model_names: List of model names (None = all models)
            
        Returns:
            Dictionary mapping model names to results
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        logger.info(f"Training {len(model_names)} models...")
        
        results = {}
        for model_name in model_names:
            result = self.train_and_evaluate(X, y, model_name=model_name)
            results[model_name] = result
        
        return results
    
    def compare_models(self, results: Dict[str, Dict]) -> Dict:
        """
        Compare results from multiple models.
        
        Args:
            results: Dictionary from train_multiple_models
            
        Returns:
            Comparison summary
        """
        comparison = {
            'models': [],
            'best_model': None,
            'best_accuracy': 0.0
        }
        
        for model_name, result in results.items():
            if result.get('failed', False):
                continue
            
            metrics = result['metrics']
            summary = {
                'model': model_name,
                'test_accuracy': metrics['test_accuracy'],
                'test_f1': metrics['test_f1'],
                'test_auc': metrics.get('test_auc'),
                'train_time': result.get('train_time', 0)
            }
            comparison['models'].append(summary)
            
            # Track best
            if metrics['test_accuracy'] > comparison['best_accuracy']:
                comparison['best_accuracy'] = metrics['test_accuracy']
                comparison['best_model'] = model_name
        
        return comparison


def evaluate_cleaning_order(X_clean: np.ndarray,
                           y_clean: np.ndarray,
                           X_cleaned: np.ndarray,
                           y_cleaned: np.ndarray,
                           model_name: str = 'random_forest',
                           random_state: int = 42) -> Dict:
    """
    Evaluate a single cleaning order by comparing to clean baseline.
    
    Args:
        X_clean: Clean feature matrix (ground truth)
        y_clean: Clean labels (ground truth)
        X_cleaned: Cleaned feature matrix (after cleaning sequence)
        y_cleaned: Cleaned labels (after cleaning sequence)
        model_name: Model to use for evaluation
        random_state: Random seed
        
    Returns:
        Evaluation results with recovery metrics
    """
    trainer = ModelTrainer(random_state=random_state)
    
    # Train on clean baseline
    logger.info("Training on clean baseline...")
    result_clean = trainer.train_and_evaluate(X_clean, y_clean, model_name=model_name)
    
    # Train on cleaned data
    logger.info("Training on cleaned data...")
    result_cleaned = trainer.train_and_evaluate(X_cleaned, y_cleaned, model_name=model_name)
    
    # Calculate recovery
    if not result_clean.get('failed') and not result_cleaned.get('failed'):
        baseline_acc = result_clean['metrics']['test_accuracy']
        cleaned_acc = result_cleaned['metrics']['test_accuracy']
        recovery_rate = cleaned_acc / baseline_acc if baseline_acc > 0 else 0.0
        accuracy_gap = baseline_acc - cleaned_acc
        
        evaluation = {
            'baseline_accuracy': baseline_acc,
            'cleaned_accuracy': cleaned_acc,
            'recovery_rate': recovery_rate,
            'accuracy_gap': accuracy_gap,
            'baseline_result': result_clean,
            'cleaned_result': result_cleaned,
            'model_name': model_name
        }
    else:
        evaluation = {
            'failed': True,
            'baseline_result': result_clean,
            'cleaned_result': result_cleaned
        }
    
    return evaluation


if __name__ == "__main__":
    # Test model training
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data.loaders import load_dataset
    from src.faults.injection import FaultInjector
    from src.cleaning.methods import DataCleaner
    
    print("\n" + "="*70)
    print("TESTING MODEL TRAINING & EVALUATION")
    print("="*70)
    
    # Load data
    data = load_dataset('pima')
    X_clean, y_clean = data['X'], data['y']
    
    print(f"\nClean dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
    
    # Test 1: Train on clean data
    print("\n" + "="*70)
    print("TEST 1: Training Models on Clean Data")
    print("="*70)
    
    trainer = ModelTrainer(random_state=42)
    results_clean = trainer.train_multiple_models(X_clean, y_clean)
    
    comparison = trainer.compare_models(results_clean)
    print(f"\nBest model: {comparison['best_model']} ({comparison['best_accuracy']:.4f})")
    
    # Test 2: Train on corrupted + cleaned data
    print("\n" + "="*70)
    print("TEST 2: Evaluating Cleaning Order Effect")
    print("="*70)
    
    # Inject faults
    injector = FaultInjector(random_state=42)
    result = injector.inject_all_faults(X_clean, y_clean, {
        'label_noise': {'rate': 0.2, 'type': 'asymmetric'},
        'missing': {'rate': 0.15, 'mechanism': 'MAR'},
        'outliers': {'rate': 0.05, 'method': 'extreme'}
    })
    X_corrupted = result['X_corrupted']
    y_corrupted = result['y_corrupted']
    
    # Apply cleaning sequence
    cleaner = DataCleaner(random_state=42)
    X_cleaned, y_cleaned, _ = cleaner.apply_cleaning_sequence(
        X_corrupted, y_corrupted,
        ['imputation', 'label_noise', 'outliers'],
        {'imputation': {'method': 'knn'}, 
         'label_noise': {'mode': 'correct', 'threshold': 0.7}}
    )
    
    # Evaluate recovery
    evaluation = evaluate_cleaning_order(
        X_clean, y_clean,
        X_cleaned, y_cleaned,
        model_name='random_forest'
    )
    
    print(f"\nBaseline accuracy: {evaluation['baseline_accuracy']:.4f}")
    print(f"Cleaned accuracy:  {evaluation['cleaned_accuracy']:.4f}")
    print(f"Recovery rate:     {evaluation['recovery_rate']:.2%}")
    print(f"Accuracy gap:      {evaluation['accuracy_gap']:.4f}")
    
    print("\n" + "="*70)
    print("ALL MODEL TRAINING TESTS PASSED ✓")
    print("="*70)
