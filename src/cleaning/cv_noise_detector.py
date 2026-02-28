"""
CV-Based Label Noise Detection - Production Grade

CRITICAL RULES:
1. Out-of-fold predictions ONLY
2. Corruption mask for evaluation ONLY (never training)
3. Log EVERYTHING for reproducibility
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def detect_label_noise_cv(X: np.ndarray,
                          y: np.ndarray,
                          threshold: float = 0.8,
                          n_splits: int = 5,
                          corruption_mask: Optional[np.ndarray] = None,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Detect label noise using cross-validation predictions.
    
    CRITICAL: Uses out-of-fold predictions only - no sample is predicted
    by a model that saw it during training.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - potentially noisy
        threshold: Confidence threshold (samples with max_proba < threshold are flagged)
        n_splits: Number of CV folds
        corruption_mask: Ground truth (n_samples,) boolean array - True = corrupted label
        random_state: Random seed
        
    Returns:
        flagged_indices: Indices of samples flagged as noisy
        cv_probabilities: (n_samples, n_classes) CV probability estimates
        metrics: Dict with detection metrics (includes P/R/F1 if corruption_mask provided)
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    # Initialize probability storage
    cv_proba = np.zeros((n_samples, n_classes))
    cv_counts = np.zeros(n_samples)  # Track how many times each sample was in holdout
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    logger.info(f"Running {n_splits}-fold CV for noise detection...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Train on training fold
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        
        # Train classifier (using same params as before)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state + fold_idx,
            n_jobs=-1
        )
        rf.fit(X_train_fold, y_train_fold)
        
        # Predict on holdout fold (OUT-OF-FOLD PREDICTIONS)
        proba_fold = rf.predict_proba(X[val_idx])
        
        # Accumulate probabilities
        cv_proba[val_idx] += proba_fold
        cv_counts[val_idx] += 1
    
    # Average probabilities (some samples might be predicted multiple times if folds overlap)
    # In standard k-fold, each sample appears exactly once, but being safe
    valid_mask = cv_counts > 0
    cv_proba[valid_mask] /= cv_counts[valid_mask, np.newaxis]
    
    # Sanity check
    if not np.all(valid_mask):
        logger.warning(f"{(~valid_mask).sum()} samples not predicted by any fold!")
    
    # Get predicted class and confidence
    predicted_classes = cv_proba.argmax(axis=1)
    confidences = cv_proba[np.arange(n_samples), predicted_classes]
    
    # Flag low-confidence samples that disagree with original label
    low_conf = confidences < threshold
    disagrees = predicted_classes != y
    flagged_mask = low_conf & disagrees
    flagged_indices = np.where(flagged_mask)[0]
    
    n_flagged = len(flagged_indices)
    flag_rate = n_flagged / n_samples
    
    # Basic metrics
    metrics = {
        'n_flagged': n_flagged,
        'flag_rate': flag_rate,
        'threshold': threshold,
        'n_splits': n_splits,
        'mean_confidence': confidences.mean(),
        'flagged_mean_confidence': confidences[flagged_indices].mean() if n_flagged > 0 else 0.0
    }
    
    # Ground truth evaluation (if available)
    if corruption_mask is not None:
        corruption_mask = np.array(corruption_mask, dtype=bool)
        
        # Calculate confusion matrix
        true_noisy = corruption_mask
        predicted_noisy = flagged_mask
        
        tp = np.sum(true_noisy & predicted_noisy)  # Correctly flagged as noisy
        fp = np.sum(~true_noisy & predicted_noisy)  # Clean but flagged
        fn = np.sum(true_noisy & ~predicted_noisy)  # Noisy but not flagged
        tn = np.sum(~true_noisy & ~predicted_noisy)  # Clean and not flagged
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.update({
            'n_truly_noisy': int(true_noisy.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return flagged_indices, cv_proba, metrics


if __name__ == "__main__":
    # Test with synthetic data
    print("\n" + "="*70)
    print("TESTING CV-BASED NOISE DETECTION")
    print("="*70)
    
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Inject synthetic label noise
    np.random.seed(42)
    noise_rate = 0.15
    n_noisy = int(len(y) * noise_rate)
    noisy_indices = np.random.choice(len(y), size=n_noisy, replace=False)
    
    corruption_mask = np.zeros(len(y), dtype=bool)
    corruption_mask[noisy_indices] = True
    
    y_noisy = y.copy()
    y_noisy[noisy_indices] = 1 - y_noisy[noisy_indices]  # Flip labels
    
    print(f"\nGenerated data: {len(y)} samples, {X.shape[1]} features")
    print(f"Injected noise: {n_noisy} labels ({noise_rate:.1%})")
    
    # Test detection at different thresholds
    for threshold in [0.7, 0.8, 0.9]:
        print(f"\n{'='*70}")
        print(f"Testing threshold = {threshold}")
        print("="*70)
        
        flagged, proba, metrics = detect_label_noise_cv(
            X, y_noisy,
            threshold=threshold,
            corruption_mask=corruption_mask,
            random_state=42
        )
        
        print(f"\nResults:")
        print(f"  Flagged: {metrics['n_flagged']} samples ({metrics['flag_rate']:.1%})")
        print(f"  Precision: {metrics['precision']:.1%} (of flagged, truly noisy)")
        print(f"  Recall: {metrics['recall']:.1%} (of noisy, caught)")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  Mean confidence (all): {metrics['mean_confidence']:.3f}")
        print(f"  Mean confidence (flagged): {metrics['flagged_mean_confidence']:.3f}")
    
    print("\n" + "="*70)
    print("CV-BASED DETECTION TEST COMPLETE ✓")
    print("="*70)
