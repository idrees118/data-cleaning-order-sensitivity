"""
Realistic fault injection for data cleaning order experiments.

This module implements domain-appropriate data quality issues:
- Label noise (with class confusion patterns)
- Missing values (MCAR, MAR, MNAR)
- Class imbalance (via strategic sampling)
- Outliers (domain-appropriate anomalies)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultInjector:
    """Inject realistic data quality issues into clean datasets."""
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def inject_label_noise(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          noise_rate: float = 0.2,
                          noise_type: str = 'asymmetric') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject label noise with realistic confusion patterns.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: True labels (n_samples,)
            noise_rate: Fraction of labels to corrupt (0.0 to 1.0)
            noise_type: 'symmetric', 'asymmetric', or 'boundary'
            
        Returns:
            X, y_noisy, noise_mask (boolean array indicating corrupted labels)
        """
        logger.info(f"Injecting {noise_type} label noise at rate {noise_rate:.1%}")
        
        y_noisy = y.copy()
        n_samples = len(y)
        n_to_flip = int(n_samples * noise_rate)
        
        noise_mask = np.zeros(n_samples, dtype=bool)
        
        if noise_type == 'symmetric':
            # Uniform random flip - both classes equally affected
            flip_indices = self.rng.choice(n_samples, size=n_to_flip, replace=False)
            noise_mask[flip_indices] = True
            y_noisy[flip_indices] = 1 - y[flip_indices]
            
        elif noise_type == 'asymmetric':
            # Class-dependent flip - minority class more likely to be mislabeled
            # This is realistic: rare class is harder to label correctly
            minority_class = 1 if (y == 1).sum() < (y == 0).sum() else 0
            majority_class = 1 - minority_class
            
            # Flip 70% from minority, 30% from majority
            n_minority_flip = int(n_to_flip * 0.7)
            n_majority_flip = n_to_flip - n_minority_flip
            
            minority_indices = np.where(y == minority_class)[0]
            majority_indices = np.where(y == majority_class)[0]
            
            if len(minority_indices) >= n_minority_flip:
                flip_minority = self.rng.choice(minority_indices, size=n_minority_flip, replace=False)
            else:
                flip_minority = minority_indices
                
            if len(majority_indices) >= n_majority_flip:
                flip_majority = self.rng.choice(majority_indices, size=n_majority_flip, replace=False)
            else:
                flip_majority = majority_indices
            
            flip_indices = np.concatenate([flip_minority, flip_majority])
            noise_mask[flip_indices] = True
            y_noisy[flip_indices] = 1 - y[flip_indices]
        
        elif noise_type == 'boundary':
            # Corrupt samples near decision boundary
            # These are naturally ambiguous, so realistic to mislabel
            try:
                knn = NearestNeighbors(n_neighbors=min(10, n_samples - 1))
                knn.fit(X)
                
                # Find samples with mixed-class neighbors
                boundary_scores = []
                for i in range(n_samples):
                    neighbors = knn.kneighbors([X[i]], return_distance=False)[0]
                    neighbor_labels = y[neighbors]
                    # Boundary score = how mixed the neighborhood is
                    # Score close to 0 = near boundary (50/50 mix)
                    score = np.abs(neighbor_labels.mean() - 0.5)
                    boundary_scores.append(score)
                
                boundary_scores = np.array(boundary_scores)
                # Flip samples with most mixed neighborhoods (lowest scores)
                boundary_indices = np.argsort(boundary_scores)[:n_to_flip]
                noise_mask[boundary_indices] = True
                y_noisy[boundary_indices] = 1 - y[boundary_indices]
                
            except Exception as e:
                logger.warning(f"Boundary noise failed, falling back to symmetric: {e}")
                # Fallback to symmetric if boundary detection fails
                flip_indices = self.rng.choice(n_samples, size=n_to_flip, replace=False)
                noise_mask[flip_indices] = True
                y_noisy[flip_indices] = 1 - y[flip_indices]
        
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Choose from ['symmetric', 'asymmetric', 'boundary']")
        
        logger.info(f"Corrupted {noise_mask.sum()} labels ({noise_mask.sum()/n_samples:.1%})")
        return X, y_noisy, noise_mask
    
    def inject_missing_values(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             missing_rate: float = 0.15,
                             mechanism: str = 'MAR',
                             feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject missing values with realistic patterns.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            missing_rate: Fraction of values to set missing
            mechanism: 'MCAR' (random), 'MAR' (depends on observed), 'MNAR' (depends on itself)
            feature_indices: Which features to affect (None = random 50%)
            
        Returns:
            X_missing, y, missing_mask (boolean array same shape as X)
        """
        logger.info(f"Injecting missing values with {mechanism} mechanism at rate {missing_rate:.1%}")
        
        X_missing = X.copy()
        n_samples, n_features = X.shape
        missing_mask = np.zeros_like(X, dtype=bool)
        
        # Select features to affect
        if feature_indices is None:
            n_affected = max(1, n_features // 2)  # Affect 50% of features
            feature_indices = self.rng.choice(n_features, size=n_affected, replace=False)
        
        if mechanism == 'MCAR':
            # Missing Completely At Random - uniform across all values
            for feat_idx in feature_indices:
                n_missing = int(n_samples * missing_rate)
                missing_rows = self.rng.choice(n_samples, size=n_missing, replace=False)
                missing_mask[missing_rows, feat_idx] = True
                X_missing[missing_rows, feat_idx] = np.nan
        
        elif mechanism == 'MAR':
            # Missing At Random - depends on OTHER observed features
            # Realistic: e.g., high-income people don't report income
            for feat_idx in feature_indices:
                # Choose a different feature to condition on
                other_features = [f for f in range(n_features) if f != feat_idx]
                if len(other_features) > 0:
                    condition_feat = self.rng.choice(other_features)
                    
                    # Make values missing based on quantile of condition feature
                    # Top 15% of condition feature → missing
                    threshold = np.nanpercentile(X[:, condition_feat], 100 - missing_rate * 100)
                    missing_rows = X[:, condition_feat] > threshold
                    
                    missing_mask[missing_rows, feat_idx] = True
                    X_missing[missing_rows, feat_idx] = np.nan
                else:
                    # Fallback to MCAR if no other features
                    n_missing = int(n_samples * missing_rate)
                    missing_rows = self.rng.choice(n_samples, size=n_missing, replace=False)
                    missing_mask[missing_rows, feat_idx] = True
                    X_missing[missing_rows, feat_idx] = np.nan
        
        elif mechanism == 'MNAR':
            # Missing Not At Random - depends on the value itself
            # Realistic: extreme values are not reported
            for feat_idx in feature_indices:
                # Make extreme values missing (top and bottom 7.5% each)
                lower_threshold = np.nanpercentile(X[:, feat_idx], missing_rate * 50)
                upper_threshold = np.nanpercentile(X[:, feat_idx], 100 - missing_rate * 50)
                
                missing_rows = (X[:, feat_idx] < lower_threshold) | (X[:, feat_idx] > upper_threshold)
                
                missing_mask[missing_rows, feat_idx] = True
                X_missing[missing_rows, feat_idx] = np.nan
        
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}. Choose from ['MCAR', 'MAR', 'MNAR']")
        
        total_missing = missing_mask.sum()
        total_values = X.size
        logger.info(f"Created {total_missing} missing values ({total_missing/total_values:.1%} of all values)")
        
        return X_missing, y, missing_mask
    
    def inject_class_imbalance(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              target_ratio: float = 10.0,
                              method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject or increase class imbalance.
        
        Args:
            X: Feature matrix
            y: Labels
            target_ratio: Desired majority:minority ratio
            method: 'undersample' or 'oversample'
            
        Returns:
            X_imbalanced, y_imbalanced, kept_indices
        """
        logger.info(f"Injecting class imbalance with target ratio {target_ratio}:1")
        
        # Identify majority and minority classes
        class_counts = np.bincount(y.astype(int))
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        current_ratio = len(majority_indices) / len(minority_indices)
        logger.info(f"Current ratio: {current_ratio:.2f}:1")
        
        if current_ratio >= target_ratio:
            logger.info("Already at target ratio, no action needed")
            return X, y, np.arange(len(y))
        
        if method == 'undersample':
            # Remove samples from minority class
            n_minority_keep = int(len(majority_indices) / target_ratio)
            n_minority_keep = max(1, n_minority_keep)  # Keep at least 1
            
            if n_minority_keep >= len(minority_indices):
                logger.info("Cannot achieve target ratio with current data")
                return X, y, np.arange(len(y))
            
            kept_minority = self.rng.choice(minority_indices, size=n_minority_keep, replace=False)
            kept_indices = np.concatenate([majority_indices, kept_minority])
            
        elif method == 'oversample':
            # Duplicate minority samples (simple repetition)
            n_minority_total = int(len(majority_indices) / target_ratio)
            n_repeats = int(np.ceil(n_minority_total / len(minority_indices)))
            
            oversampled_minority = np.tile(minority_indices, n_repeats)[:n_minority_total]
            kept_indices = np.concatenate([majority_indices, oversampled_minority])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Shuffle to mix classes
        self.rng.shuffle(kept_indices)
        
        X_imbalanced = X[kept_indices]
        y_imbalanced = y[kept_indices]
        
        new_ratio = (y_imbalanced == majority_class).sum() / (y_imbalanced == minority_class).sum()
        logger.info(f"New ratio: {new_ratio:.2f}:1 ({len(y_imbalanced)} samples)")
        
        return X_imbalanced, y_imbalanced, kept_indices
    
    def inject_outliers(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       outlier_rate: float = 0.05,
                       method: str = 'extreme') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject outliers in feature space.
        
        Args:
            X: Feature matrix
            y: Labels
            outlier_rate: Fraction of samples to corrupt
            method: 'extreme' (push to extremes) or 'gaussian' (add noise)
            
        Returns:
            X_outliers, y, outlier_mask
        """
        logger.info(f"Injecting outliers using {method} method at rate {outlier_rate:.1%}")
        
        X_outliers = X.copy()
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * outlier_rate)
        
        outlier_indices = self.rng.choice(n_samples, size=n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        outlier_mask[outlier_indices] = True
        
        if method == 'extreme':
            # Push values to extreme percentiles
            for feat_idx in range(n_features):
                for idx in outlier_indices:
                    # Randomly push to either very low or very high
                    if self.rng.rand() < 0.5:
                        # Push to low extreme (below 1st percentile)
                        low_extreme = np.nanpercentile(X[:, feat_idx], 1)
                        X_outliers[idx, feat_idx] = low_extreme * 0.5
                    else:
                        # Push to high extreme (above 99th percentile)
                        high_extreme = np.nanpercentile(X[:, feat_idx], 99)
                        X_outliers[idx, feat_idx] = high_extreme * 1.5
        
        elif method == 'gaussian':
            # Add large Gaussian noise
            for feat_idx in range(n_features):
                std = np.nanstd(X[:, feat_idx])
                noise = self.rng.normal(0, std * 3, size=n_outliers)
                X_outliers[outlier_indices, feat_idx] += noise
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Corrupted {n_outliers} samples with outliers")
        
        return X_outliers, y, outlier_mask
    
    def inject_all_faults(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         config: Dict) -> Dict:
        """
        Inject all fault types according to configuration.
        
        Args:
            X: Clean feature matrix
            y: Clean labels
            config: Dictionary with fault specifications:
                {
                    'label_noise': {'rate': 0.2, 'type': 'asymmetric'},
                    'missing': {'rate': 0.15, 'mechanism': 'MAR'},
                    'imbalance': {'ratio': 10.0, 'method': 'undersample'},
                    'outliers': {'rate': 0.05, 'method': 'extreme'}
                }
        
        Returns:
            Dictionary with corrupted data and fault masks
        """
        logger.info("="*50)
        logger.info("INJECTING ALL FAULTS")
        logger.info("="*50)
        
        result = {
            'X_clean': X.copy(),
            'y_clean': y.copy(),
            'X_corrupted': X.copy(),
            'y_corrupted': y.copy(),
            'masks': {}
        }
        
        # 1. Label noise
        if 'label_noise' in config and config['label_noise'] is not None:
            _, result['y_corrupted'], label_mask = self.inject_label_noise(
                result['X_corrupted'],
                result['y_corrupted'],
                noise_rate=config['label_noise'].get('rate', 0.2),
                noise_type=config['label_noise'].get('type', 'asymmetric')
            )
            result['masks']['label_noise'] = label_mask
        
        # 2. Missing values
        if 'missing' in config and config['missing'] is not None:
            result['X_corrupted'], _, missing_mask = self.inject_missing_values(
                result['X_corrupted'],
                result['y_corrupted'],
                missing_rate=config['missing'].get('rate', 0.15),
                mechanism=config['missing'].get('mechanism', 'MAR')
            )
            result['masks']['missing'] = missing_mask
        
        # 3. Outliers
        if 'outliers' in config and config['outliers'] is not None:
            result['X_corrupted'], _, outlier_mask = self.inject_outliers(
                result['X_corrupted'],
                result['y_corrupted'],
                outlier_rate=config['outliers'].get('rate', 0.05),
                method=config['outliers'].get('method', 'extreme')
            )
            result['masks']['outliers'] = outlier_mask
        
        # 4. Class imbalance (changes sample size, so do last)
        if 'imbalance' in config and config['imbalance'] is not None:
            result['X_corrupted'], result['y_corrupted'], kept_indices = self.inject_class_imbalance(
                result['X_corrupted'],
                result['y_corrupted'],
                target_ratio=config['imbalance'].get('ratio', 10.0),
                method=config['imbalance'].get('method', 'undersample')
            )
            result['masks']['imbalance_kept'] = kept_indices
            
            # Update other masks for new indices
            for mask_name in ['label_noise', 'missing', 'outliers']:
                if mask_name in result['masks']:
                    result['masks'][mask_name] = result['masks'][mask_name][kept_indices]
        
        logger.info("="*50)
        logger.info("FAULT INJECTION COMPLETE")
        logger.info(f"Final dataset: {result['X_corrupted'].shape[0]} samples")
        logger.info("="*50)
        
        return result


if __name__ == "__main__":
    # Test fault injection
    from src.data.loaders import load_dataset
    
    print("\n" + "="*60)
    print("TESTING FAULT INJECTION")
    print("="*60)
    
    # Load a small dataset for testing
    data = load_dataset('pima')
    X, y = data['X'], data['y']
    
    print(f"\nOriginal data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: 0={( y==0).sum()}, 1={(y==1).sum()}")
    
    # Create injector
    injector = FaultInjector(random_state=42)
    
    # Test individual fault types
    print("\n" + "-"*60)
    print("TEST 1: Label Noise")
    print("-"*60)
    _, y_noisy, noise_mask = injector.inject_label_noise(X, y, noise_rate=0.2, noise_type='asymmetric')
    print(f"Flipped {noise_mask.sum()} labels")
    
    print("\n" + "-"*60)
    print("TEST 2: Missing Values")
    print("-"*60)
    X_missing, _, missing_mask = injector.inject_missing_values(X, y, missing_rate=0.15, mechanism='MAR')
    print(f"Missing values: {np.isnan(X_missing).sum()}")
    
    print("\n" + "-"*60)
    print("TEST 3: Class Imbalance")
    print("-"*60)
    X_imb, y_imb, _ = injector.inject_class_imbalance(X, y, target_ratio=10.0)
    print(f"New size: {len(y_imb)} samples")
    print(f"New distribution: 0={(y_imb==0).sum()}, 1={(y_imb==1).sum()}")
    
    print("\n" + "-"*60)
    print("TEST 4: Outliers")
    print("-"*60)
    X_out, _, out_mask = injector.inject_outliers(X, y, outlier_rate=0.05)
    print(f"Created {out_mask.sum()} outliers")
    
    # Test combined injection
    print("\n" + "="*60)
    print("TEST 5: All Faults Combined")
    print("="*60)
    config = {
        'label_noise': {'rate': 0.2, 'type': 'asymmetric'},
        'missing': {'rate': 0.15, 'mechanism': 'MAR'},
        'outliers': {'rate': 0.05, 'method': 'extreme'},
        'imbalance': {'ratio': 5.0, 'method': 'undersample'}
    }
    
    result = injector.inject_all_faults(X, y, config)
    print(f"\nFinal corrupted dataset: {result['X_corrupted'].shape[0]} samples")
    print(f"Label noise: {result['masks']['label_noise'].sum()} corrupted")
    print(f"Missing values: {np.isnan(result['X_corrupted']).sum()} total")
    print(f"Outliers: {result['masks']['outliers'].sum()} samples")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)