"""
Alternative cleaners for the robustness experiments. Same interface as
DataCleaner, so the runner can swap them in without other changes.

DataCleanerB   one method per step from a different family than set A:
               median imputation, Local Outlier Factor, edited nearest
               neighbours (noisy rows are removed, not relabelled), and
               random oversampling. Target ratio and skip rules match SMOTE.
DataCleanerRatio
               set A with a different SMOTE target ratio (ratio ablation).
               With ratio=0.8 it reproduces DataCleaner.balance exactly.

Both accept cat_cols (indices of integer-encoded nominal columns). Categorical
columns are filled with their most frequent value instead of the median, and
the ratio cleaner uses SMOTENC. With cat_cols=None or [] both cleaners are
numerically identical to the round-1/2 versions.
"""
import warnings
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.cleaning.methods import DataCleaner

TARGET_RATIO = 0.8


def _median_filled(X: np.ndarray) -> np.ndarray:
    """Temporary median fill, used only where a method cannot take NaNs."""
    if not np.isnan(X).any():
        return X
    return SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(X)


def _balance_checks(y: np.ndarray, ratio: float):
    """Skip rules shared by every balancing method (same as DataCleaner.balance)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {'skipped': True, 'reason': 'single_class'}, None
    n_min, n_maj = int(counts.min()), int(counts.max())
    if n_min / n_maj >= ratio * 0.95:
        return {'skipped': True, 'reason': 'already_balanced'}, None
    if n_min < 2:
        return {'skipped': True, 'reason': 'too_few_minority'}, None
    return None, (classes[np.argmin(counts)], n_min, n_maj)


class DataCleanerB(DataCleaner):

    def _scoring_fill(self, X: np.ndarray) -> np.ndarray:
        """Temporary fill for LOF and ENN: median, or most frequent value for categorical columns."""
        if not self.cat_cols:
            return _median_filled(X)
        return self._temp_fill(X) if np.isnan(X).any() else X

    def impute(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        X, y = self._to_numpy(X, y)
        n_missing = int(np.isnan(X).sum())
        if n_missing == 0:
            return X.copy(), y.copy(), {'n_imputed': 0}
        X_imp = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(X)
        if self.cat_cols:
            mode = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
            X_imp[:, self.cat_cols] = mode.fit_transform(X[:, self.cat_cols])
        return X_imp, y.copy(), {'n_imputed': n_missing}

    def remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Local Outlier Factor (20 neighbours, 5% expected outliers) on robust-scaled data."""
        X, y = self._to_numpy(X, y)
        X_scaled = RobustScaler().fit_transform(self._scoring_fill(X))
        lof = LocalOutlierFactor(n_neighbors=min(20, len(y) - 1), contamination=0.05)
        with warnings.catch_warnings():
            # oversampled duplicates trigger a warning about tied distances; scores are still defined
            warnings.filterwarnings('ignore', message='Duplicate values')
            keep = lof.fit_predict(X_scaled) == 1
        return X[keep].copy(), y[keep].copy(), {'n_removed': int((~keep).sum())}

    def correct_label_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Edited nearest neighbours (Wilson editing): a row is removed when at least
        two of its three nearest neighbours carry a different label. Distances are
        computed on standardised features."""
        X, y = self._to_numpy(X, y)
        if len(np.unique(y)) < 2 or len(y) < 4:
            return X.copy(), y.copy(), {'n_removed_noise': 0}
        Z = StandardScaler().fit_transform(self._scoring_fill(X))
        # kneighbors() without a query excludes each point from its own neighbour list,
        # which stays correct when the data contain exact duplicates
        idx = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(return_distance=False)
        drop = (y[idx] != y[:, None]).sum(axis=1) >= 2
        if len(np.unique(y[~drop])) < 2:              # never wipe out a class
            drop[:] = False
        keep = ~drop
        return X[keep].copy(), y[keep].copy(), {'n_removed_noise': int(drop.sum())}

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Random oversampling of the minority class to the same target ratio as SMOTE.
        Duplicates existing rows, so categorical codes stay valid."""
        X, y = self._to_numpy(X, y)
        skip, info = _balance_checks(y, TARGET_RATIO)
        if skip:
            return X.copy(), y.copy(), skip
        minority, n_min, n_maj = info
        n_new = int(n_maj * TARGET_RATIO - n_min)     # same count imbalanced-learn uses
        rng = np.random.RandomState(self.random_state)
        pick = rng.choice(np.where(y == minority)[0], size=n_new, replace=True)
        return (np.vstack([X, X[pick]]), np.concatenate([y, y[pick]]),
                {'skipped': False, 'n_synthetic': n_new})


class DataCleanerRatio(DataCleaner):

    def __init__(self, random_state: int = 42, ratio: float = TARGET_RATIO,
                 cat_cols: Optional[Sequence[int]] = None):
        super().__init__(random_state=random_state, cat_cols=cat_cols)
        self.ratio = ratio

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """DataCleaner.balance with the target ratio as a parameter."""
        X, y = self._to_numpy(X, y)
        skip, info = _balance_checks(y, self.ratio)
        if skip:
            return X.copy(), y.copy(), skip
        _, n_min, _ = info
        has_nan = bool(np.isnan(X).any())
        X_smote = self._temp_fill(X) if has_nan else X
        k = min(5, n_min - 1)
        if self.cat_cols:
            sampler = SMOTENC(categorical_features=self.cat_cols, sampling_strategy=self.ratio,
                              k_neighbors=k, random_state=self.random_state)
        else:
            sampler = SMOTE(sampling_strategy=self.ratio, k_neighbors=k,
                            random_state=self.random_state)
        X_bal, y_bal = sampler.fit_resample(X_smote, y)
        if has_nan:
            X_bal = np.vstack([X, X_bal[len(y):]])
        return X_bal, y_bal, {'skipped': False, 'n_synthetic': len(y_bal) - len(y)}
