import numpy as np
from typing import Tuple, Dict, List, Optional, Sequence
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, SMOTENC


class DataCleaner:
    """
    Four standard data cleaning operations used in the permutation study.

    Operations:
        - imputation   : KNN imputation (k=5) with per-column standardisation;
                         categorical columns take the most frequent observed value
        - outliers     : Isolation Forest (contamination=0.05) with RobustScaler
        - label_noise  : CV-based Confident Learning, threshold=0.7, relabelling mode
        - balance      : SMOTE (k=5, target ratio=0.8); SMOTENC when categorical
                         columns are present

    cat_cols: indices of integer-encoded nominal columns. With cat_cols=None or []
    every method behaves exactly as the round-1/2 version (numerically identical).
    """

    def __init__(self, random_state: int = 42, cat_cols: Optional[Sequence[int]] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.cat_cols = sorted(int(c) for c in (cat_cols or []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, y

    def _num_cols(self, n_features: int) -> List[int]:
        cats = set(self.cat_cols)
        return [c for c in range(n_features) if c not in cats]

    def _temp_fill(self, X_fit: np.ndarray, X_apply: Optional[np.ndarray] = None):
        """
        Temporary fill used only for scoring/fitting inside a step:
        median for numeric columns, most frequent value for categorical columns.
        Fitted on X_fit, applied to X_apply (defaults to X_fit).
        """
        if X_apply is None:
            X_apply = X_fit
        if not self.cat_cols:
            imp = SimpleImputer(strategy='median')
            imp.fit(X_fit)
            return imp.transform(X_apply)
        out = X_apply.copy()
        num = self._num_cols(X_fit.shape[1])
        if num:
            imp_n = SimpleImputer(strategy='median').fit(X_fit[:, num])
            out[:, num] = imp_n.transform(X_apply[:, num])
        imp_c = SimpleImputer(strategy='most_frequent').fit(X_fit[:, self.cat_cols])
        out[:, self.cat_cols] = imp_c.transform(X_apply[:, self.cat_cols])
        return out

    # ------------------------------------------------------------------
    # 1. Missing value imputation
    # ------------------------------------------------------------------

    def impute(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        KNN imputation (k=5) with per-column StandardScaler fit on observed values.
        Categorical columns: most frequent observed value (KNN would average codes).
        """
        X, y = self._to_numpy(X, y)

        if not np.isnan(X).any():
            return X.copy(), y.copy(), {'n_imputed': 0}

        n_neighbors = min(5, max(1, len(y) - 1))
        scalers = {}
        X_scaled = X.copy()

        for col in range(X.shape[1]):
            valid = ~np.isnan(X[:, col])
            if valid.sum() > 1:
                sc = StandardScaler()
                X_scaled[valid, col] = sc.fit_transform(X[valid, col].reshape(-1, 1)).ravel()
                scalers[col] = sc

        imputer = KNNImputer(n_neighbors=n_neighbors)
        X_imp_scaled = imputer.fit_transform(X_scaled)

        X_imp = X_imp_scaled.copy()
        for col, sc in scalers.items():
            X_imp[:, col] = sc.inverse_transform(X_imp_scaled[:, col].reshape(-1, 1)).ravel()

        # categorical columns: keep observed codes exactly (the scale round trip adds
        # floating-point error) and fill missing cells with the most frequent observed code
        for col in self.cat_cols:
            miss = np.isnan(X[:, col])
            X_imp[~miss, col] = X[~miss, col]
            if miss.any():
                observed = X[~miss, col]
                if observed.size:
                    vals, cnts = np.unique(observed, return_counts=True)
                    X_imp[miss, col] = vals[cnts.argmax()]

        n_imputed = int(np.isnan(X).sum()) - int(np.isnan(X_imp).sum())
        return X_imp, y.copy(), {'n_imputed': n_imputed}

    # ------------------------------------------------------------------
    # 2. Outlier removal
    # ------------------------------------------------------------------

    def remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Isolation Forest (contamination=0.05, n_estimators=100) with RobustScaler.
        NaN values receive a temporary fill before scoring; flagged rows are dropped.
        """
        X, y = self._to_numpy(X, y)

        X_score = self._temp_fill(X) if np.isnan(X).any() else X

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_score)

        iso = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=self.random_state,
            n_jobs=-1
        )
        preds = iso.fit_predict(X_scaled)
        keep = preds == 1

        return X[keep].copy(), y[keep].copy(), {'n_removed': int((~keep).sum())}

    # ------------------------------------------------------------------
    # 3. Label noise correction (Confident Learning approach)
    # ------------------------------------------------------------------

    def correct_label_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        5-fold CV with RandomForest for out-of-fold probabilities; labels where the
        predicted class differs with confidence >= 0.7 are relabelled.
        The temporary imputer is fitted on the training folds only.
        """
        X, y = self._to_numpy(X, y)

        has_nan = bool(np.isnan(X).any())

        n_samples = len(y)
        n_classes = len(np.unique(y))
        cv_proba = np.zeros((n_samples, n_classes))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            if has_nan:
                X_val = self._temp_fill(X_tr, X_val)
                X_tr = self._temp_fill(X_tr)
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state + fold,
                n_jobs=-1
            )
            rf.fit(X_tr, y[train_idx])
            cv_proba[val_idx] = rf.predict_proba(X_val)

        predicted = cv_proba.argmax(axis=1)
        confidence = cv_proba[np.arange(n_samples), predicted]

        flagged = (predicted != y) & (confidence >= 0.7)
        n_relabelled = int(flagged.sum())

        y_corrected = y.copy()
        y_corrected[flagged] = predicted[flagged]

        return X.copy(), y_corrected, {'n_relabelled': n_relabelled}

    # ------------------------------------------------------------------
    # 4. Class balancing
    # ------------------------------------------------------------------

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        SMOTE (k=5, sampling_strategy=0.8); SMOTENC when categorical columns exist,
        so synthetic rows take real category codes instead of interpolated ones.
        If NaN present, a temporary fill is used and original rows are kept as they came in.
        """
        X, y = self._to_numpy(X, y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X.copy(), y.copy(), {'skipped': True, 'reason': 'single_class'}

        n_min = int(counts.min())
        n_maj = int(counts.max())
        current_ratio = n_min / n_maj

        if current_ratio >= 0.8 * 0.95:
            return X.copy(), y.copy(), {'skipped': True, 'reason': 'already_balanced'}

        if n_min < 2:
            return X.copy(), y.copy(), {'skipped': True, 'reason': 'too_few_minority'}

        has_nan = bool(np.isnan(X).any())
        X_smote = self._temp_fill(X) if has_nan else X

        k_neighbors = min(5, n_min - 1)
        if self.cat_cols:
            sampler = SMOTENC(
                categorical_features=self.cat_cols,
                sampling_strategy=0.8,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
        else:
            sampler = SMOTE(
                sampling_strategy=0.8,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
        X_bal, y_bal = sampler.fit_resample(X_smote, y)

        if has_nan:
            X_bal = np.vstack([X, X_bal[len(y):]])

        return X_bal, y_bal, {'skipped': False, 'n_synthetic': len(y_bal) - len(y)}

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    OPERATION_MAP = {
        'imputation':  'impute',
        'outliers':    'remove_outliers',
        'label_noise': 'correct_label_noise',
        'balance':     'balance',
    }

    def apply_sequence(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       sequence: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        X, y = self._to_numpy(X, y)
        log = []
        for op in sequence:
            if op not in self.OPERATION_MAP:
                raise ValueError(f"Unknown operation '{op}'. "
                                 f"Valid: {list(self.OPERATION_MAP)}")
            method = getattr(self, self.OPERATION_MAP[op])
            X, y, meta = method(X, y)
            log.append({'operation': op, **meta})
        return X, y, log
