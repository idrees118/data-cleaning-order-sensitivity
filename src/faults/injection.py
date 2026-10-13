import numpy as np
from typing import Tuple, Dict, Optional, Sequence


class FaultInjector:
    """Inject controlled data quality faults into training sets only.

    cat_cols: indices of integer-encoded nominal columns. Outlier injection leaves
    these columns unchanged, because mean +/- k*std is not a valid category code.
    With cat_cols=None or [] every method is numerically identical to the
    round-1/2 version, and the random stream is the same in both cases.
    """

    def __init__(self, random_state: int = 42, cat_cols: Optional[Sequence[int]] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.cat_cols = set(int(c) for c in (cat_cols or []))

    def inject_label_noise(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           noise_rate: float = 0.15,
                           noise_type: str = 'asymmetric') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        asymmetric        : 70% of flips from the minority class, 30% from the majority
        symmetric         : flipped rows drawn uniformly, independent of class
        minority_anchored : noise_rate of minority labels plus the same number of majority labels
        """
        y_noisy = y.copy()
        n_samples = len(y)
        n_to_flip = int(n_samples * noise_rate)
        noise_mask = np.zeros(n_samples, dtype=bool)

        if noise_type == 'symmetric':
            flip_idx = self.rng.choice(n_samples, size=n_to_flip, replace=False)
            noise_mask[flip_idx] = True
            y_noisy[flip_idx] = 1 - y[flip_idx]

        elif noise_type == 'asymmetric':
            minority_class = 1 if (y == 1).sum() <= (y == 0).sum() else 0
            majority_class = 1 - minority_class

            minority_idx = np.where(y == minority_class)[0]
            majority_idx = np.where(y == majority_class)[0]

            n_min_flip = min(int(n_to_flip * 0.7), len(minority_idx))
            n_maj_flip = min(n_to_flip - n_min_flip, len(majority_idx))

            flip_min = self.rng.choice(minority_idx, size=n_min_flip, replace=False)
            flip_maj = self.rng.choice(majority_idx, size=n_maj_flip, replace=False)
            flip_idx = np.concatenate([flip_min, flip_maj])

            noise_mask[flip_idx] = True
            y_noisy[flip_idx] = 1 - y[flip_idx]

        elif noise_type == 'minority_anchored':
            minority_class = 1 if (y == 1).sum() <= (y == 0).sum() else 0
            minority_idx = np.where(y == minority_class)[0]
            majority_idx = np.where(y != minority_class)[0]

            n_flip = int(len(minority_idx) * noise_rate)
            flip_min = self.rng.choice(minority_idx, size=n_flip, replace=False)
            flip_maj = self.rng.choice(majority_idx, size=min(n_flip, len(majority_idx)), replace=False)
            flip_idx = np.concatenate([flip_min, flip_maj])

            noise_mask[flip_idx] = True
            y_noisy[flip_idx] = 1 - y[flip_idx]

        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        return X, y_noisy, noise_mask

    def inject_missing_values(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              missing_rate: float = 0.10,
                              mechanism: str = 'MAR') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject missing values under MAR mechanism (missingness depends on observed features).
        """
        X_out = X.copy().astype(float)
        n_samples, n_features = X_out.shape
        missing_mask = np.zeros_like(X_out, dtype=bool)

        n_affected = max(1, n_features // 2)
        feature_idx = self.rng.choice(n_features, size=n_affected, replace=False)

        for feat in feature_idx:
            if mechanism == 'MCAR':
                n_miss = int(n_samples * missing_rate)
                rows = self.rng.choice(n_samples, size=n_miss, replace=False)
            elif mechanism == 'MAR':
                other = [f for f in range(n_features) if f != feat]
                cond_feat = self.rng.choice(other) if other else feat
                threshold = np.nanpercentile(X_out[:, cond_feat], 100 - missing_rate * 100)
                rows = np.where(X_out[:, cond_feat] > threshold)[0]
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")

            missing_mask[rows, feat] = True
            X_out[rows, feat] = np.nan

        return X_out, y, missing_mask

    def inject_outliers(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        outlier_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject extreme outliers (3.5 to 4.5 standard deviations from the mean) into the
        numeric features of 5% of the rows. Categorical columns keep their values.
        The random draws are consumed for every feature, so the outlier rows and the
        numeric outlier values are identical to the version without categorical handling.
        """
        X_out = X.copy().astype(float)
        n_samples, n_features = X_out.shape
        n_outliers = int(n_samples * outlier_rate)

        outlier_idx = self.rng.choice(n_samples, size=n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        outlier_mask[outlier_idx] = True

        col_means = np.nanmean(X_out, axis=0)
        col_stds = np.nanstd(X_out, axis=0)
        col_stds[col_stds == 0] = 1.0

        for idx in outlier_idx:
            for feat in range(n_features):
                sign = 1 if self.rng.rand() > 0.5 else -1
                mag = 3.5 + self.rng.rand()
                if feat in self.cat_cols:
                    continue
                X_out[idx, feat] = col_means[feat] + sign * mag * col_stds[feat]

        return X_out, y, outlier_mask

    def inject_all_faults(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          config: Dict) -> Dict:
        """
        Apply all fault types in sequence. Returns corrupted data and masks.

        Config format:
            {
                'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
                'missing':     {'rate': 0.10, 'mechanism': 'MAR'},
                'outliers':    {'rate': 0.05}
            }
        """
        masks = {}
        X_c, y_c = X.copy(), y.copy()

        if 'label_noise' in config:
            cfg = config['label_noise']
            X_c, y_c, masks['label_noise'] = self.inject_label_noise(
                X_c, y_c,
                noise_rate=cfg.get('rate', 0.15),
                noise_type=cfg.get('type', 'asymmetric')
            )

        if 'missing' in config:
            cfg = config['missing']
            X_c, y_c, masks['missing'] = self.inject_missing_values(
                X_c, y_c,
                missing_rate=cfg.get('rate', 0.10),
                mechanism=cfg.get('mechanism', 'MAR')
            )

        if 'outliers' in config:
            cfg = config['outliers']
            X_c, y_c, masks['outliers'] = self.inject_outliers(
                X_c, y_c,
                outlier_rate=cfg.get('rate', 0.05)
            )

        return {'X_corrupted': X_c, 'y_corrupted': y_c, 'masks': masks}
