import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import warnings

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def build_model(name: str, random_state: int = 42):
    """
    Build classifier with fixed hyperparameters as specified in paper (Table 2).

    Random Forest  : n_estimators=100, max_depth=10
    Logistic Reg.  : solver=lbfgs, max_iter=1000
    XGBoost        : n_estimators=100, max_depth=10, learning_rate=0.1
    """
    if name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=random_state
        )
    elif name == 'logistic_regression':
        return Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            ))
        ])
    elif name == 'xgboost':
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,          # Paper Table 2: max_depth=10
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss',
            use_label_encoder=False
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Valid: random_forest, logistic_regression, xgboost")


def train_and_evaluate(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str,
                       random_state: int = 42) -> Dict:
    """
    Train model on (X_train, y_train) and evaluate on (X_test, y_test).
    Returns balanced_accuracy as primary metric.
    Test set is NEVER modified — only training set is cleaned.
    """
    try:
        model = build_model(model_name, random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_train, y_train)

        converged = True
        if model_name == 'logistic_regression':
            clf = model.named_steps['clf']
            converged = bool(np.max(clf.n_iter_) < clf.max_iter)

        y_pred = model.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        return {
            'balanced_accuracy': bal_acc,
            'converged': converged,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'failed': False
        }

    except Exception as e:
        return {
            'balanced_accuracy': np.nan,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'failed': True,
            'error': str(e)
        }
