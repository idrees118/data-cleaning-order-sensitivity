import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Nominal columns stored as numbers in the source file, so dtype detection misses them.
CREDIT_DEFAULT_NOMINAL = ['SEX', 'EDUCATION', 'MARRIAGE']


def _encode_categoricals(df: pd.DataFrame):
    """Label-encode text columns. Returns the frame and the indices of encoded columns."""
    le = LabelEncoder()
    cat_names = list(df.select_dtypes(include=['object', 'category']).columns)
    for col in cat_names:
        df[col] = le.fit_transform(df[col].astype(str))
    cat_idx = [df.columns.get_loc(c) for c in cat_names]
    return df, cat_idx


def load_pima() -> dict:
    fp = DATA_DIR / "pima.csv"
    df = pd.read_csv(fp, header=None)
    if pd.to_numeric(df.iloc[0], errors='coerce').notna().all():
        # header-less file: every row is data (pandas would otherwise eat row 1)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].astype(int).values
    else:
        df = pd.read_csv(fp)
        target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
        X = df.drop(target, axis=1).values.astype(float)
        y = df[target].astype(int).values
    return {'X': X, 'y': y, 'name': 'pima', 'n_features': X.shape[1], 'cat_cols': []}


def load_adult() -> dict:
    fp = DATA_DIR / "adult.csv"
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(fp, names=cols, skipinitialspace=True, na_values='?')
    df = df.dropna().reset_index(drop=True)
    y = (df['income'].str.strip() == '>50K').astype(int).values
    X_df, cat_idx = _encode_categoricals(df.drop('income', axis=1))
    return {'X': X_df.values.astype(float), 'y': y, 'name': 'adult',
            'n_features': X_df.shape[1], 'cat_cols': cat_idx}


def load_credit_default() -> dict:
    fp = DATA_DIR / "credit_default.xls"
    df = pd.read_excel(fp, header=1)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    target_cols = [c for c in df.columns if 'default' in c.lower()]
    target = target_cols[0] if target_cols else df.columns[-1]
    y = df[target].astype(int).values
    X_df, cat_idx = _encode_categoricals(df.drop(target, axis=1))
    upper = {str(c).upper(): i for i, c in enumerate(X_df.columns)}
    missing = [c for c in CREDIT_DEFAULT_NOMINAL if c not in upper]
    if missing:
        raise KeyError(f"credit_default: nominal columns not found: {missing}; columns are {list(X_df.columns)}")
    cat_idx = sorted(set(cat_idx) | {upper[c] for c in CREDIT_DEFAULT_NOMINAL})
    return {'X': X_df.values.astype(float), 'y': y, 'name': 'credit_default',
            'n_features': X_df.shape[1], 'cat_cols': cat_idx}


def load_breast_cancer() -> dict:
    from sklearn.datasets import load_breast_cancer as _load
    data = _load()
    return {'X': data.data.astype(float), 'y': data.target, 'name': 'breast_cancer',
            'n_features': data.data.shape[1], 'cat_cols': []}


def load_spambase() -> dict:
    fp = DATA_DIR / "spambase.csv"
    df = pd.read_csv(fp, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].astype(int).values
    return {'X': X, 'y': y, 'name': 'spambase', 'n_features': X.shape[1], 'cat_cols': []}


def load_ionosphere() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml('ionosphere', version=1, as_frame=False, parser='auto')
    y = (d.target == 'g').astype(int)
    return {'X': d.data.astype(float), 'y': y, 'name': 'ionosphere',
            'n_features': d.data.shape[1], 'cat_cols': []}


def load_banknote() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml('banknote-authentication', version=1, as_frame=False, parser='auto')
    y = LabelEncoder().fit_transform(d.target)
    return {'X': d.data.astype(float), 'y': y, 'name': 'banknote',
            'n_features': d.data.shape[1], 'cat_cols': []}


LOADERS = {
    'pima':           load_pima,
    'adult':          load_adult,
    'credit_default': load_credit_default,
    'breast_cancer':  load_breast_cancer,
    'spambase':       load_spambase,
    'ionosphere':     load_ionosphere,
    'banknote':       load_banknote,
}


def load_dataset(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS)}")
    return LOADERS[name]()


# ---------------------------------------------------------------------------
# Additional benchmarks from OpenML. Each is downloaded once and cached as CSV
# in data/raw/, so later runs (and the archived code) do not need the network.
# The target is the last column; text features are label-encoded and reported
# in 'cat_cols'.
# ---------------------------------------------------------------------------

def _load_openml_cached(data_id: int, cache_name: str) -> dict:
    fp = DATA_DIR / f"{cache_name}.csv"
    if not fp.exists():
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        d.frame.to_csv(fp, index=False)
        print(f"[loaders] cached OpenML '{d.details['name']}' (id {data_id}) as {fp.name}")
    df = pd.read_csv(fp)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str))
    X_df, cat_idx = _encode_categoricals(df.iloc[:, :-1].copy())
    return {'X': X_df.values.astype(float), 'y': y, 'name': cache_name,
            'n_features': X_df.shape[1], 'cat_cols': cat_idx}


def load_credit_g() -> dict:
    return _load_openml_cached(31, 'credit_g')


def load_qsar_biodeg() -> dict:
    return _load_openml_cached(1494, 'qsar_biodeg')


def load_telescope() -> dict:
    return _load_openml_cached(1120, 'telescope')


def load_ozone() -> dict:
    return _load_openml_cached(1487, 'ozone')


def load_kc1() -> dict:
    return _load_openml_cached(1067, 'kc1')


LOADERS.update({
    'credit_g':    load_credit_g,
    'qsar_biodeg': load_qsar_biodeg,
    'telescope':   load_telescope,
    'ozone':       load_ozone,
    'kc1':         load_kc1,
})
