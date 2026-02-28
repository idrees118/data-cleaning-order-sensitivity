"""
Dataset loading utilities for data cleaning order experiments.
Handles loading and basic preprocessing of benchmark datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and basic preprocessing of benchmark datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_adult(self) -> dict:
        """
        Load Adult Income dataset.
        Returns: dict with keys 'X', 'y', 'feature_names', 'task'
        """
        logger.info("Loading Adult Income dataset...")
        
        filepath = self.data_dir / "adult.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Adult dataset not found at {filepath}")
        
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        df = pd.read_csv(filepath, names=column_names, skipinitialspace=True, 
                        na_values='?', low_memory=False)
        
        # Remove rows with missing values (we'll inject them artificially later)
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('income', axis=1)
        y = (df['income'] == '>50K').astype(int)
        
        # Encode categorical features
        X_encoded = self._encode_features(X)
        
        # Convert to numpy array to avoid indexing issues in cleaning pipeline
        X_array = X_encoded.values
        
        logger.info(f"Adult: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"Adult: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': X_encoded.columns.tolist(),
            'task': 'binary_classification',
            'original_features': X.columns.tolist()
        }
    
    def load_credit_default(self) -> dict:
        """
        Load Credit Default dataset.
        Returns: dict with keys 'X', 'y', 'feature_names', 'task'
        """
        logger.info("Loading Credit Default dataset...")
        
        filepath = self.data_dir / "credit_default.xls"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Credit Default dataset not found at {filepath}")
        
        # Read Excel file, skip first row (metadata)
        df = pd.read_excel(filepath, header=1)
        
        # Remove ID column if present
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        # Target is usually 'default payment next month' or similar
        target_col = [col for col in df.columns if 'default' in col.lower()]
        if not target_col:
            # Assume last column is target
            target_col = df.columns[-1]
        else:
            target_col = target_col[0]
        
        X = df.drop(target_col, axis=1)
        y = df[target_col].astype(int)
        
        # Encode categorical features if any
        X_encoded = self._encode_features(X)
        
        # Convert to numpy array to avoid indexing issues
        X_array = X_encoded.values
        
        logger.info(f"Credit Default: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"Credit Default: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': X_encoded.columns.tolist(),
            'task': 'binary_classification',
            'original_features': X.columns.tolist()
        }
    
    def load_pima(self) -> dict:
        """
        Load PIMA Diabetes dataset.
        Returns: dict with keys 'X', 'y', 'feature_names', 'task'
        """
        logger.info("Loading PIMA Diabetes dataset...")
        
        filepath = self.data_dir / "pima.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"PIMA dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Target is usually 'Outcome' or last column
        if 'Outcome' in df.columns:
            target_col = 'Outcome'
        else:
            target_col = df.columns[-1]
        
        X = df.drop(target_col, axis=1)
        y = df[target_col].astype(int)
        
        logger.info(f"PIMA: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"PIMA: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X.values if isinstance(X, pd.DataFrame) else X,
            'y': y.values,
            'feature_names': X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])],
            'task': 'binary_classification',
            'original_features': X.columns.tolist() if isinstance(X, pd.DataFrame) else []
        }
    
    def load_home_credit(self, sample_size: int = 50000, random_state: int = 42) -> dict:
        """
        Load Home Credit dataset (sampled for computational efficiency).
        
        Args:
            sample_size: Number of samples to use (default 50000 for speed)
            random_state: Random seed for reproducibility
            
        Returns: dict with keys 'X', 'y', 'feature_names', 'task'
        """
        logger.info("Loading Home Credit dataset...")
        
        filepath = self.data_dir / "home_credit_train.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Home Credit dataset not found at {filepath}")
        
        # Read with sampling for efficiency
        df = pd.read_csv(filepath)
        
        # Sample if dataset is too large
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} from {len(df)} total samples")
            df = df.sample(n=sample_size, random_state=random_state)
        
        # Separate target
        y = df['TARGET'].astype(int)
        X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)  # Drop ID and target
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        logger.info(f"Found {len(categorical_cols)} categorical columns")
        
        # Encode categorical features
        X_encoded = self._encode_features(X)
        
        # Handle any remaining missing values (Home Credit has many)
        X_encoded = X_encoded.fillna(X_encoded.median())
        
        # Convert to numpy array to avoid indexing issues
        X_array = X_encoded.values
        
        logger.info(f"Home Credit: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"Home Credit: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': X_encoded.columns.tolist(),
            'task': 'binary_classification',
            'original_features': X.columns.tolist()
        }
    
    def load_german_credit(self) -> dict:
        """
        Load German Credit dataset from UCI.
        Space-separated file, no header, target is last column (1=good, 2=bad).
        """
        logger.info("Loading German Credit dataset...")
        
        filepath = self.data_dir / "german_credit.data"
        
        if not filepath.exists():
            raise FileNotFoundError(f"German Credit not found at {filepath}")
        
        # Read space-separated file with no header
        df = pd.read_csv(filepath, sep=' ', header=None)
        
        # Last column is target: 1=good credit, 2=bad credit
        # Convert to binary: 0=good, 1=bad
        y = (df[df.columns[-1]] == 2).astype(int)
        X = df.drop(df.columns[-1], axis=1)
        
        # Encode categorical features (marked as A11, A12, etc.)
        X_encoded = self._encode_features(X)
        X_array = X_encoded.values.astype(np.float64)
        
        logger.info(f"German Credit: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"German Credit: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': [f'feature_{i}' for i in range(X_array.shape[1])],
            'task': 'binary_classification',
            'original_features': [f'feature_{i}' for i in range(X_array.shape[1])]
        }
    
    def load_spambase(self) -> dict:
        """
        Load Spambase dataset from UCI.
        CSV file with no header, all numeric, last column is target (0/1).
        """
        logger.info("Loading Spambase dataset...")
        
        filepath = self.data_dir / "spambase.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spambase not found at {filepath}")
        
        # Read CSV with no header
        df = pd.read_csv(filepath, header=None)
        
        # Last column (57) is target: 0=not spam, 1=spam
        y = df[df.columns[-1]].astype(int)
        X = df.drop(df.columns[-1], axis=1)
        
        # All features are already numeric, just convert to float64
        X_array = X.values.astype(np.float64)
        
        logger.info(f"Spambase: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"Spambase: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': [f'feature_{i}' for i in range(X_array.shape[1])],
            'task': 'binary_classification',
            'original_features': [f'feature_{i}' for i in range(X_array.shape[1])]
        }
    
    def load_bank_marketing(self) -> dict:
        """
        Load Bank Marketing dataset from UCI.
        CSV file with semicolon separator, has header, target column is 'y' (yes/no).
        """
        logger.info("Loading Bank Marketing dataset...")
        
        filepath = self.data_dir / "bank_marketing.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Bank Marketing not found at {filepath}")
        
        # Read CSV with semicolon separator
        df = pd.read_csv(filepath, sep=';')
        
        # Target column is 'y': 'yes'=1, 'no'=0
        y = (df['y'] == 'yes').astype(int)
        X = df.drop('y', axis=1)
        
        # Encode categorical features
        X_encoded = self._encode_features(X)
        X_array = X_encoded.values.astype(np.float64)
        
        logger.info(f"Bank Marketing: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        logger.info(f"Bank Marketing: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X_array,
            'y': y.values,
            'feature_names': X_encoded.columns.tolist(),
            'task': 'binary_classification',
            'original_features': X.columns.tolist()
        }
    
    def load_breast_cancer(self) -> dict:
        """
        Load Breast Cancer Wisconsin dataset from sklearn.
        Returns: dict with keys 'X', 'y', 'feature_names', 'task'
        """
        logger.info("Loading Breast Cancer Wisconsin dataset...")
        
        from sklearn.datasets import load_breast_cancer as load_bc_sklearn
        
        data = load_bc_sklearn()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)
        
        logger.info(f"Breast Cancer: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Breast Cancer: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return {
            'X': X,
            'y': y,
            'feature_names': list(data.feature_names),
            'task': 'binary_classification',
            'original_features': list(data.feature_names)
        }
    
    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        For multi-categorical, could use One-Hot, but Label Encoding is simpler for trees.
        """
        X_encoded = X.copy()
        
        # Identify categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        # Label encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def load_all_datasets(self) -> dict:
        """
        Load all 4 benchmark datasets.
        Returns: dict mapping dataset names to data dicts
        """
        datasets = {}
        
        try:
            datasets['adult'] = self.load_adult()
        except Exception as e:
            logger.error(f"Failed to load Adult dataset: {e}")
        
        try:
            datasets['credit_default'] = self.load_credit_default()
        except Exception as e:
            logger.error(f"Failed to load Credit Default dataset: {e}")
        
        try:
            datasets['pima'] = self.load_pima()
        except Exception as e:
            logger.error(f"Failed to load PIMA dataset: {e}")
        
        try:
            datasets['home_credit'] = self.load_home_credit()
        except Exception as e:
            logger.error(f"Failed to load Home Credit dataset: {e}")
        
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets


# Convenience function for quick loading
def load_dataset(name: str, data_dir: str = "data/raw") -> dict:
    """
    Load a single dataset by name.
    
    Args:
        name: One of ['adult', 'credit_default', 'pima', 'home_credit', 
                      'breast_cancer', 'german_credit', 'spambase', 'bank_marketing']
        data_dir: Path to raw data directory
        
    Returns: Dataset dictionary with X, y, feature_names, task
    """
    loader = DatasetLoader(data_dir)
    
    if name == 'adult':
        return loader.load_adult()
    elif name == 'credit_default':
        return loader.load_credit_default()
    elif name == 'pima':
        return loader.load_pima()
    elif name == 'home_credit':
        return loader.load_home_credit()
    elif name == 'breast_cancer':
        return loader.load_breast_cancer()
    elif name == 'german_credit':
        return loader.load_german_credit()
    elif name == 'spambase':
        return loader.load_spambase()
    elif name == 'bank_marketing':
        return loader.load_bank_marketing()
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from ['adult', 'credit_default', 'pima', 'home_credit', 'breast_cancer', 'german_credit', 'spambase', 'bank_marketing']")


if __name__ == "__main__":
    # Test loading
    loader = DatasetLoader()
    datasets = loader.load_all_datasets()
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for name, data in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Samples: {len(data['y'])}")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Class 0: {(data['y']==0).sum()} ({(data['y']==0).sum()/len(data['y'])*100:.1f}%)")
        print(f"  Class 1: {(data['y']==1).sum()} ({(data['y']==1).sum()/len(data['y'])*100:.1f}%)")
        print(f"  Imbalance ratio: {(data['y']==0).sum() / (data['y']==1).sum():.2f}:1")
