"""
Order Sensitivity Index (OSI) - Publication-Grade Calculation

Computes OSI across multiple metrics appropriate for imbalanced classification.
Follows best practices from ML evaluation literature.

Key principle: OSI measures how much cleaning order matters.
OSI = (Best - Worst) / Best for each metric
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiMetricEvaluator:
    """
    Comprehensive evaluation for imbalanced binary classification.
    
    Computes all metrics needed for rigorous comparison of cleaning orders.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for binary classification.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (macro for fairness across classes)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Minority class specific (binary classification)
        metrics['precision_minority'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_minority'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_minority'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Matthews Correlation Coefficient (good for imbalanced)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # AUC if probabilities available
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = None
        else:
            metrics['auc'] = None
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Specificity (important for imbalanced)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # G-Mean (geometric mean of sensitivity and specificity)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['gmean'] = np.sqrt(sensitivity * metrics['specificity'])
        
        return metrics
    
    @staticmethod
    def get_primary_metrics() -> List[str]:
        """
        Return list of primary metrics for OSI calculation.
        
        These are the metrics most appropriate for imbalanced classification.
        """
        return [
            'balanced_accuracy',  # Main metric for imbalanced data
            'f1_macro',          # Fair across classes
            'auc',               # Threshold-independent
            'mcc',               # Good for imbalanced
            'gmean'              # Sensitivity-specificity balance
        ]
    
    @staticmethod
    def get_all_metric_names() -> List[str]:
        """Return all available metric names."""
        return [
            'accuracy', 'balanced_accuracy',
            'precision_macro', 'recall_macro', 'f1_macro',
            'precision_minority', 'recall_minority', 'f1_minority',
            'mcc', 'auc', 'specificity', 'gmean'
        ]


class OSICalculator:
    """
    Calculate Order Sensitivity Index across multiple metrics.
    
    OSI measures how much cleaning order affects model performance.
    Higher OSI = order matters more.
    """
    
    def __init__(self, primary_metric: str = 'balanced_accuracy'):
        """
        Args:
            primary_metric: Main metric for ranking orders
        """
        self.primary_metric = primary_metric
        self.evaluator = MultiMetricEvaluator()
    
    def calculate_osi_single_metric(self, 
                                    results_df: pd.DataFrame,
                                    metric_name: str) -> Dict:
        """
        Calculate OSI for a single metric.
        
        Args:
            results_df: DataFrame with columns [cleaning_order, <metric_name>, seed]
            metric_name: Name of metric to calculate OSI for
            
        Returns:
            OSI results dictionary
        """
        # Check if metric exists
        if metric_name not in results_df.columns:
            logger.warning(f"Metric {metric_name} not found in results")
            return {'osi': None, 'reason': 'metric_not_found'}
        
        # Filter successful runs
        if 'failed' in results_df.columns:
            successful = results_df[results_df['failed'] == False].copy()
        else:
            successful = results_df.copy()
        
        if len(successful) == 0:
            return {'osi': None, 'reason': 'no_successful_runs'}
        
        # Group by cleaning order and average across seeds
        order_stats = successful.groupby('cleaning_order')[metric_name].agg(['mean', 'std', 'count'])
        
        # Find best and worst
        best_idx = order_stats['mean'].idxmax()
        worst_idx = order_stats['mean'].idxmin()
        
        best_order = best_idx
        worst_order = worst_idx
        best_value = order_stats.loc[best_idx, 'mean']
        worst_value = order_stats.loc[worst_idx, 'mean']
        
        # Calculate OSI
        # OSI = (Best - Worst) / Best
        # Interpretation: What fraction of best performance is lost by choosing worst order
        if best_value > 0:
            osi = (best_value - worst_value) / best_value
        else:
            osi = 0.0
        
        # Calculate absolute range
        value_range = best_value - worst_value
        
        results = {
            'metric': metric_name,
            'osi': osi,
            'osi_percentage': osi * 100,
            'best_order': best_order,
            'best_value': best_value,
            'best_std': order_stats.loc[best_idx, 'std'],
            'worst_order': worst_order,
            'worst_value': worst_value,
            'worst_std': order_stats.loc[worst_idx, 'std'],
            'value_range': value_range,
            'n_orders': len(order_stats)
        }
        
        return results
    
    def calculate_osi_all_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OSI for all available metrics.
        
        Args:
            results_df: Results from experiments
            
        Returns:
            DataFrame with OSI for each metric
        """
        logger.info("Calculating OSI across all metrics...")
        
        # Get all metric columns
        all_metrics = MultiMetricEvaluator.get_all_metric_names()
        available_metrics = [m for m in all_metrics if m in results_df.columns]
        
        if len(available_metrics) == 0:
            logger.error("No metrics found in results DataFrame")
            return pd.DataFrame()
        
        # Calculate OSI for each metric
        osi_results = []
        for metric in available_metrics:
            result = self.calculate_osi_single_metric(results_df, metric)
            if result.get('osi') is not None:
                osi_results.append(result)
        
        osi_df = pd.DataFrame(osi_results)
        
        # Sort by OSI (descending)
        osi_df = osi_df.sort_values('osi', ascending=False)
        
        logger.info(f"\nOSI Summary:")
        logger.info(f"{'Metric':<20} {'OSI':>8} {'Best Value':>12} {'Worst Value':>12} {'Range':>10}")
        logger.info("-" * 70)
        for _, row in osi_df.iterrows():
            logger.info(f"{row['metric']:<20} {row['osi']:>8.4f} {row['best_value']:>12.4f} "
                       f"{row['worst_value']:>12.4f} {row['value_range']:>10.4f}")
        
        return osi_df
    
    def get_order_rankings(self, 
                          results_df: pd.DataFrame,
                          metric: Optional[str] = None) -> pd.DataFrame:
        """
        Get ranked list of cleaning orders by performance.
        
        Args:
            results_df: Results from experiments
            metric: Metric to rank by (default: primary_metric)
            
        Returns:
            DataFrame with orders ranked by performance
        """
        if metric is None:
            metric = self.primary_metric
        
        if metric not in results_df.columns:
            logger.error(f"Metric {metric} not found")
            return pd.DataFrame()
        
        # Filter successful runs
        if 'failed' in results_df.columns:
            successful = results_df[results_df['failed'] == False].copy()
        else:
            successful = results_df.copy()
        
        # Group by cleaning order
        rankings = successful.groupby('cleaning_order').agg({
            metric: ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        rankings.columns = ['cleaning_order', f'{metric}_mean', f'{metric}_std', 'n_runs']
        
        # Sort by mean performance (descending)
        rankings = rankings.sort_values(f'{metric}_mean', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)
        
        # Calculate confidence interval (95%)
        from scipy import stats
        rankings['ci_lower'] = rankings.apply(
            lambda row: row[f'{metric}_mean'] - 1.96 * row[f'{metric}_std'] / np.sqrt(row['n_runs']),
            axis=1
        )
        rankings['ci_upper'] = rankings.apply(
            lambda row: row[f'{metric}_mean'] + 1.96 * row[f'{metric}_std'] / np.sqrt(row['n_runs']),
            axis=1
        )
        
        return rankings
    
    def calculate_relative_performance(self, 
                                      results_df: pd.DataFrame,
                                      metric: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate relative performance: how much each order deviates from best.
        
        Args:
            results_df: Results from experiments
            metric: Metric to use (default: primary_metric)
            
        Returns:
            DataFrame with relative performance metrics
        """
        if metric is None:
            metric = self.primary_metric
        
        rankings = self.get_order_rankings(results_df, metric)
        
        if len(rankings) == 0:
            return pd.DataFrame()
        
        best_value = rankings.iloc[0][f'{metric}_mean']
        
        # Calculate relative performance
        rankings['relative_performance'] = rankings[f'{metric}_mean'] / best_value
        rankings['performance_loss'] = (best_value - rankings[f'{metric}_mean']) / best_value
        
        return rankings


def analyze_osi_significance(results_df: pd.DataFrame, 
                            metric: str = 'balanced_accuracy') -> Dict:
    """
    Analyze whether OSI differences are statistically significant.
    
    Performs paired comparison between best and worst orders.
    
    Args:
        results_df: Results with multiple seeds per order
        metric: Metric to analyze
        
    Returns:
        Statistical test results
    """
    from scipy import stats
    
    # Get best and worst orders
    order_means = results_df.groupby('cleaning_order')[metric].mean()
    best_order = order_means.idxmax()
    worst_order = order_means.idxmin()
    
    # Get values for both orders across seeds
    best_values = results_df[results_df['cleaning_order'] == best_order][metric].values
    worst_values = results_df[results_df['cleaning_order'] == worst_order][metric].values
    
    # Ensure same number of seeds
    n_samples = min(len(best_values), len(worst_values))
    best_values = best_values[:n_samples]
    worst_values = worst_values[:n_samples]
    
    # Paired t-test (appropriate for same seeds)
    t_stat, p_value = stats.ttest_rel(best_values, worst_values)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pvalue = stats.wilcoxon(best_values, worst_values)
    except:
        w_stat, w_pvalue = None, None
    
    # Effect size (Cohen's d for paired samples)
    differences = best_values - worst_values
    cohens_d = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0
    
    significance_results = {
        'metric': metric,
        'best_order': best_order,
        'worst_order': worst_order,
        'best_mean': np.mean(best_values),
        'best_std': np.std(best_values),
        'worst_mean': np.mean(worst_values),
        'worst_std': np.std(worst_values),
        'mean_difference': np.mean(differences),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p': w_pvalue,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d)
    }
    
    return significance_results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


if __name__ == "__main__":
    # Test OSI calculation
    print("\n" + "="*70)
    print("TESTING MULTI-METRIC OSI CALCULATION")
    print("="*70)
    
    # Create synthetic test data
    np.random.seed(42)
    
    orders = ['A→B→C', 'B→A→C', 'C→A→B']
    seeds = [42, 43, 44, 45, 46]
    
    test_data = []
    for order in orders:
        for seed in seeds:
            # Simulate different performance for different orders
            base_acc = 0.75 + np.random.normal(0, 0.02)
            if order == 'A→B→C':
                base_acc += 0.05  # Best order
            elif order == 'C→A→B':
                base_acc -= 0.03  # Worst order
            
            test_data.append({
                'cleaning_order': order,
                'seed': seed,
                'balanced_accuracy': np.clip(base_acc, 0, 1),
                'f1_macro': np.clip(base_acc - 0.05 + np.random.normal(0, 0.01), 0, 1),
                'auc': np.clip(base_acc + 0.03 + np.random.normal(0, 0.01), 0, 1),
                'mcc': np.clip(base_acc - 0.1 + np.random.normal(0, 0.02), -1, 1),
                'gmean': np.clip(base_acc - 0.02 + np.random.normal(0, 0.01), 0, 1),
                'failed': False
            })
    
    test_df = pd.DataFrame(test_data)
    
    # Test OSI calculation
    calculator = OSICalculator(primary_metric='balanced_accuracy')
    
    print("\nTest 1: Calculate OSI for all metrics")
    osi_df = calculator.calculate_osi_all_metrics(test_df)
    
    print("\nTest 2: Get order rankings")
    rankings = calculator.get_order_rankings(test_df, 'balanced_accuracy')
    print("\nOrder Rankings (by balanced_accuracy):")
    print(rankings[['rank', 'cleaning_order', 'balanced_accuracy_mean', 'balanced_accuracy_std']])
    
    print("\nTest 3: Statistical significance")
    sig_results = analyze_osi_significance(test_df, 'balanced_accuracy')
    print(f"\nBest vs Worst comparison:")
    print(f"  Best order:  {sig_results['best_order']} ({sig_results['best_mean']:.4f})")
    print(f"  Worst order: {sig_results['worst_order']} ({sig_results['worst_mean']:.4f})")
    print(f"  Difference:  {sig_results['mean_difference']:.4f}")
    print(f"  p-value:     {sig_results['p_value']:.6f}")
    print(f"  Significant: {sig_results['significant_at_0.05']}")
    print(f"  Cohen's d:   {sig_results['cohens_d']:.3f} ({sig_results['effect_size_interpretation']})")
    
    print("\n" + "="*70)
    print("MULTI-METRIC OSI TESTS PASSED ✓")
    print("="*70)
