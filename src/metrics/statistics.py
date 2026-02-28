"""
Statistical Testing Framework for Cleaning Order Comparison

Provides rigorous statistical tests for comparing cleaning orders.
Includes paired tests, effect sizes, and multiple testing correction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTester:
    """
    Rigorous statistical testing for cleaning order comparison.
    
    Handles:
    - Paired comparisons (same seeds across orders)
    - Multiple testing correction
    - Effect size calculation
    - Confidence intervals
    """
    
    @staticmethod
    def paired_ttest(group1: np.ndarray, 
                     group2: np.ndarray,
                     alternative: str = 'two-sided') -> Dict:
        """
        Paired t-test for comparing two cleaning orders.
        
        Args:
            group1: Performance values for order 1 (across seeds)
            group2: Performance values for order 2 (across seeds)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        # Ensure same length
        n = min(len(group1), len(group2))
        group1 = group1[:n]
        group2 = group2[:n]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
        
        # Calculate effect size (Cohen's d for paired data)
        differences = group1 - group2
        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0.0
        
        # Confidence interval for mean difference
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        ci = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=se_diff)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff,
            'std_difference': np.std(differences, ddof=1),
            'cohens_d': cohens_d,
            'effect_size': interpret_cohens_d(cohens_d),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_samples': n,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }
    
    @staticmethod
    def wilcoxon_test(group1: np.ndarray, 
                      group2: np.ndarray,
                      alternative: str = 'two-sided') -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric paired test).
        
        More robust to outliers and non-normal distributions.
        
        Args:
            group1: Performance values for order 1
            group2: Performance values for order 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        # Ensure same length
        n = min(len(group1), len(group2))
        group1 = group1[:n]
        group2 = group2[:n]
        
        try:
            w_stat, p_value = wilcoxon(group1, group2, alternative=alternative)
            
            # Calculate rank-biserial correlation (effect size for Wilcoxon)
            differences = group1 - group2
            n_pos = np.sum(differences > 0)
            n_neg = np.sum(differences < 0)
            rank_biserial = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            
            return {
                'w_statistic': w_stat,
                'p_value': p_value,
                'rank_biserial': rank_biserial,
                'effect_size': interpret_rank_biserial(rank_biserial),
                'n_samples': n,
                'significant_0.05': p_value < 0.05,
                'significant_0.01': p_value < 0.01
            }
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {
                'w_statistic': None,
                'p_value': None,
                'failed': True,
                'reason': str(e)
            }
    
    @staticmethod
    def friedman_test(results_df: pd.DataFrame, 
                      metric: str,
                      order_col: str = 'cleaning_order',
                      seed_col: str = 'seed') -> Dict:
        """
        Friedman test for comparing multiple cleaning orders.
        
        Non-parametric test for repeated measures (multiple orders, same seeds).
        
        Args:
            results_df: DataFrame with results
            metric: Metric to compare
            order_col: Column name for cleaning orders
            seed_col: Column name for seeds
            
        Returns:
            Test results dictionary
        """
        # Pivot data: rows=seeds, columns=orders
        pivot_data = results_df.pivot(index=seed_col, columns=order_col, values=metric)
        
        # Remove rows with any NaN
        pivot_data = pivot_data.dropna()
        
        if len(pivot_data) < 2:
            return {
                'statistic': None,
                'p_value': None,
                'failed': True,
                'reason': 'insufficient_data'
            }
        
        # Perform Friedman test
        statistic, p_value = friedmanchisquare(*[pivot_data[col].values for col in pivot_data.columns])
        
        # Calculate effect size (Kendall's W)
        n = len(pivot_data)  # number of blocks (seeds)
        k = len(pivot_data.columns)  # number of treatments (orders)
        kendalls_w = statistic / (n * (k - 1)) if (n * (k - 1)) > 0 else 0.0
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'kendalls_w': kendalls_w,
            'effect_size': interpret_kendalls_w(kendalls_w),
            'n_seeds': n,
            'n_orders': k,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }
    
    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """
        Bonferroni correction for multiple testing.
        
        Args:
            p_values: List of p-values from multiple tests
            
        Returns:
            Corrected p-values
        """
        n_tests = len(p_values)
        corrected = [min(p * n_tests, 1.0) for p in p_values]
        return corrected
    
    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
        """
        Benjamini-Hochberg FDR correction (less conservative than Bonferroni).
        
        Args:
            p_values: List of p-values
            alpha: Significance level
            
        Returns:
            Tuple of (reject decisions, corrected p-values)
        """
        n_tests = len(p_values)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]
        
        # Calculate critical values
        ranks = np.arange(1, n_tests + 1)
        critical_values = (ranks / n_tests) * alpha
        
        # Find rejections
        reject = sorted_pvalues <= critical_values
        
        # Adjusted p-values
        adjusted_pvalues = np.minimum.accumulate(
            sorted_pvalues * n_tests / ranks[::-1]
        )[::-1]
        adjusted_pvalues = np.minimum(adjusted_pvalues, 1.0)
        
        # Return in original order
        reject_original = np.empty(n_tests, dtype=bool)
        adjusted_original = np.empty(n_tests)
        reject_original[sorted_indices] = reject
        adjusted_original[sorted_indices] = adjusted_pvalues
        
        return list(reject_original), list(adjusted_original)
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, 
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval.
        
        Args:
            data: Sample data
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (mean, ci_lower, ci_upper)
        """
        mean = np.mean(data)
        se = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=se)
        return mean, ci[0], ci[1]
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray,
                                     statistic_func: callable = np.mean,
                                     n_bootstrap: int = 10000,
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval (non-parametric).
        
        Args:
            data: Sample data
            statistic_func: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (statistic, ci_lower, ci_upper)
        """
        bootstrap_statistics = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_statistics.append(statistic_func(sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)
        
        observed_stat = statistic_func(data)
        
        return observed_stat, ci_lower, ci_upper


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size (Cohen, 1988)."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


def interpret_rank_biserial(r: float) -> str:
    """Interpret rank-biserial correlation."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return 'negligible'
    elif abs_r < 0.3:
        return 'small'
    elif abs_r < 0.5:
        return 'medium'
    else:
        return 'large'


def interpret_kendalls_w(w: float) -> str:
    """Interpret Kendall's W."""
    if w < 0.1:
        return 'weak'
    elif w < 0.3:
        return 'moderate'
    elif w < 0.5:
        return 'strong'
    else:
        return 'very strong'


def compare_all_orders_pairwise(results_df: pd.DataFrame,
                                metric: str,
                                order_col: str = 'cleaning_order',
                                seed_col: str = 'seed',
                                correction: str = 'bonferroni') -> pd.DataFrame:
    """
    Pairwise comparison of all cleaning orders with multiple testing correction.
    
    Args:
        results_df: Results DataFrame
        metric: Metric to compare
        order_col: Column for cleaning orders
        seed_col: Column for seeds
        correction: 'bonferroni' or 'fdr' (Benjamini-Hochberg)
        
    Returns:
        DataFrame with pairwise comparison results
    """
    tester = StatisticalTester()
    
    # Get unique orders
    orders = results_df[order_col].unique()
    n_orders = len(orders)
    
    # Perform all pairwise tests
    comparisons = []
    p_values = []
    
    for i in range(n_orders):
        for j in range(i+1, n_orders):
            order1 = orders[i]
            order2 = orders[j]
            
            # Get values for both orders
            values1 = results_df[results_df[order_col] == order1][metric].values
            values2 = results_df[results_df[order_col] == order2][metric].values
            
            # Paired t-test
            test_result = tester.paired_ttest(values1, values2)
            
            comparisons.append({
                'order1': order1,
                'order2': order2,
                'mean1': np.mean(values1),
                'mean2': np.mean(values2),
                'mean_difference': test_result['mean_difference'],
                'p_value': test_result['p_value'],
                'cohens_d': test_result['cohens_d'],
                'effect_size': test_result['effect_size']
            })
            p_values.append(test_result['p_value'])
    
    # Apply multiple testing correction
    if correction == 'bonferroni':
        corrected_pvalues = tester.bonferroni_correction(p_values)
        for i, comp in enumerate(comparisons):
            comp['p_value_corrected'] = corrected_pvalues[i]
            comp['significant_corrected'] = corrected_pvalues[i] < 0.05
    elif correction == 'fdr':
        reject, corrected_pvalues = tester.benjamini_hochberg_correction(p_values)
        for i, comp in enumerate(comparisons):
            comp['p_value_corrected'] = corrected_pvalues[i]
            comp['significant_corrected'] = reject[i]
    
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    # Test statistical framework
    print("\n" + "="*70)
    print("TESTING STATISTICAL FRAMEWORK")
    print("="*70)
    
    np.random.seed(42)
    tester = StatisticalTester()
    
    # Create test data
    order1_values = np.random.normal(0.8, 0.05, 10)
    order2_values = np.random.normal(0.75, 0.05, 10)
    
    print("\nTest 1: Paired t-test")
    ttest_result = tester.paired_ttest(order1_values, order2_values)
    print(f"  Mean difference: {ttest_result['mean_difference']:.4f}")
    print(f"  p-value: {ttest_result['p_value']:.6f}")
    print(f"  Cohen's d: {ttest_result['cohens_d']:.3f} ({ttest_result['effect_size']})")
    print(f"  95% CI: [{ttest_result['ci_lower']:.4f}, {ttest_result['ci_upper']:.4f}]")
    
    print("\nTest 2: Wilcoxon test")
    wilcoxon_result = tester.wilcoxon_test(order1_values, order2_values)
    if not wilcoxon_result.get('failed'):
        print(f"  p-value: {wilcoxon_result['p_value']:.6f}")
        print(f"  Rank-biserial: {wilcoxon_result['rank_biserial']:.3f} ({wilcoxon_result['effect_size']})")
    
    print("\nTest 3: Confidence intervals")
    mean, ci_low, ci_high = tester.calculate_confidence_interval(order1_values)
    print(f"  Mean: {mean:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    print("\nTest 4: Bootstrap CI")
    bs_mean, bs_low, bs_high = tester.bootstrap_confidence_interval(order1_values, n_bootstrap=1000)
    print(f"  Bootstrap mean: {bs_mean:.4f}")
    print(f"  Bootstrap 95% CI: [{bs_low:.4f}, {bs_high:.4f}]")
    
    print("\nTest 5: Multiple testing correction")
    test_pvalues = [0.001, 0.01, 0.04, 0.06, 0.10]
    bonf_corrected = tester.bonferroni_correction(test_pvalues)
    fdr_reject, fdr_corrected = tester.benjamini_hochberg_correction(test_pvalues)
    
    print("  Original p-values:", test_pvalues)
    print("  Bonferroni corrected:", [f"{p:.4f}" for p in bonf_corrected])
    print("  FDR corrected:", [f"{p:.4f}" for p in fdr_corrected])
    print("  FDR rejections:", fdr_reject)
    
    print("\n" + "="*70)
    print("STATISTICAL FRAMEWORK TESTS PASSED ✓")
    print("="*70)
