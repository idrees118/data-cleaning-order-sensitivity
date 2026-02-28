"""
COMPREHENSIVE 4-DATASET ANALYSIS
Run all datasets: PIMA, Adult, Credit, Breast Cancer
Analyze patterns across datasets for publication
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.publication_runner import ScientificPermutationExperiment
from src.data.loaders import load_dataset
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

def run_single_dataset(dataset_name, ops_list=['imputation', 'label_noise', 'outliers', 'balance']):
    """Run full permutation experiment on one dataset."""
    
    print("\n" + "="*80)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)
    
    # Load data
    data = load_dataset(dataset_name)
    X_clean, y_clean = data['X'], data['y']
    
    print(f"Samples: {len(y_clean)}, Features: {X_clean.shape[1]}")
    print(f"Class distribution: {np.bincount(y_clean.astype(int))}")
    
    # Fault configuration
    fault_config = {
        'label_noise': {'rate': 0.15, 'type': 'asymmetric'},
        'missing': {'rate': 0.10, 'mechanism': 'MAR'},
        'outliers': {'rate': 0.05, 'method': 'extreme'}
    }
    
    # Cleaning parameters
    cleaning_params = {
        'imputation': {'method': 'knn', 'n_neighbors': 5},
        'label_noise': {'method': 'confidence', 'threshold': 0.9, 'mode': 'correct'},
        'outliers': {'method': 'isolation_forest', 'contamination': 0.05},
        'balance': {'method': 'smote', 'target_ratio': 0.8}
    }
    
    # Create experiment
    experiment = ScientificPermutationExperiment(
        dataset_name=dataset_name,
        fault_config=fault_config,
        cleaning_params=cleaning_params,
        models=['random_forest'],
        seeds=[42, 43, 44, 45, 46],
        test_size=0.2,
        primary_metric='balanced_accuracy'
    )
    
    # Override operations
    experiment.cleaning_ops = ops_list
    n_perms = len(list(__import__('itertools').permutations(ops_list)))
    
    print(f"Running {n_perms} permutations × 5 seeds = {n_perms * 5} runs")
    print(f"Estimated time: {n_perms * 5 * 0.5:.0f}-{n_perms * 5 * 1:.0f} minutes")
    
    start = datetime.now()
    
    # Run
    try:
        results_df = experiment.run_full_experiment(X_clean, y_clean)
        analysis = experiment.analyze_results(results_df)
        
        duration = (datetime.now() - start).total_seconds() / 60
        
        print(f"\n✓ Completed in {duration:.1f} minutes")
        
        return {
            'dataset': dataset_name,
            'results_df': results_df,
            'analysis': analysis,
            'duration': duration,
            'success': True
        }
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return {
            'dataset': dataset_name,
            'success': False,
            'error': str(e)
        }


def analyze_cross_dataset(all_results):
    """Analyze patterns across all datasets using proper multi-dataset statistics."""
    
    print("\n\n" + "="*80)
    print("CROSS-DATASET ANALYSIS")
    print("="*80)
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if len(successful_results) < 2:
        print("⚠ Need at least 2 datasets for cross-dataset analysis")
        return None
    
    # ============================================================
    # PART 1: Per-Dataset Summary
    # ============================================================
    summary_data = []
    
    for result in successful_results:
        dataset = result['dataset']
        df = result['results_df']
        
        # Calculate statistics
        order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
        best_order = order_means.idxmax()
        worst_order = order_means.idxmin()
        
        best_samples = df[df['cleaning_order'] == best_order]['balanced_accuracy'].values
        worst_samples = df[df['cleaning_order'] == worst_order]['balanced_accuracy'].values
        
        t_stat, p_value = stats.ttest_rel(best_samples, worst_samples)
        
        diff = best_samples.mean() - worst_samples.mean()
        cohens_d = diff / np.sqrt((best_samples.std()**2 + worst_samples.std()**2) / 2)
        osi = (order_means.max() - order_means.min()) / order_means.max()
        
        # Calculate MCC and recall_minority OSI if available
        mcc_osi = None
        recall_minority_osi = None
        
        if 'mcc' in df.columns:
            mcc_means = df.groupby('cleaning_order')['mcc'].mean()
            mcc_osi = (mcc_means.max() - mcc_means.min()) / mcc_means.max() if mcc_means.max() > 0 else 0
        
        if 'recall_minority' in df.columns:
            recall_means = df.groupby('cleaning_order')['recall_minority'].mean()
            recall_minority_osi = (recall_means.max() - recall_means.min()) / recall_means.max() if recall_means.max() > 0 else 0
        
        summary_data.append({
            'Dataset': dataset.upper(),
            'Best': best_order,
            'Best Mean': f"{best_samples.mean():.4f}",
            'Worst': worst_order,
            'Worst Mean': f"{worst_samples.mean():.4f}",
            'Diff': f"{diff:.4f}",
            'OSI': f"{osi:.1%}",
            'MCC OSI': f"{mcc_osi:.1%}" if mcc_osi else 'N/A',
            'Recall OSI': f"{recall_minority_osi:.1%}" if recall_minority_osi else 'N/A',
            "Cohen's d": f"{cohens_d:.2f}",
            'p-value': f"{p_value:.6f}",
            'Significant': '✓' if p_value < 0.05 else '✗'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("PER-DATASET SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # ============================================================
    # PART 2: FRIEDMAN TEST (Multi-Dataset Comparison)
    # ============================================================
    print("\n" + "="*80)
    print("FRIEDMAN TEST - Cross-Dataset Statistical Analysis")
    print("="*80)
    print("(Non-parametric test for multiple related samples)")
    print()
    
    # Prepare data matrix for Friedman test
    # Rows = cleaning orders, Columns = datasets
    all_orders = None
    performance_matrix = []
    dataset_names = []
    
    for result in successful_results:
        df = result['results_df']
        order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean().sort_index()
        
        if all_orders is None:
            all_orders = order_means.index.tolist()
        
        performance_matrix.append(order_means.values)
        dataset_names.append(result['dataset'].upper())
    
    # Transpose: rows=orders, columns=datasets
    performance_matrix = np.array(performance_matrix).T
    
    # Friedman test
    friedman_stat, friedman_p = stats.friedmanchisquare(*[performance_matrix[:, i] for i in range(performance_matrix.shape[1])])
    
    print(f"Friedman χ² = {friedman_stat:.4f}")
    print(f"p-value = {friedman_p:.6f}")
    
    if friedman_p < 0.05:
        print("✓ SIGNIFICANT: Cleaning orders differ significantly across datasets (p < 0.05)")
    else:
        print("✗ NOT SIGNIFICANT: No evidence that cleaning orders differ across datasets")
    
    # ============================================================
    # PART 3: AVERAGE RANKING ACROSS DATASETS
    # ============================================================
    print("\n" + "="*80)
    print("AVERAGE RANKINGS ACROSS DATASETS")
    print("="*80)
    print("(Lower rank = better performance, 1 = best)")
    print()
    
    # Calculate ranks for each dataset
    ranks_matrix = np.zeros_like(performance_matrix)
    for i, dataset_col in enumerate(performance_matrix.T):
        # Rank in descending order (higher performance = better rank)
        ranks_matrix[:, i] = stats.rankdata(-dataset_col)
    
    # Average rank per order
    avg_ranks = ranks_matrix.mean(axis=1)
    
    # Create ranking dataframe
    ranking_df = pd.DataFrame({
        'Cleaning Order': all_orders,
        'Avg Rank': avg_ranks,
        **{f'{dataset_names[i]} Rank': ranks_matrix[:, i] for i in range(len(dataset_names))}
    }).sort_values('Avg Rank')
    
    # Add interpretation
    ranking_df['Interpretation'] = ranking_df['Avg Rank'].apply(
        lambda x: 'Excellent' if x <= 5 else 'Good' if x <= 10 else 'Fair' if x <= 15 else 'Poor'
    )
    
    print(ranking_df.to_string(index=False))
    
    # Save rankings
    ranking_path = Path("results/tables/cross_dataset_rankings.csv")
    ranking_df.to_csv(ranking_path, index=False)
    print(f"\nRankings saved to: {ranking_path}")
    
    # ============================================================
    # PART 4: NEMENYI POST-HOC TEST (if Friedman significant)
    # ============================================================
    if friedman_p < 0.05:
        print("\n" + "="*80)
        print("NEMENYI POST-HOC TEST")
        print("="*80)
        print("(Pairwise comparisons between top and bottom orders)")
        print()
        
        # Compare top 3 vs bottom 3 orders
        top_orders_idx = avg_ranks.argsort()[:3]
        bottom_orders_idx = avg_ranks.argsort()[-3:]
        
        # Critical difference for Nemenyi test
        n_datasets = len(dataset_names)
        n_orders = len(all_orders)
        q_critical = 2.569  # Critical value for α=0.05, approximation
        cd = q_critical * np.sqrt((n_orders * (n_orders + 1)) / (6 * n_datasets))
        
        print(f"Critical difference (α=0.05): {cd:.2f}")
        print("\nSignificant pairwise differences (top 3 vs bottom 3):")
        
        nemenyi_results = []
        for top_idx in top_orders_idx:
            for bottom_idx in bottom_orders_idx:
                rank_diff = abs(avg_ranks[top_idx] - avg_ranks[bottom_idx])
                if rank_diff > cd:
                    nemenyi_results.append({
                        'Order 1': all_orders[top_idx],
                        'Rank 1': f"{avg_ranks[top_idx]:.1f}",
                        'Order 2': all_orders[bottom_idx],
                        'Rank 2': f"{avg_ranks[bottom_idx]:.1f}",
                        'Rank Diff': f"{rank_diff:.1f}",
                        'Significant': '✓'
                    })
        
        if nemenyi_results:
            nemenyi_df = pd.DataFrame(nemenyi_results)
            print(nemenyi_df.to_string(index=False))
        else:
            print("No significant pairwise differences found")
    
    # ============================================================
    # PART 5: KEY FINDINGS & PUBLICATION RECOMMENDATION
    # ============================================================
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    n_significant = sum([1 for r in summary_data if r['Significant'] == '✓'])
    n_total = len(summary_data)
    
    print(f"Per-dataset significance: {n_significant}/{n_total} datasets with p < 0.05")
    print(f"Cross-dataset Friedman test: {'SIGNIFICANT' if friedman_p < 0.05 else 'NOT SIGNIFICANT'} (p={friedman_p:.6f})")
    
    # Best consistent orders
    top_3_orders = ranking_df.head(3)['Cleaning Order'].tolist()
    print(f"\nMost consistent orders across datasets:")
    for i, order in enumerate(top_3_orders, 1):
        avg_rank = ranking_df[ranking_df['Cleaning Order'] == order]['Avg Rank'].values[0]
        print(f"  {i}. {order} (avg rank: {avg_rank:.1f})")
    
    print("\n" + "="*80)
    print("PUBLICATION RECOMMENDATION")
    print("="*80)
    
    # Decision based on both per-dataset AND Friedman test
    if friedman_p < 0.05 and n_significant >= 3:
        print("✓✓ STRONG EVIDENCE (Q1 Journal Level)")
        print("  - Friedman test significant across datasets")
        print("  - 3+ individual datasets show significance")
        print("  → Target: JMLR, Machine Learning, DMKD")
    elif friedman_p < 0.05 and n_significant >= 2:
        print("✓ GOOD EVIDENCE (Top Conference / Mid-Tier Journal)")
        print("  - Friedman test significant")
        print("  - 2+ individual datasets show significance")
        print("  → Target: NeurIPS, ICML, KDD, IEEE TKDE")
    elif friedman_p < 0.05 or n_significant >= 2:
        print("⚠ MODERATE EVIDENCE (Conference Level)")
        print("  - Some statistical evidence present")
        print("  → Target: ECML-PKDD, SDM, Applied journals")
    else:
        print("⚠ WEAK EVIDENCE (Workshop / Exploratory)")
        print("  - Limited statistical significance")
        print("  → Target: Workshop or reframe contribution")
    
    # Save summary
    summary_path = Path("results/tables/cross_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save comprehensive results
    comprehensive_results = {
        'per_dataset_summary': summary_df,
        'friedman_test': {'statistic': friedman_stat, 'p_value': friedman_p},
        'average_rankings': ranking_df,
        'n_significant_datasets': n_significant,
        'recommendation': 'Q1' if (friedman_p < 0.05 and n_significant >= 3) else 'Conference' if friedman_p < 0.05 else 'Workshop'
    }
    
    return comprehensive_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE 4-DATASET EXPERIMENT")
    print("="*80)
    print("Datasets: PIMA, Adult, Credit, Breast Cancer")
    print("Operations: 4-op (24 permutations × 5 seeds each)")
    print("Total runs: 4 datasets × 120 runs = 480 runs")
    print("Estimated time: 2-4 hours")
    print("="*80)
    
    input("\nPress Enter to start (or Ctrl+C to cancel)...")
    
    datasets = ['pima', 'adult', 'credit_default', 'home_credit']
    all_results = []
    
    start_time = datetime.now()
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n\n{'='*80}")
        print(f"PROGRESS: {i}/{len(datasets)} datasets")
        print(f"{'='*80}")
        
        result = run_single_dataset(dataset)
        all_results.append(result)
        
        if result['success']:
            print(f"✓ {dataset.upper()} complete")
        else:
            print(f"✗ {dataset.upper()} failed")
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n\n{'='*80}")
    print(f"ALL DATASETS COMPLETE")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"{'='*80}")
    
    # Cross-dataset analysis
    summary = analyze_cross_dataset(all_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: results/tables/")
    print(f"Summary: results/tables/cross_dataset_summary.csv")
    print("="*80)
