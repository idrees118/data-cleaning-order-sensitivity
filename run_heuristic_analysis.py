"""
Heuristic Order Analysis
========================

Tests a simple heuristic: Always put label_noise correction LAST.

Heuristic order: imputation → outliers → balance → label_noise

Compares heuristic performance vs best-of-24 for each dataset.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_heuristic_performance():
    """Analyze how close the heuristic gets to best-of-24."""
    
    # Define heuristic order (label noise last)
    heuristic_order = "imputation→outliers→balance→label_noise"
    
    datasets = ['pima', 'adult', 'credit_default', 'breast_cancer', 'spambase']
    
    results = []
    
    print("="*70)
    print("HEURISTIC ORDER ANALYSIS")
    print("="*70)
    print(f"\nHeuristic: {heuristic_order}")
    print("(Always place label noise correction last)")
    print("\n" + "="*70)
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        
        # Load Random Forest results
        rf_file = list(Path('results/tables').glob(f'{dataset}_scientific_results_*.csv'))[0]
        df = pd.read_csv(rf_file)
        
        # Compute mean performance per order
        order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
        
        # Best of 24
        best_order = order_means.idxmax()
        best_perf = order_means.max()
        
        # Heuristic performance
        if heuristic_order in order_means.index:
            heuristic_perf = order_means[heuristic_order]
        else:
            print(f"  WARNING: Heuristic order not found! Using closest...")
            heuristic_perf = None
        
        # Gap
        if heuristic_perf is not None:
            gap_pp = (best_perf - heuristic_perf) * 100
            pct_of_best = (heuristic_perf / best_perf) * 100
            
            print(f"  Best-of-24: {best_perf:.4f} ({best_order})")
            print(f"  Heuristic:  {heuristic_perf:.4f}")
            print(f"  Gap:        {gap_pp:.2f} pp ({100-pct_of_best:.2f}% loss)")
            print(f"  Heuristic achieves {pct_of_best:.2f}% of best")
            
            results.append({
                'Dataset': dataset,
                'Best_Performance': best_perf,
                'Best_Order': best_order,
                'Heuristic_Performance': heuristic_perf,
                'Gap_pp': gap_pp,
                'Pct_of_Best': pct_of_best
            })
        else:
            print(f"  ERROR: Could not find heuristic order")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        gaps = [r['Gap_pp'] for r in results]
        pcts = [r['Pct_of_Best'] for r in results]
        
        print(f"\nMean gap:          {np.mean(gaps):.2f} pp")
        print(f"Median gap:        {np.median(gaps):.2f} pp")
        print(f"Max gap:           {np.max(gaps):.2f} pp")
        print(f"Min gap:           {np.min(gaps):.2f} pp")
        print(f"\nMean % of best:    {np.mean(pcts):.2f}%")
        print(f"Median % of best:  {np.median(pcts):.2f}%")
        
        print(f"\n✓ Heuristic captures {np.mean(pcts):.1f}% of best performance on average")
        print(f"✓ Average loss: only {np.mean(gaps):.2f} percentage points")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = Path('results/tables/heuristic_analysis.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Create LaTeX table
        create_latex_table(results_df)
    
    return results


def create_latex_table(results_df):
    """Create LaTeX table for paper."""
    
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    print()
    
    latex = r"""\begin{table}[t]
\small
\centering
\caption{Heuristic order performance vs best-of-24. Heuristic places 
label noise correction last (imputation→outliers→balance→label\_noise).}
\label{tab:heuristic}
\begin{tabular}{lccc}
\toprule
Dataset & Best-of-24 & Heuristic & Gap (pp) \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        dataset_name = row['Dataset'].replace('_', ' ').title()
        best = row['Best_Performance']
        heur = row['Heuristic_Performance']
        gap = row['Gap_pp']
        latex += f"{dataset_name} & {best:.4f} & {heur:.4f} & {gap:.2f} \\\\\n"
    
    # Add means
    mean_best = results_df['Best_Performance'].mean()
    mean_heur = results_df['Heuristic_Performance'].mean()
    mean_gap = results_df['Gap_pp'].mean()
    
    latex += r"""\midrule
Mean & """ + f"{mean_best:.4f} & {mean_heur:.4f} & {mean_gap:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    print(latex)
    
    # Save to file
    with open('results/tables/heuristic_table.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: results/tables/heuristic_table.tex")


def analyze_across_classifiers():
    """Check heuristic performance across all 3 classifiers."""
    
    heuristic_order = "imputation→outliers→balance→label_noise"
    datasets = ['pima', 'adult', 'credit_default', 'breast_cancer', 'spambase']
    classifiers = {
        'RF': '_scientific_results_',
        'LR': '_logistic_regression_results',
        'XGB': '_xgboost_results'
    }
    
    print("\n" + "="*70)
    print("HEURISTIC ACROSS ALL CLASSIFIERS")
    print("="*70)
    
    all_gaps = []
    
    for clf_name, clf_pattern in classifiers.items():
        print(f"\n{clf_name}:")
        clf_gaps = []
        
        for dataset in datasets:
            try:
                if clf_name == 'RF':
                    files = list(Path('results/tables').glob(f'{dataset}{clf_pattern}*.csv'))
                else:
                    files = [Path('results/tables') / f'{dataset}{clf_pattern}.csv']
                
                if not files or not files[0].exists():
                    continue
                
                df = pd.read_csv(files[0])
                order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
                
                best_perf = order_means.max()
                
                if heuristic_order in order_means.index:
                    heur_perf = order_means[heuristic_order]
                    gap = (best_perf - heur_perf) * 100
                    clf_gaps.append(gap)
                    print(f"  {dataset}: gap = {gap:.2f} pp")
            except Exception as e:
                print(f"  {dataset}: Error - {e}")
        
        if clf_gaps:
            print(f"  Mean gap: {np.mean(clf_gaps):.2f} pp")
            all_gaps.extend(clf_gaps)
    
    if all_gaps:
        print(f"\nOverall mean gap across all classifiers: {np.mean(all_gaps):.2f} pp")


if __name__ == "__main__":
    results = analyze_heuristic_performance()
    analyze_across_classifiers()
