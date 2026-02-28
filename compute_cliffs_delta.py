"""
Compute Cliff's Delta for Best vs Worst Orders
===============================================

Cliff's delta is a nonparametric effect size measure.
Values range from -1 to +1:
- |δ| < 0.147: negligible
- 0.147 ≤ |δ| < 0.33: small
- 0.33 ≤ |δ| < 0.474: medium
- |δ| ≥ 0.474: large
"""

import numpy as np
import pandas as pd
from pathlib import Path


def cliffs_delta(x, y):
    """
    Compute Cliff's delta.
    
    δ = (# times x > y - # times x < y) / (n_x * n_y)
    """
    n = 0
    greater = 0
    
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1
            elif xi < yi:
                greater -= 1
            n += 1
    
    return greater / n if n > 0 else 0


def compute_cliffs_delta_all():
    """Compute Cliff's delta for all datasets."""
    
    datasets = ['pima', 'adult', 'credit_default', 'breast_cancer', 'spambase']
    
    print("="*70)
    print("CLIFF'S DELTA COMPUTATION")
    print("="*70)
    
    results = []
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        
        # Load Random Forest results
        rf_file = list(Path('results/tables').glob(f'{dataset}_scientific_results_*.csv'))[0]
        df = pd.read_csv(rf_file)
        
        # Find best and worst orders
        order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
        best_order = order_means.idxmax()
        worst_order = order_means.idxmin()
        
        # Get 5 seed values for each
        best_vals = df[df['cleaning_order'] == best_order]['balanced_accuracy'].values
        worst_vals = df[df['cleaning_order'] == worst_order]['balanced_accuracy'].values
        
        # Compute Cliff's delta
        delta = cliffs_delta(best_vals, worst_vals)
        
        # Interpret magnitude
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            magnitude = "negligible"
        elif abs_delta < 0.33:
            magnitude = "small"
        elif abs_delta < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        print(f"  Best:  {best_order}")
        print(f"  Worst: {worst_order}")
        print(f"  Cliff's δ = {delta:.3f} ({magnitude})")
        
        results.append({
            'Dataset': dataset,
            'Best_Order': best_order,
            'Worst_Order': worst_order,
            'Cliffs_Delta': delta,
            'Magnitude': magnitude
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path('results/tables/cliffs_delta.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Print LaTeX table update
    print("\n" + "="*70)
    print("UPDATE FOR TABLE \\ref{tab:statistical_tests}")
    print("="*70)
    print("\nAdd Cliff's δ column:")
    print()
    for _, row in results_df.iterrows():
        dataset = row['Dataset'].replace('_', ' ').title()
        delta = row['Cliffs_Delta']
        print(f"{dataset} & ... & ... & ... & {delta:.2f} \\\\")
    
    return results_df


if __name__ == "__main__":
    results = compute_cliffs_delta_all()
