"""
COMPREHENSIVE 5-DATASET ANALYSIS
Professional statistical analysis with strategic diversity
PIMA, Adult, Credit (imbalanced) + Breast Cancer, Spambase (balanced)
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob

print("="*80)
print("COMPREHENSIVE 5-DATASET ANALYSIS - STRATEGIC DIVERSITY")
print("="*80)

# Auto-detect latest result files for each dataset
dataset_names = ['pima', 'adult', 'credit_default', 'breast_cancer', 'spambase']

print("\nSearching for result files...")
datasets = {}
missing = []

for name in dataset_names:
    # Find latest file for this dataset
    pattern = f"results/tables/{name}_scientific_results_*.csv"
    files = glob.glob(pattern)
    
    if files:
        latest_file = max(files)  # Gets most recent by filename
        try:
            df = pd.read_csv(latest_file)
            if len(df) == 120:  # Verify complete
                datasets[name.upper()] = df
                print(f"  ✓ {name:20s} - {latest_file}")
            else:
                print(f"  ⚠ {name:20s} - INCOMPLETE ({len(df)}/120 runs)")
                missing.append(name)
        except Exception as e:
            print(f"  ✗ {name:20s} - ERROR: {e}")
            missing.append(name)
    else:
        print(f"  ✗ {name:20s} - NOT FOUND")
        missing.append(name)

if missing:
    print(f"\n⚠ WARNING: Missing {len(missing)} datasets: {', '.join(missing)}")
    print(f"Proceeding with {len(datasets)} available datasets")

if len(datasets) < 3:
    print("\n✗ ERROR: Need at least 3 datasets for analysis")
    exit(1)

print(f"\n✓ Loaded {len(datasets)} complete datasets")

# ============================================================
# DATASET CHARACTERISTICS
# ============================================================
print("\n" + "="*80)
print("DATASET CHARACTERISTICS")
print("="*80)

print("\nImbalanced Datasets (3):")
print("  - PIMA: Medical, 767 samples")
print("  - ADULT: Census, 30K samples")  
print("  - CREDIT_DEFAULT: Financial, 30K samples")

if 'BREAST_CANCER' in datasets:
    print("\nBalanced Datasets (2):")
    print("  - BREAST_CANCER: Medical, 569 samples")
if 'SPAMBASE' in datasets:
    print("  - SPAMBASE: Text-derived, 4.6K samples")

print("\nDiversity Achieved:")
print("  ✓ Class balance: Imbalanced vs Balanced")
print("  ✓ Domains: Medical, Census, Financial, Text")
print("  ✓ Scales: 569 to 30K samples")

# ============================================================
# PER-DATASET ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PER-DATASET STATISTICAL ANALYSIS")
print("="*80)

results = []

for name, df in datasets.items():
    order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
    best_order = order_means.idxmax()
    worst_order = order_means.idxmin()
    
    best_samples = df[df['cleaning_order'] == best_order]['balanced_accuracy'].values
    worst_samples = df[df['cleaning_order'] == worst_order]['balanced_accuracy'].values
    
    t_stat, p_value = stats.ttest_rel(best_samples, worst_samples)
    
    diff = best_samples.mean() - worst_samples.mean()
    cohens_d = diff / np.sqrt((best_samples.std()**2 + worst_samples.std()**2) / 2)
    osi = (order_means.max() - order_means.min()) / order_means.max()
    pct_diff = (diff / worst_samples.mean()) * 100
    
    results.append({
        'Dataset': name,
        'Type': 'Imbal' if name in ['PIMA', 'ADULT', 'CREDIT_DEFAULT'] else 'Balanced',
        'Best Mean': f"{best_samples.mean():.4f}",
        'Worst Mean': f"{worst_samples.mean():.4f}",
        'Diff': f"{diff:.4f}",
        'OSI': f"{osi:.1%}",
        '% Improv': f"{pct_diff:.1f}%",
        'p-value': f"{p_value:.6f}",
        'Sig': '✓' if p_value < 0.05 else '✗'
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================
# FRIEDMAN TEST
# ============================================================
print("\n" + "="*80)
print("FRIEDMAN TEST - Cross-Dataset Statistical Analysis")
print("="*80)

# Get all unique orders
all_orders = sorted(list(datasets.values())[0]['cleaning_order'].unique())

# Build performance matrix
performance_matrix = []
dataset_names_list = []

for name, df in datasets.items():
    order_means = df.groupby('cleaning_order')['balanced_accuracy'].mean()
    performance_matrix.append([order_means[order] for order in all_orders])
    dataset_names_list.append(name)

performance_matrix = np.array(performance_matrix).T

# Friedman test
friedman_stat, friedman_p = stats.friedmanchisquare(*[performance_matrix[:, i] for i in range(performance_matrix.shape[1])])

print(f"\nNumber of datasets: {len(datasets)}")
print(f"Friedman χ² = {friedman_stat:.4f}")
print(f"p-value = {friedman_p:.6f}")

if friedman_p < 0.05:
    sig_level = "***" if friedman_p < 0.001 else "**" if friedman_p < 0.01 else "*"
    print(f"✓ SIGNIFICANT {sig_level}: Cleaning orders differ significantly across datasets")
else:
    print("✗ NOT SIGNIFICANT: No evidence that cleaning orders differ across datasets")

# ============================================================
# IMBALANCED VS BALANCED COMPARISON
# ============================================================
if 'BREAST_CANCER' in datasets or 'SPAMBASE' in datasets:
    print("\n" + "="*80)
    print("IMBALANCED vs BALANCED COMPARISON")
    print("="*80)
    
    imbal_osi = [float(r['OSI'].rstrip('%')) for r in results if r['Type'] == 'Imbal']
    balanced_osi = [float(r['OSI'].rstrip('%')) for r in results if r['Type'] == 'Balanced']
    
    print(f"\nImbalanced datasets (n={len(imbal_osi)}):")
    print(f"  OSI range: {min(imbal_osi):.1f}% - {max(imbal_osi):.1f}%")
    print(f"  Mean OSI: {np.mean(imbal_osi):.1f}%")
    
    if balanced_osi:
        print(f"\nBalanced datasets (n={len(balanced_osi)}):")
        print(f"  OSI range: {min(balanced_osi):.1f}% - {max(balanced_osi):.1f}%")
        print(f"  Mean OSI: {np.mean(balanced_osi):.1f}%")
        
        print(f"\nInterpretation:")
        if np.mean(balanced_osi) > 5:  # Meaningful effect in balanced too
            print("  ✓ Order effects persist in balanced datasets")
            print("  → Effect is GENERAL, not imbalance-specific")
        else:
            print("  ⚠ Order effects weaker in balanced datasets")
            print("  → Effect may be imbalance-specific")

# ============================================================
# PUBLICATION RECOMMENDATION
# ============================================================
print("\n" + "="*80)
print("PUBLICATION RECOMMENDATION")
print("="*80)

n_significant = sum([1 for r in results if r['Sig'] == '✓'])
n_datasets = len(datasets)

print(f"\nEvidence Summary:")
print(f"  - Datasets analyzed: {n_datasets}")
print(f"  - Dataset diversity: {'High (balanced + imbalanced)' if n_datasets >= 5 else 'Moderate'}")
print(f"  - Per-dataset significance: {n_significant}/{n_datasets}")
print(f"  - Friedman test: {'SIGNIFICANT' if friedman_p < 0.05 else 'NOT SIGNIFICANT'} (p={friedman_p:.6f})")

# Publication recommendation
if friedman_p < 0.01 and n_significant >= 4 and n_datasets >= 5:
    print("\n✓✓ STRONG EVIDENCE")
    print("  Target: Top Conference (KDD, ICDM) or Q1 Journal (DMKD, TKDE)")
    print("  Claim: Cleaning order significantly affects performance across diverse datasets")
    print("  Strength: Multiple datasets, diverse characteristics, strong statistics")
elif friedman_p < 0.05 and n_significant >= 3:
    print("\n✓ GOOD EVIDENCE")
    print("  Target: Good Conference (ECML-PKDD, SDM) or Mid-tier Journal")
    print("  Claim: Statistically significant order effects observed")
    print("  Strength: Solid evidence with room for expansion")
else:
    print("\n⚠ MODERATE EVIDENCE")
    print("  Target: Conference or Specialized Venue")
    print("  Claim: Order sensitivity context-dependent")

# ============================================================
# SAVE RESULTS
# ============================================================
results_df.to_csv('results/tables/5dataset_summary.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("Files saved:")
print("  - results/tables/5dataset_summary.csv")
print("="*80)
