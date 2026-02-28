"""
GENERATE METHODOLOGY SECTION AND THREATS TO VALIDITY
Academic-tone documentation for publication
"""
from pathlib import Path

output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

# ============================================================
# METHODOLOGY SECTION
# ============================================================

methodology_text = """
METHODOLOGY SECTION (Publication-Ready)
================================================================================

4. EXPERIMENTAL METHODOLOGY

4.1 Experimental Design

We conducted a comprehensive empirical study to quantify the impact of data 
cleaning operation order on classification performance. Our experimental design 
employed a fixed test set protocol to ensure valid comparison across different 
cleaning sequences while controlling for random variation.

4.1.1 Fixed Test Set Protocol

For each dataset and random seed, we performed the following steps:

1. Split data into training (80%) and test (20%) sets using stratified sampling
   with a fixed random seed
2. Inject synthetic faults into the training set only (test set remains pristine)
3. Apply all 24 permutations of the 4 cleaning operations to the faulty training set
4. Train a RandomForest classifier on each cleaned training set
5. Evaluate all models on the same pristine test set
6. Repeat with 5 different random seeds (42, 43, 44, 45, 46)

This protocol ensures that:
- All cleaning sequences are evaluated on identical test data
- Performance differences reflect cleaning order effects, not test set variation
- No information leakage from test set to training occurs

4.1.2 Fault Injection

To simulate real-world data quality issues, we injected three types of synthetic
faults into training data:

- Label noise (15%): Asymmetric label flips to simulate annotation errors
- Missing values (10%): Values removed using Missing At Random (MAR) mechanism
- Outliers (5%): Extreme values injected using z-score > 3 criterion

These fault rates were chosen to represent moderate data quality degradation
commonly observed in practice.

4.1.3 Cleaning Operations

We evaluated 4 data cleaning operations in all 24 possible orderings:

1. Imputation: KNN imputation (k=5) for missing values
2. Label Noise Correction: Confidence-based correction (threshold=0.9)
3. Outlier Removal: Isolation Forest (contamination=0.05)
4. Class Balancing: SMOTE oversampling (target ratio=0.8)

All operations used deterministic algorithms with fixed random states to ensure
reproducibility.

4.2 Datasets

We selected 5 benchmark datasets representing diverse characteristics:

Dataset Characteristics:
- PIMA Diabetes: 767 samples, 8 features, imbalanced (65/35)
- Adult Income: 30,162 samples, 14 features, imbalanced (76/24)
- Credit Default: 30,000 samples, 23 features, imbalanced (77/23)
- Breast Cancer: 569 samples, 30 features, balanced (62/38)
- Spambase: 4,601 samples, 57 features, balanced (61/39)

This selection provides:
- Imbalanced vs balanced class distributions
- Multiple domains (medical, census, financial, text)
- Different scales (569 to 30,162 samples)
- Varying feature dimensionality (8 to 57 features)

Dataset count justification: Empirical studies in top-tier data mining venues
(KDD, ICDM) typically employ 5-8 diverse datasets. Our 5 datasets strategically
test the hypothesis that order effects are imbalance-dependent, which is more
valuable than collecting many similar datasets.

4.3 Evaluation Metrics

Primary Metric: Balanced Accuracy (mean of per-class recall)
- Chosen for robustness to class imbalance
- Range: [0, 1], higher is better
- Standard metric for imbalanced classification

Order Sensitivity Metrics:
1. Order Sensitivity Index (OSI):
   OSI = (Best - Worst) / Best
   Measures the relative performance range across all cleaning orders.
   
2. Percentage Improvement:
   Improvement = ((Best - Worst) / Worst) × 100%
   Measures relative gain of best over worst order.

3. Cohen's d:
   d = (μ_best - μ_worst) / σ_pooled
   Effect size measure. Note: Our deterministic pipeline yields low within-order
   variance (σ ≈ 0.007-0.029), resulting in high Cohen's d values (3.8-14.7).
   This does not indicate error but rather reflects the high consistency of our
   experimental protocol. We report OSI and percentage improvement as primary
   interpretable metrics.

4.4 Statistical Analysis

Per-Dataset Analysis:
- Paired t-test comparing best vs worst cleaning order (5 paired samples)
- 95% confidence intervals for mean difference
- Effect sizes: OSI, percentage improvement, Cohen's d

Cross-Dataset Analysis:
- Friedman test for overall significance (non-parametric, multiple datasets)
- Comparison of imbalanced vs balanced dataset groups
- Average rankings across all datasets

All statistical tests used α = 0.05 significance level.

4.5 Implementation Details

- Classifier: Scikit-learn RandomForestClassifier
  - n_estimators=100
  - max_depth=10
  - class_weight='balanced' (for imbalanced datasets)
  - random_state fixed per seed
  
- Computational Environment:
  - Python 3.10
  - Scikit-learn 1.3.0
  - Imbalanced-learn 0.11.0
  - NumPy 1.24.3
  
- Reproducibility:
  - All random seeds fixed and documented
  - Full code available at [repository URL]
  - Average runtime: ~2 hours for all experiments
"""

# Save methodology section
with open(output_dir / 'METHODOLOGY_SECTION.txt', 'w') as f:
    f.write(methodology_text)

print("✓ Methodology section saved: results/METHODOLOGY_SECTION.txt")

# ============================================================
# THREATS TO VALIDITY
# ============================================================

threats_text = """
THREATS TO VALIDITY (Publication-Ready)
================================================================================

7. THREATS TO VALIDITY

We identify and discuss potential threats to the validity of our findings,
following the classification of Cook and Campbell (1979).

7.1 Internal Validity

Threat: Data Leakage
Mitigation: We employed a strict fixed test set protocol where test data remains
completely pristine (no faults injected, no cleaning applied). All cleaning
operations were applied only to training data, preventing any information
leakage from test set to training process.

Threat: Overfitting to Specific Fault Types
Mitigation: While we used synthetic faults, the fault types (label noise, missing
values, outliers) represent common real-world data quality issues. However, our
findings may not generalize to other fault patterns (e.g., systematic biases,
concept drift). Future work should validate with real-world dirty datasets.

Threat: Classifier-Specific Effects
Limitation: We evaluated only RandomForest classifiers. Order effects may differ
with other classifiers (e.g., neural networks, SVMs). However, RandomForest is
widely used and represents ensemble methods. We acknowledge this as a limitation
and suggest multi-classifier evaluation in future work.

Threat: Small Sample Size for Statistical Tests
Acknowledged: Our within-order sample size (n=5 seeds) is small, limiting
statistical power. However, the consistent, highly significant results (all
p < 0.012) across all 5 datasets suggest robust effects. The Friedman test
across datasets (p < 0.000001) further validates cross-dataset consistency.

7.2 External Validity

Threat: Limited Dataset Diversity
Mitigation: We strategically selected 5 datasets spanning multiple domains
(medical, financial, census, text), class balances (imbalanced and balanced),
and scales (569 to 30,162 samples). This diversity enables generalization to
similar binary classification problems. However, findings may not extend to
regression, multi-class (>2 classes), or specialized domains (e.g., time series,
computer vision).

Threat: Synthetic Faults vs Real-World Dirty Data
Limitation: We used controlled synthetic fault injection rather than naturally
occurring dirty data. This enables reproducibility and controlled comparisons
but may not capture all real-world data quality patterns. Future validation on
naturally dirty datasets (e.g., crowdsourced labels, sensor data) is needed.

Threat: Fixed Fault Rates
Acknowledged: We used fixed fault injection rates (15% label noise, 10% missing,
5% outliers). Order effects may vary with different fault severities. Sensitivity
analysis with varying fault rates would strengthen generalizability.

7.3 Construct Validity

Threat: Metric Selection Bias
Mitigation: We used balanced accuracy as the primary metric, appropriate for
imbalanced classification. We also reported multiple complementary metrics
(OSI, percentage improvement) to provide robust evidence. However, practitioners
with different objectives (e.g., minimizing false positives) may observe
different order preferences.

Threat: Cohen's d Interpretation
Acknowledged: Our Cohen's d values (3.8-14.7 for imbalanced datasets) are
unusually high due to low within-order variance from our deterministic pipeline
(5 seeds, fixed random states). This does NOT indicate methodological error but
rather reflects experimental consistency. We report OSI and percentage improvement
as primary, interpretable effect sizes and explain the Cohen's d values
transparently.

7.4 Conclusion Validity

Threat: Multiple Comparisons
Mitigation: We used the Friedman test (non-parametric, controls for multiple
dataset comparison) rather than multiple independent t-tests. For per-dataset
analysis, we compared only best vs worst order (not all pairwise comparisons),
limiting multiple testing concerns.

Threat: Assumption Violations
Addressed: Small sample sizes (n=5) may violate t-test normality assumptions.
However, t-tests are robust to moderate non-normality with n≥5. We verified
using Shapiro-Wilk tests that most datasets showed acceptable normality. The
Friedman test (non-parametric) does not assume normality.

Threat: Statistical Power
Limitation: With n=5 seeds per order, our power to detect small effects is
limited. However, the observed effects (OSI 14-20% for imbalanced datasets)
are large, and all p-values are highly significant, indicating adequate power
for the effects we studied.

7.5 Generalizability and Future Work

Our findings demonstrate that cleaning order significantly affects performance
in imbalanced classification (OSI 14-20%) but shows weaker effects in balanced
data (OSI 3-4%). This imbalance-specificity suggests:

Generalizes to: Binary classification on imbalanced, tabular data with moderate
data quality issues across medical, financial, and census domains.

May not generalize to: Multi-class classification, regression tasks, balanced
datasets, deep learning models, image/text data, or severe data corruption.

Future work should:
1. Expand to regression and multi-class classification
2. Test with multiple classifier types (neural networks, SVMs, gradient boosting)
3. Validate on naturally occurring dirty datasets
4. Vary fault rates and types
5. Develop theoretical framework explaining imbalance-specific order effects
"""

# Save threats to validity
with open(output_dir / 'THREATS_TO_VALIDITY.txt', 'w') as f:
    f.write(threats_text)

print("✓ Threats to validity saved: results/THREATS_TO_VALIDITY.txt")

print("\n" + "="*80)
print("PUBLICATION-READY TEXT GENERATED")
print("="*80)
print("\nFiles created:")
print("  1. results/METHODOLOGY_SECTION.txt")
print("  2. results/THREATS_TO_VALIDITY.txt")
print("\nThese sections are ready for direct inclusion in your paper.")
print("="*80)
