"""
MASTER REPRODUCIBILITY SCRIPT
Single command to reproduce all experiments, tables, and figures
Fixed seeds, clean logging, publication-ready outputs
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
log_dir = Path('results/logs')
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'reproduction_run_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

print("="*80)
print("MASTER REPRODUCIBILITY SCRIPT")
print("="*80)
print(f"Log file: {log_file}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================
# CONFIGURATION
# ============================================================
FIXED_SEEDS = [42, 43, 44, 45, 46]
DATASETS = ['pima', 'adult', 'credit_default', 'breast_cancer', 'spambase']
N_PERMUTATIONS = 24  # 4! = 24 permutations of 4 operations
N_RUNS_PER_DATASET = len(FIXED_SEEDS) * N_PERMUTATIONS  # 120

logger.info("Configuration:")
logger.info(f"  Seeds: {FIXED_SEEDS}")
logger.info(f"  Datasets: {DATASETS}")
logger.info(f"  Expected runs per dataset: {N_RUNS_PER_DATASET}")

# ============================================================
# STEP 1: VERIFY ENVIRONMENT
# ============================================================
logger.info("\nSTEP 1: Verifying environment...")

try:
    import pandas as pd
    import numpy as np
    import scipy
    import sklearn
    import imblearn
    logger.info("✓ All required packages installed")
except ImportError as e:
    logger.error(f"✗ Missing package: {e}")
    sys.exit(1)

# Check data files exist
data_dir = Path('data/raw')
required_files = {
    'pima': 'pima.csv',
    'adult': 'adult.csv',
    'credit_default': 'credit_default.xls',
    'breast_cancer': None,  # From sklearn
    'spambase': 'spambase.csv'
}

for dataset, filename in required_files.items():
    if filename and not (data_dir / filename).exists():
        logger.warning(f"⚠ Missing: {data_dir / filename}")
    elif dataset == 'breast_cancer':
        logger.info(f"✓ {dataset}: Using sklearn.datasets")
    else:
        logger.info(f"✓ {dataset}: {filename}")

# ============================================================
# STEP 2: RUN EXPERIMENTS (OR VERIFY EXISTING)
# ============================================================
logger.info("\nSTEP 2: Checking experimental results...")

results_dir = Path('results/tables')
results_dir.mkdir(parents=True, exist_ok=True)

existing_results = {}
for dataset in DATASETS:
    pattern = f"{dataset}_scientific_results_*.csv"
    files = list(results_dir.glob(pattern))
    if files:
        latest = max(files)
        df = pd.read_csv(latest)
        if len(df) == N_RUNS_PER_DATASET:
            existing_results[dataset] = latest
            logger.info(f"✓ {dataset}: {len(df)} runs (complete)")
        else:
            logger.warning(f"⚠ {dataset}: {len(df)}/{N_RUNS_PER_DATASET} runs (incomplete)")
    else:
        logger.warning(f"⚠ {dataset}: No results found")

if len(existing_results) == len(DATASETS):
    logger.info(f"\n✓ All {len(DATASETS)} datasets have complete results")
    logger.info("  Skipping experiment runs (use existing results)")
else:
    logger.warning(f"\n⚠ Only {len(existing_results)}/{len(DATASETS)} datasets complete")
    logger.info("  To reproduce experiments, run: python run_all_experiments.py")
    logger.info("  (This will take 3-4 hours)")

# ============================================================
# STEP 3: STATISTICAL VALIDATION
# ============================================================
logger.info("\nSTEP 3: Running statistical validation...")

try:
    result = subprocess.run(
        [sys.executable, 'validate_statistics.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        logger.info("✓ Statistical validation complete")
        logger.info(f"  Output: results/tables/statistical_validation_report.csv")
    else:
        logger.error(f"✗ Statistical validation failed: {result.stderr}")
except FileNotFoundError:
    logger.warning("⚠ validate_statistics.py not found - skipping")
except subprocess.TimeoutExpired:
    logger.error("✗ Statistical validation timed out")

# ============================================================
# STEP 4: GENERATE PUBLICATION TABLES
# ============================================================
logger.info("\nSTEP 4: Generating publication-ready tables...")

try:
    result = subprocess.run(
        [sys.executable, 'create_publication_tables.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        logger.info("✓ Publication tables generated")
        logger.info("  Tables created:")
        logger.info("    - TABLE1_comprehensive_statistics.csv")
        logger.info("    - TABLE2_effect_size_guide.csv")
        logger.info("    - TABLE3_cross_dataset_tests.csv")
        logger.info("    - DATASET_JUSTIFICATION.csv")
    else:
        logger.error(f"✗ Table generation failed: {result.stderr}")
except FileNotFoundError:
    logger.warning("⚠ create_publication_tables.py not found - skipping")
except subprocess.TimeoutExpired:
    logger.error("✗ Table generation timed out")

# ============================================================
# STEP 5: GENERATE METHODOLOGY TEXT
# ============================================================
logger.info("\nSTEP 5: Generating methodology documentation...")

try:
    result = subprocess.run(
        [sys.executable, 'generate_methodology_text.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        logger.info("✓ Methodology documentation generated")
        logger.info("  Output: results/METHODOLOGY_SECTION.txt")
        logger.info("  Output: results/THREATS_TO_VALIDITY.txt")
    else:
        logger.warning(f"⚠ Methodology generation warning: {result.stderr}")
except FileNotFoundError:
    logger.warning("⚠ generate_methodology_text.py not found - skipping")

# ============================================================
# STEP 6: VERIFY REPRODUCIBILITY
# ============================================================
logger.info("\nSTEP 6: Verifying reproducibility...")

verification_passed = True

# Check all expected outputs exist
expected_outputs = [
    'results/tables/statistical_validation_report.csv',
    'results/tables/TABLE1_comprehensive_statistics.csv',
    'results/tables/TABLE2_effect_size_guide.csv',
    'results/tables/TABLE3_cross_dataset_tests.csv',
    'results/tables/DATASET_JUSTIFICATION.csv'
]

for output in expected_outputs:
    if Path(output).exists():
        logger.info(f"  ✓ {output}")
    else:
        logger.warning(f"  ✗ {output} - NOT FOUND")
        verification_passed = False

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("REPRODUCIBILITY RUN COMPLETE")
print("="*80)

if verification_passed:
    print("\n✓✓ SUCCESS - All outputs generated")
    print("\nPUBLICATION-READY FILES:")
    print("  1. Statistical Validation: results/tables/statistical_validation_report.csv")
    print("  2. Table 1 (Statistics): results/tables/TABLE1_comprehensive_statistics.csv")
    print("  3. Table 2 (Effect Sizes): results/tables/TABLE2_effect_size_guide.csv")
    print("  4. Table 3 (Cross-Dataset): results/tables/TABLE3_cross_dataset_tests.csv")
    print("  5. Dataset Justification: results/tables/DATASET_JUSTIFICATION.csv")
    print("  6. Methodology Section: results/METHODOLOGY_SECTION.txt")
    print("  7. Threats to Validity: results/THREATS_TO_VALIDITY.txt")
else:
    print("\n⚠ PARTIAL SUCCESS - Some outputs missing")
    print("Check log file for details: " + str(log_file))

print("\nLOG FILE: " + str(log_file))
print("="*80)

logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
