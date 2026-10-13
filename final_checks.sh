#!/bin/bash
set -eo pipefail
python run_extended.py --stage seeds20 --datasets breast_cancer ionosphere banknote credit_g --results-dir results_r3
python e2e_foldsafe.py
python robustness_r4.py --skip-mechanism
echo "FINAL CHECKS COMPLETE"
