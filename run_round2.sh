#!/bin/zsh
# Round 2, run from the project root after round 1 has finished. Stops at the first error.
set -e
for stage in high_ir methods_b smote_050 smote_100 noise_05 noise_10 noise_20; do
  echo "=== $stage $(date)"
  python run_extended.py --stage $stage
done
echo "=== diagnostics $(date)"; python diagnostics.py
echo "=== analysis $(date)";    python analyze.py
echo "=== all done $(date)"
