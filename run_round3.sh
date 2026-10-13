#!/bin/bash
# Round 3: rerun every condition that uses the three datasets with categorical
# features (now with SMOTENC and category-aware imputation and injection),
# plus two new experiments: symmetric label noise and 20 seeds.
set -eo pipefail
cd "$(dirname "$0")"
R=results_r3
CAT="credit_g adult credit_default"
run () { echo ">>> $(date '+%F %T')  $*"; python run_extended.py --results-dir "$R" "$@"; }

run --stage main      --datasets $CAT
run --stage noise_sym
run --stage seeds20
run --stage clean     --datasets $CAT
run --stage methods_b --datasets $CAT
run --stage subsets   --datasets credit_g
for s in smote_050 smote_100 noise_05 noise_10 noise_20; do
  run --stage $s --datasets credit_g
done
echo ">>> $(date '+%F %T')  ROUND 3 COMPLETE"
