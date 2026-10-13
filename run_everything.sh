#!/bin/zsh
# Full experiment, run from the project root. Stops at the first error.
set -e
AB=(pima breast_cancer ionosphere banknote credit_g qsar_biodeg spambase)

echo "=== main     $(date)"; python run_extended.py --stage main
echo "=== clean    $(date)"; python run_extended.py --stage clean
echo "=== subsets  $(date)"; python run_extended.py --stage subsets
echo "=== A/B leaky $(date)"; (cd ../order_ab_leaky && python run_extended.py --stage main --datasets "${AB[@]}")
echo "=== A/B fix12 $(date)"; (cd ../order_ab_fix12 && python run_extended.py --stage main --datasets "${AB[@]}")
echo "=== all done $(date)"
