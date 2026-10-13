"""
e2e_foldsafe.py - end-to-end version of the fold-dependence check (~30-40 min).

Section V showed that label correction gains precision when its forest sees balanced
classes, even without fold dependence. This script asks the end-to-end question:
does the final classifier still gain?

Four pipelines, same splits, faults, seeds and classifiers as the main condition:
  A  IMP->BAL->LNR->OUT            (main results; read from results/main)
  B  IMP->LNR->BAL->OUT            (main results; read from results/main)
  C  IMP->LNR[SMOTE in folds]->BAL->OUT   (no fold dependence)
  D  IMP->LNR[class-weighted]->BAL->OUT   (no synthetic rows in the correction step)
C and D change only how the correction forest is fitted; their training data reach BAL
and OUT exactly as in B. So C - B and D - B isolate the effect of a balanced correction
model on the final classifier, with no dependence between folds.

Before computing C and D, the script rebuilds B for one dataset and seed and checks that
it reproduces the stored main results, so the comparison is like for like.

    python e2e_foldsafe.py
Writes results/analysis/r4_e2e_foldsafe.csv and prints the summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data.loaders import load_dataset                    # noqa: E402
from src.faults.injection import FaultInjector               # noqa: E402
from src.cleaning.methods import DataCleaner                 # noqa: E402
from src.models.train import train_and_evaluate              # noqa: E402
from run_extended import FAULTS, MAIN_DATASETS, SEEDS, TEST_SIZE, CLASSIFIERS   # noqa: E402
from robustness_r4 import lnr_variant                        # noqa: E402

OUT = ROOT / "results" / "analysis"
A_ORDER, B_ORDER = "IMP→BAL→LNR→OUT", "IMP→LNR→BAL→OUT"


def finish(X, y, seed, cat, X_te, y_te):
    """BAL then OUT, then every classifier, exactly as in the main pipelines."""
    X, y, _ = DataCleaner(seed, cat_cols=cat).balance(X, y)
    X, y, _ = DataCleaner(seed, cat_cols=cat).remove_outliers(X, y)
    return {c: train_and_evaluate(X, y, X_te, y_te, c, random_state=seed)["balanced_accuracy"]
            for c in CLASSIFIERS}


def main():
    main_df = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results" / "main").glob("*.csv"))
                         if not f.name.startswith("fault_audit_")], ignore_index=True)
    stored = (main_df[main_df.cleaning_order.isin([A_ORDER, B_ORDER])]
              .pivot_table(index=["dataset", "seed", "classifier"], columns="cleaning_order",
                           values="balanced_accuracy"))
    rows, checked = [], False
    for name in MAIN_DATASETS:
        d = load_dataset(name)
        X_all, y_all, cat = d["X"], d["y"], list(d.get("cat_cols", []))
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE,
                                                      random_state=seed, stratify=y_all)
            inj = FaultInjector(random_state=seed, cat_cols=cat).inject_all_faults(X_tr, y_tr, FAULTS)
            X_imp, y_imp, _ = DataCleaner(seed, cat_cols=cat).impute(inj["X_corrupted"], inj["y_corrupted"])
            if not checked:
                b = finish(X_imp, lnr_variant(X_imp, y_imp, seed, cat, "plain"), seed, cat, X_te, y_te)
                diff = max(abs(b[c] - stored.loc[(name, seed, c), B_ORDER]) for c in CLASSIFIERS)
                if diff > 1e-9:
                    sys.exit(f"rebuilt {B_ORDER} differs from the stored main results by {diff:.2e}; stopping")
                print(f"  check passed: rebuilt {B_ORDER} reproduces the main results exactly", flush=True)
                checked = True
            res = {m: finish(X_imp, lnr_variant(X_imp, y_imp, seed, cat, m), seed, cat, X_te, y_te)
                   for m in ("fold_smote", "weighted")}
            for c in CLASSIFIERS:
                rows.append({"dataset": name, "seed": seed, "classifier": c,
                             "A_bal_first": stored.loc[(name, seed, c), A_ORDER],
                             "B_lnr_first": stored.loc[(name, seed, c), B_ORDER],
                             "C_foldsmote": res["fold_smote"][c], "D_weighted": res["weighted"][c]})
        print(f"  [{name}] done", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "r4_e2e_foldsafe.csv", index=False)
    m = t.groupby(["classifier", "dataset"]).mean(numeric_only=True).drop(columns="seed")
    for col in ["A_bal_first", "C_foldsmote", "D_weighted"]:
        m[col + "_minus_B_pp"] = 100 * (m[col] - m["B_lnr_first"])
    print("\n=== End-to-end balanced accuracy minus IMP→LNR→BAL→OUT (pp), mean over seeds ===")
    print(m[[c for c in m.columns if c.endswith("_pp")]].round(2).to_string())
    print("\n=== Across the ten datasets (positive = better than correcting labels on imbalanced data) ===")
    for clf, g in m.groupby(level=0):
        line = []
        for col, lab in [("A_bal_first", "balancing first"), ("C_foldsmote", "SMOTE inside folds"),
                         ("D_weighted", "class-weighted")]:
            dd = g[col + "_minus_B_pp"].to_numpy()
            p = stats.wilcoxon(dd).pvalue
            line.append(f"{lab}: {int((dd > 0).sum())}/10 better, median {np.median(dd):+.2f}, p = {p:.3f}")
        print(f"  {clf:<20} " + " | ".join(line))
    a, c = m["A_bal_first_minus_B_pp"], m["C_foldsmote_minus_B_pp"]
    print(f"\n  correlation of the balancing-first gain with the fold-safe gain over 30 pairs: "
          f"Spearman rho = {stats.spearmanr(a, c)[0]:.2f}")


if __name__ == "__main__":
    main()
