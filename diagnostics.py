"""
Mechanism diagnostics under the main fault model (method set A).

1. Label noise correction with and without SMOTE before it:
       IMP -> LNR          and          IMP -> BAL -> LNR
   The injector records which labels it flipped, so for every relabelled row we
   know whether the change restored the true label (fixed) or broke a correct one
   (broken). In the second sequence, original rows come first in the balanced
   data, so they can be matched back; relabelled synthetic rows are counted apart.

2. Outlier removal after SMOTE:  IMP -> BAL -> OUT
   How many removed rows are SMOTE points, against their share of the data.
   The row index is passed through as the label so kept rows can be identified;
   remove_outliers only uses the label to subset it.

Same seeds, splits and faults as run_extended.py --stage main.

    python diagnostics.py                     # all main datasets
    python diagnostics.py --datasets pima     # a quick check
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data.loaders import load_dataset                 # noqa: E402
from src.faults.injection import FaultInjector            # noqa: E402
from src.cleaning.methods import DataCleaner              # noqa: E402
from run_extended import FAULTS, MAIN_DATASETS, SEEDS, TEST_SIZE   # noqa: E402


def relabel_stats(y_obs, y_new, flipped, minority):
    changed = y_new != y_obs
    n = int(changed.sum())
    return {
        "relabelled": n,
        "fixed": int((changed & flipped).sum()),
        "broken": int((changed & ~flipped).sum()),
        "precision": (changed & flipped).sum() / n if n else np.nan,
        "recall": (changed & flipped).sum() / flipped.sum() if flipped.sum() else np.nan,
        "minority_to_majority": int((changed & (y_obs == minority)).sum()),
        "true_minority_broken": int((changed & ~flipped & (y_obs == minority)).sum()),
    }


def run(name):
    data = load_dataset(name)
    X_all, y_all = data["X"], data["y"]
    cat = list(data.get("cat_cols", []))
    relabel_rows, outlier_rows = [], []
    for seed in SEEDS:
        X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=TEST_SIZE,
                                            random_state=seed, stratify=y_all)
        injected = FaultInjector(random_state=seed, cat_cols=cat).inject_all_faults(X_tr, y_tr, FAULTS)
        X_c, y_c = injected["X_corrupted"], injected["y_corrupted"]
        flipped = injected["masks"]["label_noise"]
        minority = 1 if (y_tr == 1).sum() <= (y_tr == 0).sum() else 0
        n = len(y_c)

        X_imp, y_imp, _ = DataCleaner(seed, cat_cols=cat).impute(X_c, y_c)

        _, y_a, _ = DataCleaner(seed, cat_cols=cat).correct_label_noise(X_imp, y_imp)
        relabel_rows.append({"dataset": name, "seed": seed, "sequence": "IMP→LNR",
                             **relabel_stats(y_c, y_a, flipped, minority),
                             "synthetic_relabelled": 0})

        X_bal, y_bal, bal_meta = DataCleaner(seed, cat_cols=cat).balance(X_imp, y_imp)
        _, y_b, _ = DataCleaner(seed, cat_cols=cat).correct_label_noise(X_bal, y_bal)
        relabel_rows.append({"dataset": name, "seed": seed, "sequence": "IMP→BAL→LNR",
                             **relabel_stats(y_c, y_b[:n], flipped, minority),
                             "synthetic_relabelled": int((y_b[n:] != y_bal[n:]).sum()),
                             "smote_skipped": bal_meta.get("skipped", False)})

        _, kept, _ = DataCleaner(seed, cat_cols=cat).remove_outliers(X_bal, np.arange(len(y_bal)))
        removed = np.setdiff1d(np.arange(len(y_bal)), kept)
        n_syn = len(y_bal) - n
        outlier_rows.append({
            "dataset": name, "seed": seed, "n_rows": len(y_bal), "n_synthetic": n_syn,
            "removed": len(removed), "removed_synthetic": int((removed >= n).sum()),
            "synthetic_share_of_data": n_syn / len(y_bal),
            "synthetic_share_of_removed": (removed >= n).mean() if len(removed) else np.nan,
        })
    return relabel_rows, outlier_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=MAIN_DATASETS)
    args = ap.parse_args()

    out = ROOT / "results" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    relabel, outliers = [], []
    for name in args.datasets:
        r, o = run(name)
        relabel += r
        outliers += o
        print(f"[{name}] done", flush=True)

    rel = pd.DataFrame(relabel)
    rel.to_csv(out / "relabel_quality.csv", index=False)
    pd.DataFrame(outliers).to_csv(out / "outliers_after_smote.csv", index=False)

    summary = (rel.groupby(["dataset", "sequence"])
                  [["relabelled", "fixed", "broken", "precision", "recall", "true_minority_broken"]]
                  .mean().round(3))
    print("\nRelabelling, mean over seeds\n" + summary.to_string())
    osum = pd.DataFrame(outliers).groupby("dataset")[
        ["synthetic_share_of_data", "synthetic_share_of_removed"]].mean().round(3)
    print("\nOutlier removal after SMOTE, mean over seeds\n" + osum.to_string())
    print(f"\nwrote {out.relative_to(ROOT)}/relabel_quality.csv and outliers_after_smote.csv")


if __name__ == "__main__":
    main()
