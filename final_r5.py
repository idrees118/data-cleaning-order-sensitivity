"""
final_r5.py - the last analysis (~30 min). Two things a reviewer can still ask for.

1. Threshold sensitivity of label correction (LNR uses a fixed 0.7)
   For thresholds 0.6, 0.7, 0.8 and 0.9, on the ten main datasets and five seeds:
     - precision of IMP->LNR and IMP->BAL->LNR against the known flips
     - end-to-end balanced accuracy of IMP->BAL->LNR->OUT and IMP->LNR->BAL->OUT
   The out-of-fold probabilities do not depend on the threshold, so they are computed
   once per input and reused; only the relabelling and later steps change.
   Before anything new is computed, the 0.7 results are checked against the stored
   main results and diagnostics (exact match required).
2. Effect sizes with 95% intervals, one value per dataset (no seed bootstrap):
   Hodges-Lehmann median of the per-dataset differences, with the exact
   Wilcoxon-based interval, for
     - LNR first minus BAL first (main condition, 12 paired orderings)
     - fold-safe correction gain, end to end (from e2e_foldsafe.py)
     - BAL->LNR minus LNR->BAL at each threshold

    python final_r5.py
Writes results/analysis/r5_*.csv and prints the numbers for the paper.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data.loaders import load_dataset                    # noqa: E402
from src.faults.injection import FaultInjector               # noqa: E402
from src.cleaning.methods import DataCleaner                 # noqa: E402
from src.models.train import train_and_evaluate              # noqa: E402
from run_extended import FAULTS, MAIN_DATASETS, SEEDS, TEST_SIZE, CLASSIFIERS   # noqa: E402

A = ROOT / "results" / "analysis"
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


# ------------------------------------------------------------------ helpers
def oof_proba(X, y, seed):
    """Out-of-fold probabilities exactly as DataCleaner.correct_label_noise (no NaNs here)."""
    proba = np.zeros((len(y), 2))
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed + fold, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        proba[va] = rf.predict_proba(X[va])
    return proba


def relabel(y, proba, t):
    pred = proba.argmax(axis=1)
    flag = (pred != y) & (proba[np.arange(len(y)), pred] >= t)
    out = y.copy()
    out[flag] = pred[flag]
    return out


def precision(y_obs, y_new, flipped):
    ch = y_new != y_obs
    return (ch & flipped).sum() / ch.sum() if ch.sum() else np.nan


def hodges_lehmann(d, alpha=0.05):
    """HL estimate and exact Wilcoxon-based interval for a one-sample location."""
    d = np.asarray(d, float)
    n = len(d)
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    m = len(walsh)
    # exact null distribution of the signed-rank statistic
    ranks = np.arange(1, n + 1)
    counts = np.zeros(m + 1)
    for signs in itertools.product([0, 1], repeat=n):
        counts[int(np.dot(signs, ranks))] += 1
    cdf = np.cumsum(counts) / counts.sum()
    k = int(np.searchsorted(cdf, alpha / 2, side="right"))      # P(T <= k-1) <= alpha/2
    lo, hi = walsh[k - 1], walsh[m - k]                          # W_(d), W_(M+1-d) with d = k
    return float(np.median(walsh)), float(lo), float(hi)


def report(label, d):
    est, lo, hi = hodges_lehmann(d)
    p = stats.wilcoxon(d).pvalue
    print(f"  {label:<46} {int((np.asarray(d) > 0).sum()):>2}/{len(d)} positive, "
          f"HL {est:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f}), p = {p:.3f}")
    return {"comparison": label, "n": len(d), "positive": int((np.asarray(d) > 0).sum()),
            "hl_pp": est, "ci_lo_pp": lo, "ci_hi_pp": hi, "wilcoxon_p": p}


# ------------------------------------------------------------------ 1. thresholds
def thresholds():
    main_df = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results" / "main").glob("*.csv"))
                         if not f.name.startswith("fault_audit_")], ignore_index=True)
    stored = (main_df[main_df.cleaning_order.isin(["IMP→BAL→LNR→OUT", "IMP→LNR→BAL→OUT"])]
              .pivot_table(index=["dataset", "seed", "classifier"], columns="cleaning_order",
                           values="balanced_accuracy"))
    acc_rows, prec_rows, checked = [], [], False
    for name in MAIN_DATASETS:
        d = load_dataset(name)
        X_all, y_all, cat = d["X"], d["y"], list(d.get("cat_cols", []))
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE,
                                                      random_state=seed, stratify=y_all)
            inj = FaultInjector(random_state=seed, cat_cols=cat).inject_all_faults(X_tr, y_tr, FAULTS)
            y_c, flipped = inj["y_corrupted"], inj["masks"]["label_noise"]
            X_imp, y_imp, _ = DataCleaner(seed, cat_cols=cat).impute(inj["X_corrupted"], y_c)
            n = len(y_imp)
            X_bal, y_bal, _ = DataCleaner(seed, cat_cols=cat).balance(X_imp, y_imp)
            p_first = oof_proba(X_imp, y_imp, seed)        # LNR on imbalanced data
            p_after = oof_proba(X_bal, y_bal, seed)        # LNR after balancing
            for t in THRESHOLDS:
                y_b = relabel(y_imp, p_first, t)            # B: IMP->LNR->BAL->OUT
                Xb, yb, _ = DataCleaner(seed, cat_cols=cat).balance(X_imp, y_b)
                Xb, yb, _ = DataCleaner(seed, cat_cols=cat).remove_outliers(Xb, yb)
                y_a = relabel(y_bal, p_after, t)            # A: IMP->BAL->LNR->OUT
                Xa, ya, _ = DataCleaner(seed, cat_cols=cat).remove_outliers(X_bal, y_a)
                res = {c: (train_and_evaluate(Xa, ya, X_te, y_te, c, random_state=seed)["balanced_accuracy"],
                           train_and_evaluate(Xb, yb, X_te, y_te, c, random_state=seed)["balanced_accuracy"])
                       for c in CLASSIFIERS}
                if t == 0.7 and not checked:
                    ref = DataCleaner(seed, cat_cols=cat).correct_label_noise(X_imp, y_imp)[1]
                    if not np.array_equal(ref, y_b):
                        sys.exit("threshold 0.7 does not reproduce the paper's LNR; stopping")
                    diff = max(max(abs(res[c][0] - stored.loc[(name, seed, c), "IMP→BAL→LNR→OUT"]),
                                   abs(res[c][1] - stored.loc[(name, seed, c), "IMP→LNR→BAL→OUT"]))
                               for c in CLASSIFIERS)
                    if diff > 1e-9:
                        sys.exit(f"threshold 0.7 pipelines differ from the main results by {diff:.2e}; stopping")
                    print("  check passed: threshold 0.7 reproduces the paper's LNR and both main pipelines exactly",
                          flush=True)
                    checked = True
                prec_rows.append({"dataset": name, "seed": seed, "threshold": t,
                                  "precision_lnr_first": precision(y_c, y_b, flipped),
                                  "precision_bal_first": precision(y_c, y_a[:n], flipped)})
                for c, (a, b) in res.items():
                    acc_rows.append({"dataset": name, "seed": seed, "threshold": t, "classifier": c,
                                     "bal_first": a, "lnr_first": b})
        print(f"  [{name}] done", flush=True)
    acc, prec = pd.DataFrame(acc_rows), pd.DataFrame(prec_rows)
    acc.to_csv(A / "r5_threshold_accuracy.csv", index=False)
    prec.to_csv(A / "r5_threshold_precision.csv", index=False)
    return acc, prec


# ------------------------------------------------------------------ main
def main():
    print("=== 1. threshold sensitivity (computing) ===", flush=True)
    acc, prec = thresholds()
    rows = []

    print("\n=== 1a. precision of label correction: balancing first vs correcting first ===")
    pm = prec.groupby(["threshold", "dataset"]).mean(numeric_only=True)
    for t in THRESHOLDS:
        g = pm.loc[t]
        print(f"  threshold {t}: precision higher with balancing first on "
              f"{int((g.precision_bal_first > g.precision_lnr_first).sum())} of {len(g)} datasets; "
              f"median {g.precision_lnr_first.median():.2f} -> {g.precision_bal_first.median():.2f}")

    print("\n=== 1b. end to end: IMP→BAL→LNR→OUT minus IMP→LNR→BAL→OUT, per dataset (pp) ===")
    am = acc.groupby(["threshold", "classifier", "dataset"]).mean(numeric_only=True)
    for t in THRESHOLDS:
        for c in CLASSIFIERS:
            g = am.loc[(t, c)]
            rows.append({"threshold": t, **report(f"t={t} {c}", 100 * (g.bal_first - g.lnr_first).to_numpy())})

    print("\n=== 2. effect sizes with 95% intervals, one value per dataset ===")
    l = pd.read_csv(A / "lnr_vs_bal_main.csv")
    for c in CLASSIFIERS:
        g = l[l.classifier == c]
        rows.append(report(f"main: BAL first minus LNR first, {c}", -g.mean_diff_pp.to_numpy()))
    e = pd.read_csv(A / "r4_e2e_foldsafe.csv").groupby(["classifier", "dataset"]).mean(numeric_only=True)
    for c in CLASSIFIERS:
        g = e.loc[c]
        rows.append(report(f"fold-safe correction gain, {c}", 100 * (g.C_foldsmote - g.B_lnr_first).to_numpy()))
    pd.DataFrame(rows).to_csv(A / "r5_effect_sizes.csv", index=False, float_format="%.4g")
    print("\nwrote results/analysis/r5_threshold_*.csv and r5_effect_sizes.csv")


if __name__ == "__main__":
    main()
