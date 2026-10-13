"""
robustness_r4.py - the checks a reviewer asked for. No new pipeline runs.

    python robustness_r4.py                    # everything (~30 min, mostly part 1)
    python robustness_r4.py --skip-mechanism   # parts 2-4 only (~3 min)

1. Mechanism without fold dependence
   In IMP->BAL->LNR, SMOTE rows are built from all training rows before the 5-fold
   cross-validation inside LNR, so a row's synthetic neighbours can sit in the folds
   that judge it. Two variants remove that dependence:
     fold-SMOTE : SMOTE (SMOTENC with nominal features) inside each LNR training fold only
     weighted   : no synthetic rows; the correction forest uses balanced class weights
   If precision still rises over IMP->LNR, balance, not dependence, explains the gain.
2. Uncertainty for the main condition
   seed-bootstrap 95% interval for Delta-OSI, and a permutation test of the plain gap
   (best minus worst mean) to show that the OSI normalisation drives no significance call.
3. Transfer of a default ordering
   one-sided exact Wilcoxon test of regret against random regret over the ten held-out
   datasets, and a dataset-bootstrap 95% interval for the regret reduction.
4. 20-seed stability on every dataset present in results_r3/seeds20/.

Writes results/analysis/r4_*.csv and prints the numbers the paper needs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum(1, p[o] * n / (np.arange(n) + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = adj
    return out


def osi_rows(means):
    """means: (..., k) -> OSI along the last axis."""
    best = means.max(axis=-1)
    return (best - means.min(axis=-1)) / best


def perm_means(X, n_perm, rng):
    """X: k x s. Returns (n_perm, k) means after shuffling orderings within each seed."""
    k, s = X.shape
    idx = rng.random((n_perm, k, s)).argsort(axis=1)
    return np.take_along_axis(np.broadcast_to(X, (n_perm, k, s)), idx, axis=1).mean(axis=2)


def load_stage(folder):
    files = sorted(p for p in Path(folder).glob("*.csv") if not p.name.startswith("fault_audit_"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["failed"].eq(False)]


def matrices(df):
    for (ds, clf), g in df.groupby(["dataset", "classifier"]):
        m = g.pivot_table(index="cleaning_order", columns="seed", values="balanced_accuracy", aggfunc="mean")
        if not m.isna().any().any():
            yield ds, clf, m.to_numpy()


# ---------------------------------------------------------------------------- 1. mechanism
def lnr_variant(X, y, seed, cat, mode):
    from imblearn.over_sampling import SMOTE, SMOTENC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    proba = np.zeros((len(y), 2))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr], y[tr]
        if mode == "fold_smote":
            _, c = np.unique(ytr, return_counts=True)
            if len(c) == 2 and c.min() / c.max() < 0.8 * 0.95 and c.min() >= 2:
                k = min(5, int(c.min()) - 1)
                s = (SMOTENC(categorical_features=cat, sampling_strategy=0.8, k_neighbors=k, random_state=seed)
                     if cat else SMOTE(sampling_strategy=0.8, k_neighbors=k, random_state=seed))
                Xtr, ytr = s.fit_resample(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed + fold, n_jobs=-1,
                                    class_weight="balanced" if mode == "weighted" else None)
        rf.fit(Xtr, ytr)
        proba[va] = rf.predict_proba(X[va])
    pred = proba.argmax(axis=1)
    conf = proba[np.arange(len(y)), pred]
    flagged = (pred != y) & (conf >= 0.7)
    out = y.copy()
    out[flagged] = pred[flagged]
    return out


def relabel_stats(y_obs, y_new, flipped):
    ch = y_new != y_obs
    n = int(ch.sum())
    fixed = int((ch & flipped).sum())
    return {"relabelled": n, "fixed": fixed, "broken": n - fixed,
            "precision": fixed / n if n else np.nan}


def part1(datasets):
    from sklearn.model_selection import train_test_split
    from src.data.loaders import load_dataset
    from src.faults.injection import FaultInjector
    from src.cleaning.methods import DataCleaner
    from run_extended import FAULTS, SEEDS, TEST_SIZE

    rows, checked = [], False
    for name in datasets:
        d = load_dataset(name)
        X_all, y_all, cat = d["X"], d["y"], list(d.get("cat_cols", []))
        for seed in SEEDS:
            X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=TEST_SIZE,
                                                random_state=seed, stratify=y_all)
            inj = FaultInjector(random_state=seed, cat_cols=cat).inject_all_faults(X_tr, y_tr, FAULTS)
            X_c, y_c, flipped = inj["X_corrupted"], inj["y_corrupted"], inj["masks"]["label_noise"]
            X_imp, y_imp, _ = DataCleaner(seed, cat_cols=cat).impute(X_c, y_c)
            if not checked:     # the plain variant must reproduce the paper's LNR exactly
                ref = DataCleaner(seed, cat_cols=cat).correct_label_noise(X_imp, y_imp)[1]
                if not np.array_equal(ref, lnr_variant(X_imp, y_imp, seed, cat, "plain")):
                    sys.exit("plain variant does not reproduce DataCleaner.correct_label_noise; stopping")
                print("  check passed: re-implemented LNR matches the paper's LNR exactly", flush=True)
                checked = True
            for mode, label in [("fold_smote", "IMP→LNR (SMOTE inside folds)"),
                                ("weighted", "IMP→LNR (class-weighted)")]:
                rows.append({"dataset": name, "seed": seed, "sequence": label,
                             **relabel_stats(y_c, lnr_variant(X_imp, y_imp, seed, cat, mode), flipped)})
        print(f"  [{name}] done", flush=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "results" / "diagnostics" / "relabel_quality.csv")
    old = old[old["dataset"].isin(datasets)][["dataset", "seed", "sequence", "relabelled", "fixed", "broken", "precision"]]
    allr = pd.concat([old, new], ignore_index=True)
    allr.to_csv(OUT / "r4_mechanism_dependence.csv", index=False)
    s = allr.groupby(["dataset", "sequence"])[["relabelled", "fixed", "broken", "precision"]].mean().round(3)
    print("\n=== 1. Label correction without fold dependence (means over seeds) ===")
    print(s.to_string())
    p = allr.groupby(["dataset", "sequence"])["precision"].mean().unstack()
    base = p["IMP→LNR"]
    for col in ["IMP→BAL→LNR", "IMP→LNR (SMOTE inside folds)", "IMP→LNR (class-weighted)"]:
        print(f"  precision higher than IMP→LNR on {int((p[col] > base).sum())} of {len(p)} datasets: {col}")


# ---------------------------------------------------------------------------- 2. uncertainty
def part2(n_boot=1000, n_perm_boot=1000, n_perm=10000):
    rng = np.random.default_rng(2026)
    df = load_stage(ROOT / "results" / "main")
    rows = []
    for ds, clf, X in matrices(df):
        k, s = X.shape
        osi = osi_rows(X.mean(axis=1))
        pm = perm_means(X, n_perm, rng)
        null = osi_rows(pm)
        gap = X.mean(axis=1).max() - X.mean(axis=1).min()
        gnull = pm.max(axis=1) - pm.min(axis=1)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            Xb = X[:, rng.integers(0, s, s)]
            boots[b] = osi_rows(Xb.mean(axis=1)) - osi_rows(perm_means(Xb, n_perm_boot, rng)).mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"dataset": ds, "classifier": clf, "delta_osi_pct": 100 * (osi - null.mean()),
                     "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                     "p_osi": (1 + (null >= osi - 1e-12).sum()) / (n_perm + 1),
                     "p_gap": (1 + (gnull >= gap - 1e-12).sum()) / (n_perm + 1)})
    t = pd.DataFrame(rows)
    t["p_osi_bh"], t["p_gap_bh"] = bh(t.p_osi), bh(t.p_gap)
    t.to_csv(OUT / "r4_delta_osi_ci.csv", index=False, float_format="%.4g")
    sig = t.p_osi_bh < 0.05
    print("\n=== 2. Delta-OSI with 95% seed-bootstrap intervals (main condition) ===")
    print(t[["dataset", "classifier", "delta_osi_pct", "ci_lo_pct", "ci_hi_pct", "p_osi_bh"]].round(2).to_string(index=False))
    print(f"  significant pairs whose interval excludes zero: {int((sig & (t.ci_lo_pct > 0)).sum())} of {int(sig.sum())}")
    print(f"  significant by OSI: {int(sig.sum())}; by plain gap: {int((t.p_gap_bh < 0.05).sum())}; "
          f"same call in {int(((t.p_gap_bh < 0.05) == sig).sum())} of {len(t)} pairs")


# ---------------------------------------------------------------------------- 3. transfer
def part3(n_boot=10000):
    rng = np.random.default_rng(2026)
    t = pd.read_csv(OUT / "rule_transfer_main.csv")
    rows = []
    print("\n=== 3. Transfer: chosen ordering vs random ordering on held-out datasets ===")
    for clf, g in t.groupby("classifier"):
        r, rr = g.regret_pp.to_numpy(), g.random_regret_pp.to_numpy()
        p = stats.wilcoxon(rr - r, alternative="greater").pvalue
        red = 1 - r.mean() / rr.mean()
        bs = [1 - r[i].mean() / rr[i].mean() for i in (rng.integers(0, len(r), len(r)) for _ in range(n_boot))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"classifier": clf, "reduction_pct": 100 * red, "ci_lo_pct": 100 * lo,
                     "ci_hi_pct": 100 * hi, "wilcoxon_p_one_sided": p})
        print(f"  {clf:<20} regret reduction {100*red:.0f}% (95% CI {100*lo:.0f} to {100*hi:.0f}%), "
              f"one-sided Wilcoxon p = {p:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "r4_transfer_test.csv", index=False, float_format="%.4g")


# ---------------------------------------------------------------------------- 4. 20 seeds
def part4(n_perm=10000):
    rng = np.random.default_rng(2026)
    d20 = load_stage(ROOT / "results_r3" / "seeds20")
    if d20 is None:
        print("\n(no results_r3/seeds20; skipping part 4)")
        return
    d5 = load_stage(ROOT / "results" / "main")
    d5 = d5[d5.dataset.isin(d20.dataset.unique())]
    out = {}
    for label, df in [("5 seeds", d5), ("20 seeds", d20)]:
        rows = []
        for ds, clf, X in matrices(df):
            osi = osi_rows(X.mean(axis=1))
            null = osi_rows(perm_means(X, n_perm, rng))
            rows.append({"dataset": ds, "classifier": clf, "osi_pct": 100 * osi,
                         "excess_pct": 100 * (osi - null.mean()),
                         "p": (1 + (null >= osi - 1e-12).sum()) / (n_perm + 1)})
        t = pd.DataFrame(rows)
        t["p_bh"] = bh(t.p)
        out[label] = t.set_index(["dataset", "classifier"])
    m = out["5 seeds"].join(out["20 seeds"], lsuffix="_5", rsuffix="_20")
    m.reset_index().to_csv(OUT / "r4_seeds20_stability.csv", index=False, float_format="%.4g")
    s5, s20 = m.p_bh_5 < 0.05, m.p_bh_20 < 0.05
    print(f"\n=== 4. 20-seed stability ({m.index.get_level_values(0).nunique()} datasets, {len(m)} pairs) ===")
    print(f"  same call in {int((s5 == s20).sum())} of {len(m)}; significant with 5 seeds only: "
          f"{int((s5 & ~s20).sum())}; with 20 seeds only: {int((~s5 & s20).sum())}")
    for (ds, clf), r in m[s5 != s20].iterrows():
        print(f"    changed: {ds} / {clf}: 5 seeds p_BH = {r.p_bh_5:.3f}, 20 seeds p_BH = {r.p_bh_20:.3f}")
    print(f"  median excess: {m.excess_pct_5.median():.2f} with 5 seeds, {m.excess_pct_20.median():.2f} with 20")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mechanism", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args()
    if not a.skip_mechanism:
        from run_extended import MAIN_DATASETS
        print("=== 1. running the dependence-free correction variants ===", flush=True)
        part1(a.datasets or MAIN_DATASETS)
    part2()
    part3()
    part4()
