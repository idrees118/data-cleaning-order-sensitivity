"""
Analysis of the extended runs. Reads results/<stage>/*.csv (and the two A/B
worktrees next to this folder, if present) and writes tables to results/analysis/.

    python analyze.py

Statistics, and why they differ from src/metrics/osi.py
  * OSI is computed from per-ordering means over seeds, as in the paper.
  * Significance of OSI comes from a permutation test: ordering labels are
    shuffled within each seed (the seed fixes the split and the injected faults),
    and OSI is recomputed. This tests OSI itself and also gives the OSI expected
    from noise alone. Max-minus-min over 24 orderings is never zero, even when
    ordering does not matter, so the excess over that null is reported too.
  * The old report t-tested the best against the worst ordering after picking
    them from the same data. That selection inflates the effect, so those
    p-values are not valid and are not used here.
  * A Friedman test over all orderings (seeds as blocks) with Kendall's W is
    given as the omnibus check.
  * Label noise correction before vs after SMOTE is a planned comparison:
    each ordering is paired with the ordering that swaps only those two steps.
    With 5 seeds a per-dataset test cannot reach p < 0.05 in a signed-rank test
    (the smallest two-sided p is 0.0625), so inference is made across datasets.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.metrics.osi import benjamini_hochberg  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "analysis"
N_PERM = 10000
RNG = np.random.default_rng(2026)
KEY = ["dataset", "classifier", "seed", "cleaning_order"]


# -- loading ----------------------------------------------------------------
def load_stage(stage, base=RESULTS):
    folder = base / stage
    files = sorted(p for p in folder.glob("*.csv") if not p.name.startswith("fault_audit_"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_failed = int(df["failed"].eq(True).sum())
    if n_failed:
        print(f"  {stage}: dropping {n_failed} failed runs (see the 'error' column)")
    return df[df["failed"].eq(False)].copy()


def seed_matrix(sub):
    """Orderings x seeds matrix of balanced accuracy. Fails loudly on gaps or duplicates."""
    if sub.duplicated(["cleaning_order", "seed"]).any():
        raise ValueError("duplicate (ordering, seed) rows")
    mat = sub.pivot(index="cleaning_order", columns="seed", values="balanced_accuracy")
    if mat.isna().any().any():
        raise ValueError("missing (ordering, seed) cells")
    return mat


def complete_groups(df):
    """Yield (dataset, classifier, rows) where every ordering has every seed.
    Incomplete groups (failed runs) are reported and left out of the statistics."""
    by_ds = df.groupby("dataset")
    full = by_ds["cleaning_order"].nunique() * by_ds["seed"].nunique()
    for (ds, clf), sub in df.groupby(["dataset", "classifier"]):
        if len(sub.drop_duplicates(["cleaning_order", "seed"])) < full[ds]:
            print(f"  skipping {ds} / {clf}: {full[ds] - len(sub)} runs missing (failed)")
            continue
        yield ds, clf, sub


# -- OSI and its tests --------------------------------------------------------
def osi_of_means(means):
    best = means.max(axis=-1)
    return (best - means.min(axis=-1)) / best


def osi_permutation(X):
    """X: orderings x seeds. Returns observed OSI, mean null OSI, permutation p."""
    k, s = X.shape
    observed = osi_of_means(X.mean(axis=1))
    idx = RNG.random((N_PERM, k, s)).argsort(axis=1)        # shuffle within each seed
    shuffled = np.take_along_axis(np.broadcast_to(X, (N_PERM, k, s)), idx, axis=1)
    null = osi_of_means(shuffled.mean(axis=2))
    p = (1 + np.sum(null >= observed - 1e-12)) / (N_PERM + 1)
    return observed, float(null.mean()), float(p)


def friedman(X):
    k, s = X.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            chi2, p = stats.friedmanchisquare(*X)
        except ValueError:
            chi2, p = np.nan, np.nan
    if not np.isfinite(chi2):                                # every value tied
        return np.nan, np.nan, np.nan
    return chi2, p, chi2 / (s * (k - 1))


def osi_table(df, stage):
    rows = []
    for ds, clf, sub in complete_groups(df):
        mat = seed_matrix(sub)
        X = mat.to_numpy()
        means, sds = mat.mean(axis=1), mat.std(axis=1, ddof=1)
        best, worst = means.idxmax(), means.idxmin()
        osi, null_osi, p_perm = osi_permutation(X)
        chi2, p_fr, w = friedman(X)
        rows.append({
            "stage": stage, "dataset": ds, "classifier": clf,
            "n_orderings": X.shape[0], "n_seeds": X.shape[1],
            "best_order": best, "best_mean": means[best], "best_sd": sds[best],
            "worst_order": worst, "worst_mean": means[worst], "worst_sd": sds[worst],
            "gap_pp": 100 * (means[best] - means[worst]),
            "osi_pct": 100 * osi, "osi_null_pct": 100 * null_osi,
            "osi_excess_pct": 100 * (osi - null_osi), "p_perm": p_perm,
            "friedman_chi2": chi2, "friedman_p": p_fr, "kendall_w": w,
        })
    t = pd.DataFrame(rows)
    t["p_perm_bh"] = benjamini_hochberg(t["p_perm"].tolist())
    return t


# -- label noise correction vs SMOTE ------------------------------------------
def swap_lnr_bal(order):
    steps = order.split("→")
    i, j = steps.index("LNR"), steps.index("BAL")
    steps[i], steps[j] = steps[j], steps[i]
    return "→".join(steps)


def lnr_first(order):
    steps = order.split("→")
    return steps.index("LNR") < steps.index("BAL")


def lnr_bal_tables(df, stage):
    rows = []
    for ds, clf, sub in complete_groups(df):
        mat = seed_matrix(sub)
        firsts = [o for o in mat.index if lnr_first(o)]
        diffs = np.array([mat.loc[o].to_numpy() - mat.loc[swap_lnr_bal(o)].to_numpy()
                          for o in firsts])                  # pairs x seeds
        per_seed = diffs.mean(axis=0)
        act_col = ("label_noise__n_removed_noise"
                   if sub.get("label_noise__n_removed_noise", pd.Series(dtype=float)).notna().any()
                   else "n_relabelled")
        relab = sub.groupby(sub["cleaning_order"].map(lnr_first))[act_col].mean()
        rows.append({
            "stage": stage, "dataset": ds, "classifier": clf,
            "n_pairs": diffs.shape[0], "n_seeds": diffs.shape[1],
            "mean_diff_pp": 100 * per_seed.mean(),
            "sd_diff_pp": 100 * per_seed.std(ddof=1),
            "seeds_favouring_lnr_first": int((per_seed > 0).sum()),
            "pairs_favouring_lnr_first": float((diffs > 0).mean()),
            "label_step_measure": act_col,
            "label_actions_lnr_first": relab.get(True, np.nan),
            "label_actions_bal_first": relab.get(False, np.nan),
        })
    per_ds = pd.DataFrame(rows)

    across = []
    for clf, sub in per_ds.groupby("classifier"):
        d = sub["mean_diff_pp"].to_numpy()
        try:
            _, p = stats.wilcoxon(d)
        except ValueError:
            p = np.nan
        across.append({"stage": stage, "classifier": clf, "n_datasets": len(d),
                       "datasets_favouring_lnr_first": int((d > 0).sum()),
                       "median_diff_pp": float(np.median(d)), "wilcoxon_p": p})
    return per_ds, pd.DataFrame(across)


# -- stage-specific helpers -----------------------------------------------------
def collapse_noop_imputation(df):
    """With no missing values, imputation does nothing, so the 24 orderings are 6
    orderings of the other steps, each run 4 times. Check the copies agree, then drop IMP."""
    df = df.copy()
    noop = df["imputation__n_imputed"].fillna(0).eq(0).groupby(df["dataset"]).transform("all")
    short = df["cleaning_order"].str.replace("IMP→", "", regex=False).str.replace("→IMP", "", regex=False)
    df.loc[noop, "cleaning_order"] = short[noop]
    spread = df[noop].groupby(KEY)["balanced_accuracy"].agg(lambda s: s.max() - s.min())
    if len(spread) and spread.max() > 1e-12:
        print(f"  WARNING: equivalent orderings differ by up to {spread.max():.2e}; not deterministic")
    else:
        print("  clean stage: equivalent orderings are identical (determinism check passed)")
    return df.drop_duplicates(KEY)


def subsets_table(sub_df, main_df):
    allseq = pd.concat([sub_df, main_df[main_df["dataset"].isin(sub_df["dataset"].unique())]],
                       ignore_index=True)
    rows = []
    for (ds, clf), g in allseq.groupby(["dataset", "classifier"]):
        means = g.groupby("cleaning_order")["balanced_accuracy"].mean()
        full = means[[o for o in means.index if o.count("→") == 3]]
        best = means.idxmax()
        rows.append({
            "dataset": ds, "classifier": clf, "n_sequences": len(means),
            "best_sequence": best, "best_len": 0 if best == "NONE" else best.count("→") + 1,
            "best_mean": means[best], "best_full_order": full.idxmax(),
            "best_full_mean": full.max(), "best_minus_best_full_pp": 100 * (means[best] - full.max()),
            "no_cleaning_mean": means.get("NONE", np.nan),
            "osi_all_sequences_pct": 100 * (means.max() - means.min()) / means.max(),
        })
    return pd.DataFrame(rows)


def ab_table(fixed, variant, label):
    m = fixed.merge(variant, on=KEY, suffixes=("_fixed", "_var"))
    steps = m["cleaning_order"].str.split("→")
    unaffected = steps.map(lambda t: t.index("IMP") < min(t.index("LNR"), t.index("BAL")))
    m["diff_pp"] = 100 * (m["balanced_accuracy_fixed"] - m["balanced_accuracy_var"])
    rows = []
    for (ds, clf), g in m.groupby(["dataset", "classifier"]):
        u = unaffected[g.index]
        osi_f = g.groupby("cleaning_order")["balanced_accuracy_fixed"].mean()
        osi_v = g.groupby("cleaning_order")["balanced_accuracy_var"].mean()
        rows.append({
            "comparison": label, "dataset": ds, "classifier": clf,
            "max_abs_diff_unaffected_pp": g.loc[u, "diff_pp"].abs().max(),
            "mean_diff_affected_pp": g.loc[~u, "diff_pp"].mean(),
            "mean_abs_diff_affected_pp": g.loc[~u, "diff_pp"].abs().mean(),
            "osi_fixed_pct": 100 * (osi_f.max() - osi_f.min()) / osi_f.max(),
            "osi_variant_pct": 100 * (osi_v.max() - osi_v.min()) / osi_v.max(),
            "best_fixed": osi_f.idxmax(), "best_variant": osi_v.idxmax(),
            "worst_fixed": osi_f.idxmin(), "worst_variant": osi_v.idxmin(),
        })
    return pd.DataFrame(rows)


# -- position effects, extremes, OSI vs imbalance, transfer of an ordering rule ----
OPS = ["IMP", "OUT", "LNR", "BAL"]


def centred(df):
    """Balanced accuracy minus the mean over orderings, within dataset x classifier x seed.
    Only complete dataset x classifier groups are used, so every mean covers all orderings."""
    df = pd.concat([sub for _, _, sub in complete_groups(df)], ignore_index=True)
    g = df.groupby(["dataset", "classifier", "seed"])["balanced_accuracy"]
    return df.assign(centred=df["balanced_accuracy"] - g.transform("mean"))


def position_effects(df):
    c = centred(df)
    rows = []
    for op in OPS:
        pos = c["cleaning_order"].str.split("→").map(lambda s: s.index(op) + 1)
        g = c.assign(position=pos).groupby(["dataset", "classifier", "position"])["centred"].mean()
        for (ds, clf, p), v in g.items():
            rows.append({"dataset": ds, "classifier": clf, "operation": op,
                         "position": p, "centred_pp": 100 * v})
    per_ds = pd.DataFrame(rows)
    pooled = (per_ds.groupby(["classifier", "operation", "position"])["centred_pp"]
                    .agg(mean="mean", sd="std", n_datasets="size").reset_index())
    return per_ds, pooled


def plot_position_effects(pooled, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping the figure")
        return
    clfs = sorted(pooled["classifier"].unique())
    lim = np.nanmax(np.abs(pooled["mean"])) or 1.0
    fig, axes = plt.subplots(1, len(clfs), figsize=(4.2 * len(clfs), 3.4), squeeze=False)
    for ax, clf in zip(axes[0], clfs):
        m = (pooled[pooled["classifier"] == clf]
             .pivot(index="operation", columns="position", values="mean").reindex(OPS))
        im = ax.imshow(m.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_xticks(range(4), [str(p) for p in m.columns])
        ax.set_yticks(range(4), m.index)
        ax.set_xlabel("position in pipeline")
        ax.set_title(clf.replace("_", " "))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m.iat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8,
                 label="balanced accuracy vs. mean ordering (pp)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/analysis/{path.name}")


def extreme_positions(osi_tab):
    rows = []
    for kind in ("best", "worst"):
        for _, r in osi_tab.iterrows():
            steps = r[f"{kind}_order"].split("→")
            rows += [{"kind": kind, "classifier": r["classifier"], "operation": op,
                      "position": steps.index(op) + 1} for op in OPS]
    counts = pd.DataFrame(rows).groupby(["kind", "operation", "position"]).size()
    return counts.unstack("position", fill_value=0).reset_index()


def osi_vs_ir(osi_tab, audit):
    ir = audit.set_index("dataset")["minority_share"].map(lambda s: (1 - s) / s)
    rows = []
    for clf, g in osi_tab.groupby("classifier"):
        g = g.assign(ir=g["dataset"].map(ir)).dropna(subset=["ir"])
        for col in ("osi_pct", "osi_excess_pct"):
            rho, p = stats.spearmanr(g["ir"], g[col])
            rows.append({"classifier": clf, "measure": col, "n_datasets": len(g),
                         "spearman_rho": rho, "p": p})
    return pd.DataFrame(rows)


def leave_one_dataset_out(df):
    """Pick the ordering with the best mean centred accuracy on the other datasets and
    score it on the held-out one. Regret is measured against the held-out best ordering;
    a random ordering's expected regret is the baseline."""
    c = centred(df)
    m = c.groupby(["classifier", "dataset", "cleaning_order"])["centred"].mean()
    rows = []
    for clf in m.index.get_level_values(0).unique():
        mc = m.loc[clf]
        for ds in mc.index.get_level_values(0).unique():
            rule = mc.drop(index=ds, level=0).groupby(level=1).mean().idxmax()
            held = mc.loc[ds]
            rows.append({"classifier": clf, "held_out": ds, "rule": rule,
                         "rank_of_rule": int((held > held[rule]).sum()) + 1,
                         "regret_pp": 100 * (held.max() - held[rule]),
                         "random_regret_pp": 100 * (held.max() - held.mean())})
    per = pd.DataFrame(rows)
    summary = (per.groupby("classifier")
                  .agg(n_datasets=("held_out", "size"), median_rank=("rank_of_rule", "median"),
                       mean_regret_pp=("regret_pp", "mean"),
                       mean_random_regret_pp=("random_regret_pp", "mean"),
                       datasets_rule_beats_random=("regret_pp",
                                                   lambda s: int((s < per.loc[s.index, "random_regret_pp"]).sum())))
                  .reset_index())
    return per, summary


# -- main ---------------------------------------------------------------------------
ROUND2_STAGES = ["methods_b", "high_ir", "smote_050", "smote_100", "noise_05", "noise_10", "noise_20", "noise_sym"]

def save(df, name):
    df.to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"  wrote results/analysis/{name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_df = load_stage("main")
    if main_df is None:
        sys.exit("No results in results/main yet.")

    print("main stage")
    osi_main = osi_table(main_df, "main")
    save(osi_main, "osi_main.csv")
    per_ds, across = lnr_bal_tables(main_df, "main")
    save(per_ds, "lnr_vs_bal_main.csv")
    save(across, "lnr_vs_bal_main_across_datasets.csv")
    pos_ds, pos_pooled = position_effects(main_df)
    save(pos_ds, "position_effects_main.csv")
    save(pos_pooled, "position_effects_main_pooled.csv")
    plot_position_effects(pos_pooled, OUT / "position_effects_main.png")
    save(extreme_positions(osi_main), "extreme_positions_main.csv")
    lodo, lodo_sum = leave_one_dataset_out(main_df)
    save(lodo, "rule_transfer_main.csv")
    save(lodo_sum, "rule_transfer_main_summary.csv")

    audits = sorted((RESULTS / "main").glob("fault_audit_*.csv"))
    if audits:
        a = pd.concat([pd.read_csv(f) for f in audits])
        audit = a.groupby("dataset").mean(numeric_only=True).drop(columns="seed").reset_index()
        save(audit, "fault_audit_main.csv")
        save(osi_vs_ir(osi_main, audit), "osi_vs_ir_main.csv")
    skips = main_df.groupby(["dataset", "cleaning_order"])["smote_skipped"].mean().reset_index()
    save(skips[skips["smote_skipped"] > 0], "smote_skips_main.csv")

    clean_df = load_stage("clean")
    if clean_df is not None:
        print("clean stage")
        clean_df = collapse_noop_imputation(clean_df)
        save(osi_table(clean_df, "clean"), "osi_clean.csv")
        per_ds, across = lnr_bal_tables(clean_df, "clean")
        save(per_ds, "lnr_vs_bal_clean.csv")
        save(across, "lnr_vs_bal_clean_across_datasets.csv")

    sub_df = load_stage("subsets")
    if sub_df is not None:
        print("subsets stage")
        save(subsets_table(sub_df, main_df), "subsets_vs_full.csv")

    sens = [osi_main.assign(stage="main")]
    for stage in ROUND2_STAGES:
        df = load_stage(stage)
        if df is None:
            continue
        print(f"{stage} stage")
        tab = osi_table(df, stage)
        save(tab, f"osi_{stage}.csv")
        per_ds, across = lnr_bal_tables(df, stage)
        save(per_ds, f"lnr_vs_bal_{stage}.csv")
        save(across, f"lnr_vs_bal_{stage}_across_datasets.csv")
        if stage == "methods_b":
            save(position_effects(df)[1], "position_effects_methods_b_pooled.csv")
        sens.append(tab)
    if len(sens) > 1:
        s = pd.concat(sens, ignore_index=True)
        wide = s.pivot_table(index=["dataset", "classifier"], columns="stage",
                             values=["osi_pct", "osi_excess_pct", "p_perm"]).round(4)
        wide.columns = [f"{v}__{st}" for v, st in wide.columns]
        save(wide.reset_index(), "robustness_summary.csv")

    ab = []
    for folder, label in [("order_ab_leaky", "fixed_vs_leaky"), ("order_ab_fix12", "fixed_vs_fix12")]:
        var = load_stage("main", ROOT.parent / folder / "results")
        if var is not None:
            ab.append(ab_table(main_df, var, label))
    if ab:
        print("leak A/B")
        save(pd.concat(ab, ignore_index=True), "leak_ab.csv")
    print("done")


if __name__ == "__main__":
    main()
