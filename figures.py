"""
Figures for the paper, drawn only from the files that analyze.py and
diagnostics.py write, so every plotted value is the value in the tables.

    python make_figures.py

Writes figures/fig_*.pdf (vector, for LaTeX) and figures/fig_*.png (300 dpi),
then prints a check of the numbers the text quotes. Any line marked
MISMATCH means the text and the results disagree and must be fixed first.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.metrics.osi import benjamini_hochberg  # noqa: E402

A = ROOT / "results" / "analysis"
DIAG = ROOT / "results" / "diagnostics"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Same names and order as the tables in the paper.
NAMES = {"pima": "Pima", "breast_cancer": "Breast Cancer", "ionosphere": "Ionosphere",
         "banknote": "Banknote", "credit_g": "German Credit", "qsar_biodeg": "QSAR",
         "spambase": "Spambase", "telescope": "Telescope", "adult": "Adult",
         "credit_default": "Credit Default", "ozone": "Ozone", "kc1": "KC1"}
MAIN = ["pima", "breast_cancer", "ionosphere", "banknote", "credit_g", "qsar_biodeg",
        "spambase", "telescope", "adult", "credit_default"]
CLFS = ["logistic_regression", "random_forest", "xgboost"]
CLF_NAME = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
            "xgboost": "XGBoost"}
# Okabe-Ito colours: distinguishable in colour-blind vision and in greyscale print.
CLF_COLOR = {"logistic_regression": "#0072B2", "random_forest": "#D55E00", "xgboost": "#009E73"}
GREY = "#8C8C8C"
FULL_W = 6.6        # inches, full text width of the journal page

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def read(name):
    return pd.read_csv(A / name)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf and .png")


def dataset_axis(ax, order):
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[d] for d in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)


# -- Figure: observed OSI against the noise floor ----------------------------
def fig_noise_floor():
    t = read("osi_main.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), sharey=True, sharex=True)
    for ax, clf in zip(axes, CLFS):
        g = t[t.classifier == clf].set_index("dataset").loc[MAIN]
        y = np.arange(len(MAIN))
        sig = g.p_perm_bh.to_numpy() < 0.05
        ax.hlines(y, g.osi_null_pct, g.osi_pct, color=GREY, linewidth=0.8, zorder=1)
        ax.scatter(g.osi_null_pct, y, s=16, facecolor="white", edgecolor=GREY,
                   linewidth=0.8, zorder=2)
        ax.scatter(g.osi_pct[sig], y[sig], s=22, color=CLF_COLOR[clf], zorder=3)
        ax.scatter(g.osi_pct[~sig], y[~sig], s=22, facecolor="white",
                   edgecolor=CLF_COLOR[clf], linewidth=1.0, zorder=3)
        dataset_axis(ax, MAIN)
        ax.set_title(CLF_NAME[clf])
        ax.set_xlabel("OSI (%)")
    hi = 1.08 * max(t.osi_pct.max(), t.osi_null_pct.max())
    axes[0].set_xlim(0, hi)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, label="Baseline (OSI$_0$)"),
               Line2D([], [], marker="o", ls="", color="black", label="Observed OSI, significant"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="black",
                      label="Observed OSI, not significant")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_noise_floor")


# -- Figure: dose response in label noise rate and oversampling ratio --------
def dose_tables():
    seven = read("osi_noise_05.csv").dataset.unique()
    rows = []
    for stage, kind, x in [("noise_05", "noise", 5), ("noise_10", "noise", 10), ("main", "noise", 15),
                           ("noise_20", "noise", 20), ("smote_050", "ratio", 0.5),
                           ("main", "ratio", 0.8), ("smote_100", "ratio", 1.0)]:
        t = read(f"osi_{stage}.csv")
        t = t[t.dataset.isin(seven)].copy()
        # same correction as Table dose: recomputed over these 21 pairs
        bh = np.array(benjamini_hochberg(t.p_perm.tolist()))
        e = t.osi_excess_pct
        rows.append({"kind": kind, "x": x, "median": e.median(), "q25": e.quantile(0.25),
                     "q75": e.quantile(0.75), "significant": int((bh < 0.05).sum()), "pairs": len(t)})
    return pd.DataFrame(rows)


def fig_dose():
    d = dose_tables()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True,
                             gridspec_kw={"width_ratios": [4, 3]})
    for ax, kind, xlabel in [(axes[0], "noise", "Label noise rate (%)"),
                             (axes[1], "ratio", "SMOTE target ratio")]:
        a = d[d.kind == kind].sort_values("x")
        ax.fill_between(a.x, a.q25, a.q75, color="#D9D9D9", lw=0, label="Interquartile range")
        ax.plot(a.x, a["median"], marker="s", ms=4, lw=1.4, color="black", label="Median")
        for _, r in a.iterrows():
            ax.annotate(f"{int(r.significant)}/{int(r.pairs)}", (r.x, r.q75),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        ax.axhline(0, color=GREY, lw=0.6, ls=":")
        ax.set_xticks(a.x.tolist(), [f"{v:g}" if kind == "noise" else f"{v:.1f}" for v in a.x])
        pad = 0.08 * (a.x.max() - a.x.min())
        ax.set_xlim(a.x.min() - pad, a.x.max() + pad)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        ax.set_axisbelow(True)
    lo, hi = d.q25.min(), d.q75.max()
    axes[0].set_ylim(min(lo, 0) - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))
    axes[0].set_ylabel(r"$\Delta$OSI (percentage points of OSI)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_dose")
    return d


# -- Figure: label correction first minus balancing first, per dataset --------
def fig_lnr_bal():
    t = read("lnr_vs_bal_main.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), sharey=True, sharex=True)
    for ax, clf in zip(axes, CLFS):
        g = t[t.classifier == clf].set_index("dataset").loc[MAIN]
        y = np.arange(len(MAIN))
        m, s = g.mean_diff_pp.to_numpy(), g.sd_diff_pp.to_numpy()
        ax.axvline(0, color="black", lw=0.6)
        ax.errorbar(m, y, xerr=s, fmt="none", ecolor=GREY, elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(m, y, s=22, color=np.where(m < 0, CLF_COLOR[clf], "white"),
                   edgecolor=CLF_COLOR[clf], linewidth=1.0, zorder=2)
        dataset_axis(ax, MAIN)
        ax.set_title(CLF_NAME[clf])
        ax.set_xlabel("LNR first $-$ BAL first (pp)")
    lim = 1.08 * (t.mean_diff_pp.abs() + t.sd_diff_pp).max()
    axes[0].set_xlim(-lim, lim)
    fig.tight_layout()
    save(fig, "fig_lnr_bal")


# -- Figure: the same comparison across conditions -----------------------------
def fig_lnr_bal_conditions():
    stages = [("main", "Main"), ("methods_b", "Second method set"), ("clean", "Natural data"),
              ("noise_05", "Label noise 5%"), ("noise_10", "Label noise 10%"),
              ("noise_20", "Label noise 20%"), ("smote_050", "Ratio 0.5"),
              ("smote_100", "Ratio 1.0"), ("high_ir", "High imbalance")]
    rows = []
    for st, label in stages:
        t = read(f"lnr_vs_bal_{st}_across_datasets.csv").set_index("classifier")
        rows.append([label] + [(t.loc[c, "median_diff_pp"], int(t.loc[c, "datasets_favouring_lnr_first"]),
                                int(t.loc[c, "n_datasets"])) for c in CLFS])
    vals = np.array([[r[i][0] for i in range(1, 4)] for r in rows])
    lim = np.nanmax(np.abs(vals))
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 3.1))
    im = ax.imshow(vals, cmap="PuOr", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, r in enumerate(rows):
        for j in range(3):
            v, k, n = r[j + 1]
            ax.text(j, i, f"{v:+.2f}".replace("-", "\u2212") + f"\n{k}/{n}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(3), ["Logistic\nregression", "Random\nforest", "XGBoost"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("Median LNR first $-$ BAL first (pp)")
    cb.outline.set_linewidth(0.5)
    fig.tight_layout()
    save(fig, "fig_lnr_bal_conditions")


# -- Figure: position effects ------------------------------------------------
def fig_positions():
    p = read("position_effects_main_pooled.csv")
    ops = ["IMP", "OUT", "LNR", "BAL"]
    lim = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35), sharey=True, layout="constrained")
    for k, (ax, clf) in enumerate(zip(axes, CLFS)):
        m = p[p.classifier == clf].pivot(index="operation", columns="position", values="mean").reindex(ops)
        im = ax.imshow(m.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim)
        for i in range(4):
            for j in range(4):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:+.2f}".replace("-", "\u2212"), ha="center", va="center",
                        fontsize=6.6, color="white" if abs(v) > 0.6 * lim else "black")
        ax.set_xticks(range(4), [str(c) for c in m.columns])
        ax.set_yticks(range(4), ops)
        ax.tick_params(length=0, labelleft=(k == 0))
        ax.set_xlabel("Position in pipeline")
        ax.set_title(CLF_NAME[clf])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.015, aspect=18)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"])
    cb.set_label("Difference from mean ordering (pp)")
    cb.outline.set_linewidth(0.5)
    save(fig, "fig_positions")


# -- Figure: precision of label correction, graded against the known flips ---
def fig_precision():
    q = pd.read_csv(DIAG / "relabel_quality.csv")
    pr = q.groupby(["dataset", "sequence"]).precision.mean().unstack()
    a, b = "IMP→LNR", "IMP→BAL→LNR"
    pr = pr.loc[[d for d in MAIN if d in pr.index]].sort_values(b)
    y = np.arange(len(pr))
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 2.9))
    ax.axvline(0.5, color=GREY, lw=0.6, ls=":")
    ax.text(0.5, -0.9, "fixes = breaks", ha="center", fontsize=6.8, color=GREY)
    for i, (d, r) in enumerate(pr.iterrows()):
        if abs(r[b] - r[a]) < 0.015:          # no visible change, an arrow would only clutter
            continue
        ax.annotate("", xy=(r[b], i), xytext=(r[a], i),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", mutation_scale=7,
                                    shrinkA=3, shrinkB=3))
    ax.scatter(pr[a], y, s=20, facecolor="white", edgecolor="black", lw=0.9, zorder=3,
               label="Correction first (IMP$\\to$LNR)")
    ax.scatter(pr[b], y, s=20, color="black", zorder=3, label="Balancing first (IMP$\\to$BAL$\\to$LNR)")
    dataset_axis(ax, list(pr.index))
    ax.set_ylim(len(pr) - 0.5, -1.4)
    ax.set_xlabel("Relabeling precision (fixed / relabeled)")
    ax.set_xlim(0.2, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig_precision")
    return pr


# -- Figure: transfer of a default ordering ------------------------------------
def fig_transfer():
    t = read("rule_transfer_main.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), sharey=True, sharex=True)
    for ax, clf in zip(axes, CLFS):
        g = t[t.classifier == clf].set_index("held_out").loc[MAIN]
        y = np.arange(len(MAIN))
        ax.hlines(y, g.regret_pp, g.random_regret_pp, color=GREY, lw=0.8, zorder=1)
        ax.scatter(g.random_regret_pp, y, s=18, facecolor="white", edgecolor=GREY, lw=0.9, zorder=2)
        ax.scatter(g.regret_pp, y, s=22, color=CLF_COLOR[clf], zorder=3)
        dataset_axis(ax, MAIN)
        ax.set_title(CLF_NAME[clf])
        ax.set_xlabel("Regret on held-out dataset (pp)")
    axes[0].set_xlim(0, 1.08 * max(t.regret_pp.max(), t.random_regret_pp.max()))
    handles = [Line2D([], [], marker="o", ls="", color="black", label="Ordering chosen on the other nine datasets"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, label="Random ordering (expected)")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_transfer")


# -- Checks: every figure-related number the paper quotes ---------------------
def checks(dose, precision):
    out = []

    def check(label, expected, got):
        ok = expected == got
        out.append(ok)
        print(f"  {'OK      ' if ok else 'MISMATCH'} {label}: paper {expected}, results {got}")

    m = read("osi_main.csv")
    sig = m[m.p_perm_bh < 0.05]
    check("main significant pairs", 15, len(sig))
    check("datasets with an effect", 8, sig.dataset.nunique())
    check("excess range, significant pairs", "0.7-3.6",
          f"{sig.osi_excess_pct.min():.1f}-{sig.osi_excess_pct.max():.1f}")
    check("noise floor range", "1.00-8.13", f"{m.osi_null_pct.min():.2f}-{m.osi_null_pct.max():.2f}")
    check("null above half of OSI in every pair", True, bool((m.osi_null_pct > m.osi_pct / 2).all()))
    for st, n in [("methods_b", 15), ("clean", 10), ("high_ir", 6)]:
        check(f"{st} significant pairs", n, int((read(f"osi_{st}.csv").p_perm_bh < 0.05).sum()))
    a = dose
    check("noise 5/10/15/20 significant", [0, 7, 9, 11],
          a[a.kind == "noise"].sort_values("x").significant.astype(int).tolist())
    check("ratio 0.5/0.8/1.0 significant", [0, 9, 6],
          a[a.kind == "ratio"].sort_values("x").significant.astype(int).tolist())
    check("median excess at 5/10/15/20% noise", [0.24, 0.35, 0.68, 1.34],
          [round(v, 2) for v in a[a.kind == "noise"].sort_values("x")["median"]])
    lb = read("lnr_vs_bal_main_across_datasets.csv").set_index("classifier")
    check("main datasets favoring LNR first (LR, RF, XGB)", [2, 3, 3],
          [int(lb.loc[c, "datasets_favouring_lnr_first"]) for c in CLFS])
    rises = int((precision["IMP→BAL→LNR"] > precision["IMP→LNR"]).sum())
    check("datasets where precision rises", 9, rises)
    below = sorted(precision.index[precision["IMP→BAL→LNR"] < 0.5])
    check("datasets below 0.5 after balancing", ["credit_default", "credit_g", "pima"], below)
    tr = read("rule_transfer_main.csv")
    for clf, rule in [("random_forest", "IMP→BAL→LNR→OUT"), ("xgboost", "varies")]:
        check(f"{clf} rule in all folds", rule,
              tr[tr.classifier == clf].rule.mode().iloc[0]
              if tr[tr.classifier == clf].rule.nunique() == 1 else "varies")
    s = read("rule_transfer_main_summary.csv").set_index("classifier")
    red = [round(100 * (1 - s.loc[c, "mean_regret_pp"] / s.loc[c, "mean_random_regret_pp"])) for c in CLFS]
    check("regret reduction % (LR, RF, XGB)", [65, 68, 37], red)
    print(f"\n  {sum(out)} of {len(out)} checks passed"
          + ("" if all(out) else ". Fix every MISMATCH before submitting."))


def main():
    print("figures")
    fig_noise_floor()
    dose = fig_dose()
    fig_lnr_bal()
    fig_lnr_bal_conditions()
    fig_positions()
    precision = fig_precision()
    fig_transfer()
    print("\nchecks against the text")
    checks(dose, precision)


if __name__ == "__main__":
    main()