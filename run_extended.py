"""
Extended experiment runner for the preprocessing-order study.

This file sits next to run_all.py and does not modify it, so the original
study stays reproducible. What it does differently:

  * Cleaning is done once per (seed, ordering) and shared by all classifiers.
    The cleaner is seeded, so the cleaned data is identical to cleaning once
    per classifier, at about a third of the cost.
  * One CSV per dataset, checkpointed per seed: safe to stop and resume.
  * A fault audit (realised label noise and missingness) per dataset and seed.
  * Extra stages: natural data without injected faults, and ordered subsets
    of the four steps.

Stages (round 1)
  main       all 24 orderings, injected faults (the fault model of the paper)
  clean      all 24 orderings, no injected faults (natural data only)
  subsets    every ordered subset of 0-3 steps, injected faults
             (subsets of size 4 are exactly the orderings in `main`)
Stages (round 2)
  methods_b  as main, with a second method for every step (DataCleanerB)
  high_ir    as main on the high-imbalance datasets, minority-anchored label noise
  smote_050, smote_100        as main with SMOTE target ratio 0.5 / 1.0
  noise_05, noise_10, noise_20  as main with 5 / 10 / 20% label noise

Examples
  python run_extended.py --stage main --datasets pima
  python run_extended.py --stage methods_b
  python run_extended.py --stage subsets --smoke
  python run_extended.py --stage main --datasets pima --results-dir results_check
"""
import argparse
import hashlib
import inspect
import itertools
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import LOADERS, load_dataset      # noqa: E402
from src.faults.injection import FaultInjector          # noqa: E402
from src.cleaning.methods import DataCleaner            # noqa: E402
from src.cleaning.alternatives import DataCleanerB, DataCleanerRatio  # noqa: E402
from src.models.train import train_and_evaluate         # noqa: E402

# -- configuration (kept identical to run_all.py) ----------------------------
SEEDS = [42, 43, 44, 45, 46]
CLASSIFIERS = ["random_forest", "logistic_regression", "xgboost"]
OPERATIONS = ["imputation", "outliers", "label_noise", "balance"]
LABEL = {"imputation": "IMP", "outliers": "OUT", "label_noise": "LNR", "balance": "BAL"}
TEST_SIZE = 0.20

FAULTS = {
    "label_noise": {"rate": 0.15, "type": "asymmetric"},
    "missing": {"rate": 0.10, "mechanism": "MAR"},
    "outliers": {"rate": 0.05},
}

# Listed small to large so that a problem with a loader shows up early.
MAIN_DATASETS = ["pima", "breast_cancer", "ionosphere", "banknote", "credit_g",
                 "qsar_biodeg", "spambase", "telescope", "adult", "credit_default"]
CLEAN_ONLY = ["ozone", "kc1"]          # high imbalance: only meaningful on natural data
SUBSET_DATASETS = ["pima", "breast_cancer", "ionosphere", "banknote", "credit_g",
                   "qsar_biodeg", "spambase"]


def faults_with(**label):
    """FAULTS with the label-noise settings replaced (fresh dicts, FAULTS is not modified)."""
    out = {k: dict(v) for k, v in FAULTS.items()}
    out["label_noise"].update(label)
    return out


# Each stage: which sequences, which faults, which cleaner, which datasets.
STAGES = {
    "main":      dict(seqs="all", faults=FAULTS, cleaner=("A", {}), datasets=MAIN_DATASETS),
    "clean":     dict(seqs="all", faults={}, cleaner=("A", {}), datasets=MAIN_DATASETS + CLEAN_ONLY),
    "subsets":   dict(seqs="subsets", faults=FAULTS, cleaner=("A", {}), datasets=SUBSET_DATASETS),
    "methods_b": dict(seqs="all", faults=FAULTS, cleaner=("B", {}), datasets=MAIN_DATASETS),
    "high_ir":   dict(seqs="all", faults=faults_with(type="minority_anchored"),
                      cleaner=("A", {}), datasets=CLEAN_ONLY),
    "smote_050": dict(seqs="all", faults=FAULTS, cleaner=("ratio", {"ratio": 0.5}),
                      datasets=SUBSET_DATASETS),
    "smote_100": dict(seqs="all", faults=FAULTS, cleaner=("ratio", {"ratio": 1.0}),
                      datasets=SUBSET_DATASETS),
    "noise_05":  dict(seqs="all", faults=faults_with(rate=0.05), cleaner=("A", {}),
                      datasets=SUBSET_DATASETS),
    "noise_10":  dict(seqs="all", faults=faults_with(rate=0.10), cleaner=("A", {}),
                      datasets=SUBSET_DATASETS),
    "noise_20":  dict(seqs="all", faults=faults_with(rate=0.20), cleaner=("A", {}),
                      datasets=SUBSET_DATASETS),
    "noise_sym": dict(seqs="all", faults=faults_with(type="symmetric"), cleaner=("A", {}),
                      datasets=SUBSET_DATASETS),
    "seeds20":   dict(seqs="all", faults=FAULTS, cleaner=("A", {}),
                      datasets=["pima", "qsar_biodeg", "spambase"],
                      seeds=list(range(42, 62))),
}


def make_cleaner(spec, seed, cat_cols=()):
    kind, kwargs = spec
    cls = {"A": DataCleaner, "B": DataCleanerB, "ratio": DataCleanerRatio}.get(kind)
    if cls is None:
        raise ValueError(f"unknown cleaner {kind!r}")
    extra = {}
    if cat_cols:
        if "cat_cols" not in inspect.signature(cls.__init__).parameters:
            raise TypeError(f"{cls.__name__} has no cat_cols argument; update "
                            f"src/cleaning/alternatives.py before running this stage")
        extra["cat_cols"] = list(cat_cols)
    return cls(random_state=seed, **kwargs, **extra)

# Without imputation the training data still has NaNs. XGBoost handles them
# natively, random forests do from scikit-learn 1.4, logistic regression never.
SKLEARN_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
RF_HANDLES_NAN = SKLEARN_VERSION >= (1, 4)


# -- helpers ------------------------------------------------------------------
def sequences(kind):
    if kind == "all":
        return list(itertools.permutations(OPERATIONS))
    seqs = []
    for k in range(0, len(OPERATIONS)):            # sizes 0..3
        seqs += list(itertools.permutations(OPERATIONS, k))
    return seqs


def classifiers_for(seq, base):
    if "imputation" in seq:
        return list(base)
    nan_ok = {"xgboost"} | ({"random_forest"} if RF_HANDLES_NAN else set())
    return [c for c in base if c in nan_ok]


def code_id():
    """Fingerprint of the code that produces results: a hash of every .py file under
    src/ plus this script. Unrelated git commits do not change it, edits to the
    pipeline do, so checkpoints from different code are never mixed."""
    h = hashlib.sha1()
    for f in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


def write_csv_atomic(df, path):
    """Write to a temporary file and rename, so a crash never leaves a half-written CSV."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def audit(name, seed, y_true, X_c, y_c):
    """Realised fault rates. The injector flips 15% of all labels, 70% of them
    from the minority class, so on skewed data most minority labels can be hit."""
    minority = 1 if (y_true == 1).sum() <= (y_true == 0).sum() else 0
    m = y_true == minority
    return {
        "dataset": name, "seed": seed, "n_train": int(len(y_true)),
        "minority_share": float(m.mean()),
        "labels_flipped": float((y_c != y_true).mean()),
        "minority_labels_flipped": float((y_c[m] != y_true[m]).mean()),
        "missing_cells": float(np.isnan(X_c).mean()),
    }


def evaluate_sequence(seq, seed, X_c, y_c, X_te, y_te, classifiers, cleaner_spec, cat_cols=()):
    order = "→".join(LABEL[o] for o in seq) or "NONE"
    base = {"cleaning_order": order, "permutation": str(seq),
            "seq_len": len(seq), "seed": seed, "n_train_input": int(len(y_c))}
    clfs = classifiers_for(seq, classifiers)

    try:
        X_cl, y_cl, log = make_cleaner(cleaner_spec, seed, cat_cols).apply_sequence(X_c, y_c, list(seq))
    except Exception as e:                          # cleaning itself failed
        return [{**base, "classifier": c, "balanced_accuracy": np.nan,
                 "failed": True, "error": f"cleaning: {e}"} for c in clfs]

    steps = {s["operation"]: s for s in log}
    bal, lnr = steps.get("balance"), steps.get("label_noise")
    base["smote_skipped"] = None if bal is None else bool(bal.get("skipped", False))
    base["smote_skip_reason"] = None if bal is None else bal.get("reason", "")
    base["n_relabelled"] = None if lnr is None else lnr.get("n_relabelled", 0)
    for s in log:                                   # keep whatever else each step reports
        for k, v in s.items():
            if k != "operation" and isinstance(v, (bool, int, float, str, np.number)):
                base[f"{s['operation']}__{k}"] = v

    rows = []
    for c in clfs:
        res = train_and_evaluate(X_cl, y_cl, X_te, y_te, c, random_state=seed)
        rows.append({**base, **res, "classifier": c})
    return rows


def summarise(name, df, seconds):
    parts = [f"[{name}] finished in {seconds / 60:.1f} min",
             f"rows={len(df)}", f"failed={int(df['failed'].eq(True).sum())}"]
    lr = df[df["classifier"] == "logistic_regression"]
    if len(lr) and "converged" in lr:
        parts.append(f"LR_not_converged={int(lr['converged'].eq(False).sum())}/{len(lr)}")
    if "smote_skipped" in df:
        has_bal = df["smote_skipped"].notna().sum()
        parts.append(f"SMOTE_skipped={int(df['smote_skipped'].eq(True).sum())}/{int(has_bal)}")
    print("  " + ", ".join(parts), flush=True)
    if df["failed"].eq(True).mean() > 0.02:
        print(f"  WARNING [{name}]: more than 2% of runs failed; look at the 'error' column "
              f"before using this dataset.", flush=True)


def run_dataset(name, stage, smoke, results_dir):
    cfg = STAGES[stage]
    faults = cfg["faults"]
    seeds = cfg.get("seeds", SEEDS)
    seeds = seeds[:1] if smoke else seeds
    seqs = sequences(cfg["seqs"])
    if smoke:
        seqs = seqs[::max(1, len(seqs) // 4)]
    out_dir = results_dir / (f"smoke_{stage}" if smoke else stage)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{name}.csv"
    if final.exists():
        written_by = set(pd.read_csv(final, usecols=["code_id"])["code_id"])
        if written_by != {code_id()}:
            sys.exit(f"{final} was produced by code {written_by}, current code is "
                     f"{code_id()}. Move it away to rerun this dataset.")
        print(f"[{name}] already complete ({final.relative_to(ROOT)}), skipping")
        return

    cid = code_id()
    data = load_dataset(name)
    X_all, y_all = data["X"], data["y"]
    cat_cols = list(data.get("cat_cols", []))
    print(f"[{name}] categorical columns: {cat_cols}", flush=True)
    classes, counts = np.unique(y_all, return_counts=True)
    if len(classes) != 2:
        sys.exit(f"[{name}] expected a binary target, found {len(classes)} classes")
    print(f"\n{'=' * 64}\n[{name}] stage={stage}  n={len(y_all)}  d={X_all.shape[1]}  "
          f"IR={counts.max() / counts.min():.2f}  sequences={len(seqs)}  seeds={len(seeds)}  "
          f"code={cid}\n{'=' * 64}", flush=True)

    expected_rows = sum(len(classifiers_for(s, CLASSIFIERS)) for s in seqs)
    t0, frames, audits = time.time(), [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=seed, stratify=y_all)
        injected = FaultInjector(random_state=seed, cat_cols=cat_cols).inject_all_faults(X_tr, y_tr, faults)
        X_c, y_c = injected["X_corrupted"], injected["y_corrupted"]
        a = audit(name, seed, y_tr, X_c, y_c)
        audits.append(a)
        if faults and a["minority_labels_flipped"] > 0.5:
            print(f"  WARNING [{name}] seed {seed}: {a['minority_labels_flipped']:.0%} of the "
                  f"minority labels were flipped; the fault model is saturated here.", flush=True)

        part = parts_dir / f"{name}_seed{seed}.csv"
        if part.exists():
            df = pd.read_csv(part)
            if set(df["code_id"]) != {cid}:
                sys.exit(f"{part} was written by code {set(df['code_id'])}, current code is {cid}. "
                         f"Delete {parts_dir} (or restore that commit) before resuming.")
            if len(df) != expected_rows:
                sys.exit(f"{part} has {len(df)} rows, expected {expected_rows}. "
                         f"Delete that file and resume.")
            frames.append(df)
            print(f"  seed {seed}: resumed from checkpoint", flush=True)
            continue

        t_seed, rows = time.time(), []
        for i, seq in enumerate(seqs, 1):
            rows += evaluate_sequence(seq, seed, X_c, y_c, X_te, y_te, CLASSIFIERS, cfg["cleaner"], cat_cols)
            if i % 6 == 0 or i == len(seqs):
                print(f"  [{name}] seed {seed}: {i}/{len(seqs)} sequences, "
                      f"{time.time() - t_seed:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.insert(0, "dataset", name)
        df["stage"], df["code_id"] = stage, cid
        write_csv_atomic(df, part)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    if len(result) != expected_rows * len(seeds):
        sys.exit(f"[{name}] {len(result)} rows, expected {expected_rows * len(seeds)}; not saving.")
    write_csv_atomic(pd.DataFrame(audits), out_dir / f"fault_audit_{name}.csv")
    write_csv_atomic(result, final)
    summarise(name, result, time.time() - t0)


def write_run_info(stage, smoke, datasets, results_dir):
    out_dir = results_dir / (f"smoke_{stage}" if smoke else stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    libs = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "imblearn"):
        try:
            libs[mod] = __import__(mod).__version__
        except Exception:
            libs[mod] = "not installed"
    with open(out_dir / "run_info.txt", "a") as f:
        f.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M} | stage={stage} smoke={smoke}\n")
        f.write(f"code: {code_id()} (git {git_commit()})\nseeds: {STAGES[stage].get('seeds', SEEDS)}\n"
                f"faults: {STAGES[stage]['faults']}\ncleaner: {STAGES[stage]['cleaner']}\n")
        f.write(f"datasets: {datasets}\nlibraries: {libs}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: the dataset list of the chosen stage")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, a few sequences, separate output folder (wiring test)")
    ap.add_argument("--results-dir", default="results",
                    help="folder for outputs, relative to this script (default: results)")
    args = ap.parse_args()
    results_dir = ROOT / args.results_dir

    datasets = args.datasets or STAGES[args.stage]["datasets"]
    unknown = [d for d in datasets if d not in LOADERS]
    if unknown:
        sys.exit(f"Unknown dataset(s) {unknown}. Available: {sorted(LOADERS)}")

    write_run_info(args.stage, args.smoke, datasets, results_dir)
    for name in datasets:
        run_dataset(name, args.stage, args.smoke, results_dir)
    print("\nAll requested datasets finished.")


if __name__ == "__main__":
    main()
