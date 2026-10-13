"""
Load every registered dataset once and print what the paper's Table 1 needs.
Run this before a long experiment: it also downloads and caches the OpenML
datasets, so a network problem shows up now and not hours into the run.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loaders import LOADERS  # noqa: E402

print(f"{'dataset':<16}{'rows':>8}{'features':>10}{'IR':>8}{'minority %':>12}{'NaN %':>8}")
for name, loader in LOADERS.items():
    try:
        d = loader()
        X, y = d["X"], d["y"]
        classes, c = np.unique(y, return_counts=True)
        flag = "" if len(classes) == 2 else f"   <-- {len(classes)} classes, binary needed"
        print(f"{name:<16}{len(y):>8}{X.shape[1]:>10}{c.max() / c.min():>8.2f}"
              f"{100 * c.min() / c.sum():>12.1f}{100 * np.isnan(X).mean():>8.2f}{flag}")
    except Exception as e:
        print(f"{name:<16} FAILED: {type(e).__name__}: {e}")
