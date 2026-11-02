"""Baseline: in-dataset CV vs. external test on real cases. Run from the repo root."""
import warnings
import pandas as pd
from dga.data.loader import load_dga
from dga.data.labels import to_coarse
from dga.ml.features import make_features
from dga.eval.cv import run_cv, external_eval

warnings.filterwarnings("ignore")
aug = load_dga("data/dga_augmented.csv")
ext = load_dga("data/dga_iec_25cases.csv")          # mixed labels dropped; PD kept
aug["cls"] = to_coarse(aug.fault).where(aug.fault != "Normal", "Normal")
ext["cls"] = to_coarse(ext.fault)

print("External rows by class:", ext.cls.value_counts().to_dict())
for include_normal in (False, True):
    tr = aug if include_normal else aug[aug.cls != "Normal"]
    te = ext[ext.cls.isin(tr.cls.unique())]          # only classes the model has seen
    Xtr, ytr = make_features(tr), tr.cls
    oof, f1 = run_cv(Xtr, ytr)
    acc_cv = (oof == ytr.values).mean()
    r = external_eval(Xtr, ytr, make_features(te), te.cls)
    print(f"\n=== train on augmented, Normal {'INCLUDED' if include_normal else 'EXCLUDED'} (n_train={len(tr)}) ===")
    print(f"in-dataset 5-fold CV (no groups): acc={acc_cv:.3f}  macro-F1={f1:.3f}")
    print(f"external test on 25-case set (n={len(te)}): acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}")
    print("labels:", r["labels"]); print(r["confusion"])
