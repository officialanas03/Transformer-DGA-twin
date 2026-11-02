"""Calibration + conformal prediction study.
(1) Inside the augmented set (random 60/20/20 split, 5 repeats).
(2) The same models applied to the real labelled cases."""
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dga.data.loader import load_dga
from dga.data.labels import to_coarse
from dga.ml.features import make_features
from dga.ml.uncertainty import (expected_calibration_error, conformal_threshold,
                                prediction_sets, coverage_and_size)

warnings.filterwarnings("ignore")
ALPHA = 0.1
rf = lambda s: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=s, n_jobs=-1)

aug = load_dga("data/dga_augmented.csv")
aug = aug[aug.fault != "Normal"].copy()
aug["cls"] = to_coarse(aug.fault)
X, y = make_features(aug), aug.cls

raw = pd.read_csv("data/dga_iec_25cases.csv")
raw = raw[raw.fault.notna() & (raw.fault.astype(str).str.strip() != "") & (raw.fault != "PD")].copy()
accept = [{to_coarse(pd.Series([p.strip()])).iloc[0] for p in str(f).split("+")} for f in raw.fault]
Xr = make_features(raw)

rows_in, rows_real, example = [], [], None
for seed in range(5):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    for name, model in [("raw RF", rf(seed).fit(Xtr, ytr)),
                        ("sigmoid-calibrated RF", CalibratedClassifierCV(rf(seed), method="sigmoid", cv=5).fit(Xtr, ytr))]:
        cls = list(model.classes_); idx = lambda s: np.array([cls.index(v) for v in s])
        pcal, pte = model.predict_proba(Xcal), model.predict_proba(Xte)
        q = conformal_threshold(pcal, idx(ycal), ALPHA)
        res = coverage_and_size(prediction_sets(pte, q), idx(yte))
        rows_in.append({"model": name, "ECE": expected_calibration_error(pte, idx(yte)),
                        "accuracy": float((pte.argmax(1) == idx(yte)).mean()), **res})
        # real cases: a set is correct if it contains any accepted class
        pr = model.predict_proba(Xr); sets = prediction_sets(pr, q)
        hit = [any(cls[j] in a for j in np.where(s)[0]) for s, a in zip(sets, accept)]
        rows_real.append({"model": name, "coverage": float(np.mean(hit)), "mean_size": float(sets.sum(1).mean()),
                          "empty_rate": float((sets.sum(1) == 0).mean()),
                          "top1_acc": float(np.mean([cls[p] in a for p, a in zip(pr.argmax(1), accept)])),
                          "mean_conf": float(pr.max(1).mean())})
        if seed == 0 and name.startswith("sigmoid"):
            example = pd.DataFrame({"case_no": raw.case_no.astype(int).values, "truth": raw.fault.values,
                                    "set": [",".join(cls[j] for j in np.where(s)[0]) or "(empty)" for s in sets],
                                    "top_prob": pr.max(1).round(2), "covered": hit})

pd.set_option("display.width", 200)
print(f"1) INSIDE THE AUGMENTED SET (n={len(aug)}, 60/20/20 split, mean of 5 splits, target coverage {1-ALPHA:.0%})")
print(pd.DataFrame(rows_in).groupby("model").mean().round(3))
print(f"\n2) SAME MODELS ON THE REAL CASES (n={len(raw)} labelled non-PD cases, mean of 5 splits)")
print(pd.DataFrame(rows_real).groupby("model").mean().round(3))
print("\nPer-case sets, sigmoid-calibrated RF, split 0:"); print(example.to_string(index=False))
