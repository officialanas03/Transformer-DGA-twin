import warnings
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def default_model(seed=0):
    return RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)

def run_cv(X, y, groups=None, model=None, n_splits=5, seed=0):
    """Cross-validation, grouped by transformer when `groups` is given. Without
    groups it is plain stratified CV, whose scores can be optimistic if
    near-copies of one case land on both sides of a split."""
    model = model or default_model(seed)
    if groups is not None:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        warnings.warn("No transformer groups: CV scores may be optimistic (possible leakage).")
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    oof = np.empty(len(y), dtype=object)
    for tr, te in splits:
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof, f1_score(y, oof, average="macro")

def external_eval(Xtr, ytr, Xte, yte, model=None, seed=0):
    """Train on one dataset, test on a different one (the leakage-free check)."""
    m = clone(model or default_model(seed)).fit(Xtr, ytr)
    pred = m.predict(Xte)
    labels = sorted(set(yte) | set(pred))
    return {
        "pred": pred,
        "accuracy": float((pred == yte.values).mean()),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=sorted(set(yte)))),
        "report": classification_report(yte, pred, zero_division=0),
        "labels": labels,
        "confusion": confusion_matrix(yte, pred, labels=labels),
    }
