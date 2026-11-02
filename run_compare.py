"""Duval vs IEC ratios vs random forest on the same cases."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dga.data.loader import load_dga
from dga.data.labels import to_coarse
from dga.ml.features import make_features
from dga.rules import duval, iec

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
SEEDS = range(5)

aug = load_dga("data/dga_augmented.csv")
aug["cls"] = np.where(aug.fault == "Normal", "Normal", to_coarse(aug.fault))
train = aug[aug.cls != "Normal"]                       # Normal rows excluded (label is suspect)
Xtr, ytr = make_features(train), train.cls

def fit_predict(df, seed):
    m = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)
    return m.fit(Xtr, ytr).predict(make_features(df))

# ---- 25 real cases, using every row that has a stated class (single or mixed) ----
raw = pd.read_csv("data/dga_iec_25cases.csv")
raw = raw[raw.fault.notna() & (raw.fault.astype(str).str.strip() != "")].copy()
raw["case_no"] = raw.case_no.astype(int)

def accepted(label):
    parts = [p.strip() for p in str(label).split("+")]
    acc = {to_coarse(pd.Series([p])).iloc[0] for p in parts}
    return acc | {"DT"} if len(parts) > 1 else acc    # for a mixed fault, DT also counts as right

raw["accept"] = raw.fault.map(accepted)
raw["duval"] = to_coarse(duval.classify_df(raw))
iec_raw = iec.classify_df(raw)
raw["iec"] = iec_raw.map(lambda s: to_coarse(pd.Series([s])).iloc[0] if s in ("PD","D1","D2","T1","T2","T3") else s)
preds = np.array([fit_predict(raw, s) for s in SEEDS])
raw["rf"] = preds[0]

def hit(col):  return np.array([p in a for p, a in zip(raw[col], raw.accept)])
raw["duval_ok"], raw["iec_ok"] = hit("duval"), hit("iec")
rf_hits = np.array([[p in a for p, a in zip(row, raw.accept)] for row in preds])   # seeds x cases
raw["rf_ok"] = rf_hits[0]

show = raw.assign(truth=raw.fault)[["case_no", "truth", "duval", "iec", "rf"]]
print("PER CASE (rf = seed 0)"); print(show.to_string(index=False))

nonpd = ~raw.fault.eq("PD").values
def row(name, ok, rf=False):
    a = rf_hits.mean(axis=1) if rf else None
    all_ = (a.mean() if rf else ok.mean()); np_ = (rf_hits[:, nonpd].mean() if rf else ok[nonpd].mean())
    return name, f"{all_:.2f}", f"{np_:.2f}"
print(f"\nACCURACY on real cases  (all stated classes n={len(raw)}, excluding PD n={int(nonpd.sum())})")
print(pd.DataFrame([row("Duval triangle 1", raw.duval_ok.values), row("IEC ratios", raw.iec_ok.values),
                    row("Random forest (5 seeds)", None, rf=True)], columns=["method", "all", "non-PD"]).to_string(index=False))
print("IEC outcomes on these cases:", raw.iec.map(lambda s: s if s in ("undetermined",) or str(s).startswith("ambiguous") else "assigned").value_counts().to_dict())
print("RF seed-to-seed accuracy range (all):", f"{rf_hits.mean(axis=1).min():.2f}-{rf_hits.mean(axis=1).max():.2f}")
raw.drop(columns=["accept"]).to_csv("comparison_25cases.csv", index=False)

# ---- augmented set (non-Normal rows): rules only, plus out-of-fold RF ----
nn = train.copy()
nn["duval"] = to_coarse(duval.classify_df(nn))
ic = iec.classify_df(nn)
nn["iec"] = ic.map(lambda s: to_coarse(pd.Series([s])).iloc[0] if s in ("PD","D1","D2","T1","T2","T3") else s)
assigned = ~nn.iec.astype(str).isin(["undetermined"]) & ~nn.iec.astype(str).str.startswith("ambiguous")
print(f"\nAUGMENTED SET, non-Normal rows (n={len(nn)}) - rules only; labels are not Duval- or IEC-derived")
print(f"Duval accuracy: {(nn.duval == nn.cls).mean():.3f}")
print(f"IEC coverage (got a single label): {assigned.mean():.3f} | accuracy on assigned: {(nn.iec[assigned] == nn.cls[assigned]).mean():.3f} | accuracy over all rows: {((nn.iec == nn.cls)).mean():.3f}")
print("IEC non-assigned breakdown:", nn.iec[~assigned].map(lambda s: "undetermined" if s == "undetermined" else "ambiguous").value_counts().to_dict())
