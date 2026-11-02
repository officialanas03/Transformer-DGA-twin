"""One-call diagnosis: Duval, IEC ratios, optional random forest with a conformal prediction set,
and ageing indicators. Used by the Streamlit app.

The rule-based and ageing parts need no data. The forest is trained from the augmented dataset when
that file is present (it is not shipped in the public repository) and is simply unavailable otherwise.
It is trained WITHOUT the 'Normal' class (its labels disagree with Duval's triangle on most rows) and
has no PD class, so it cannot output PD. Its conformal sets are only guaranteed on data that resembles
the calibration data; on the 25 real cases coverage fell short of the target (see README).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .rules import duval, iec
from .ageing import paper

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "dga_augmented.csv"
CORE = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
MAX_BATCH_ROWS = 5000

def diagnose_rules(h2, ch4, c2h6, c2h4, c2h2, co=None, co2=None, fal_ppm=None):
    """Everything that needs no training data."""
    g = {"H2": h2, "CH4": ch4, "C2H6": c2h6, "C2H4": c2h4, "C2H2": c2h2}
    if any(v < 0 for v in g.values()):
        raise ValueError("gas concentrations cannot be negative")
    out = {"warnings": []}
    pct = duval.percentages(ch4, c2h4, c2h2)
    out["duval_pct"] = None if pct is None else dict(zip(["CH4", "C2H4", "C2H2"], pct))
    out["duval"] = None if pct is None else duval.classify_pct(*pct)
    out["iec"] = iec.classify(h2, ch4, c2h6, c2h4, c2h2)
    if pct is None:
        out["warnings"].append("CH4, C2H4 and C2H2 are all zero: no fault-gas pattern to classify.")
    if co is not None and co2 is not None:
        r = paper.co2_co_ratio(co2, co)
        out["co2_co"] = {"ratio": float(r), "band": paper.co2_co_band(r)}
    if fal_ppm is not None and fal_ppm > 0:
        out["furan"] = {"dp_chendong": float(paper.dp_chendong(fal_ppm)),
                        "dp_depablo": float(paper.dp_depablo(fal_ppm)),
                        "condition": paper.furan_condition(fal_ppm)}
    return out

class Diagnoser:
    """Adds the random forest and its conformal sets on top of diagnose_rules."""
    def __init__(self, data_path=DEFAULT_DATA, alpha=0.1, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from .data.loader import load_dga
        from .data.labels import to_coarse
        from .ml.features import make_features
        from .ml.uncertainty import conformal_threshold
        aug = load_dga(data_path)
        aug = aug[aug.fault != "Normal"].copy()
        aug["cls"] = to_coarse(aug.fault)
        X, y = make_features(aug), aug.cls
        Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        self.model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        self.classes = list(self.model.classes_)
        ycal_idx = np.array([self.classes.index(v) for v in ycal])
        self.q = conformal_threshold(self.model.predict_proba(Xcal), ycal_idx, alpha)
        self.alpha = alpha
        self.train_max = aug[CORE].max()
        self.n_train = len(Xtr)

    def predict_frame(self, df):
        """Class probabilities and conformal sets for every row of a DataFrame with the five core gases."""
        from .ml.features import make_features
        proba = self.model.predict_proba(make_features(df))
        sets = [[c for c, p in zip(self.classes, row) if 1 - p <= self.q] for row in proba]
        top = [self.classes[int(i)] for i in proba.argmax(1)]
        return proba, sets, top

    def diagnose(self, h2, ch4, c2h6, c2h4, c2h2, co=None, co2=None, fal_ppm=None):
        out = diagnose_rules(h2, ch4, c2h6, c2h4, c2h2, co, co2, fal_ppm)
        g = {"H2": h2, "CH4": ch4, "C2H6": c2h6, "C2H4": c2h4, "C2H2": c2h2}
        if ch4 + c2h4 + c2h2 > 0:
            proba, sets, top = self.predict_frame(pd.DataFrame([g]))
            out["rf_proba"] = dict(zip(self.classes, proba[0].round(3)))
            out["rf_set"], out["rf_top"] = sets[0], top[0]
        else:
            out["rf_proba"], out["rf_set"], out["rf_top"] = {}, [], None
        over = [k for k, v in g.items() if v > self.train_max[k]]
        if over:
            out["warnings"].append("Above the range seen in the training data for " + ", ".join(over)
                                   + "; the forest is extrapolating.")
        out["warnings"].append("The forest has no PD class and was trained on augmented data with limited real-case validation.")
        return out

def load_diagnoser_or_none(data_path=DEFAULT_DATA):
    """The forest if the training file exists, otherwise None (rules-only mode)."""
    return Diagnoser(data_path) if Path(data_path).exists() else None

def diagnose_table(df, diagnoser=None):
    """Batch diagnosis for a table with columns H2, CH4, C2H6, C2H4, C2H2 (optional CO, CO2, FAL).
    Returns a DataFrame with one row per input row; invalid rows are flagged, not dropped."""
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"at most {MAX_BATCH_ROWS} rows per upload (got {len(df)})")
    d = df.copy()
    for c in CORE + [c for c in ("CO", "CO2", "FAL") if c in d.columns]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = d[CORE].isna().any(axis=1) | (d[CORE] < 0).any(axis=1)
    rows = []
    for i, r in d.iterrows():
        if bad[i]:
            rows.append({"duval": None, "iec": None, "note": "invalid or missing gas value"})
            continue
        co = r["CO"] if "CO" in d.columns and pd.notna(r["CO"]) else None
        co2 = r["CO2"] if "CO2" in d.columns and pd.notna(r["CO2"]) else None
        fal = r["FAL"] if "FAL" in d.columns and pd.notna(r["FAL"]) else None
        o = diagnose_rules(r.H2, r.CH4, r.C2H6, r.C2H4, r.C2H2, co, co2, fal)
        row = {"duval": o["duval"], "iec": o["iec"], "note": ""}
        if "co2_co" in o:
            row["co2_co"], row["co2_co_band"] = round(o["co2_co"]["ratio"], 2), o["co2_co"]["band"]
        if "furan" in o:
            row["dp_chendong"], row["furan_condition"] = round(o["furan"]["dp_chendong"]), o["furan"]["condition"]
        rows.append(row)
    res = pd.DataFrame(rows, index=d.index)
    if diagnoser is not None:
        ok = (~bad) & ((d.CH4 + d.C2H4 + d.C2H2) > 0)
        res["rf_set"], res["rf_top"] = None, None
        if ok.any():
            _, sets, top = diagnoser.predict_frame(d.loc[ok, CORE])
            res.loc[ok, "rf_set"] = [",".join(s) if s else "(empty)" for s in sets]
            res.loc[ok, "rf_top"] = top
    return pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
