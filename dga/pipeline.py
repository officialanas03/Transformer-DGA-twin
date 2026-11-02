"""One-call diagnosis: Duval, IEC ratios, random forest with a conformal prediction set,
and ageing indicators. Used by the Streamlit app.

The forest is trained on the augmented dataset WITHOUT the 'Normal' class (its labels
disagree with Duval's triangle on most rows) and has no PD class, so it cannot output PD.
Its conformal sets are only guaranteed on data that resembles the calibration data;
on the 25 real cases coverage fell short of the target (see README).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .data.loader import load_dga
from .data.labels import to_coarse
from .ml.features import make_features
from .ml.uncertainty import conformal_threshold, prediction_sets
from .rules import duval, iec
from .ageing import paper

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "dga_augmented.csv"
CORE = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

class Diagnoser:
    def __init__(self, data_path=DEFAULT_DATA, alpha=0.1, seed=0):
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

    def diagnose(self, h2, ch4, c2h6, c2h4, c2h2, co=None, co2=None, fal_ppm=None):
        g = {"H2": h2, "CH4": ch4, "C2H6": c2h6, "C2H4": c2h4, "C2H2": c2h2}
        if any(v < 0 for v in g.values()):
            raise ValueError("gas concentrations cannot be negative")
        out = {"warnings": []}
        total_fault = ch4 + c2h4 + c2h2
        pct = duval.percentages(ch4, c2h4, c2h2)
        out["duval_pct"] = None if pct is None else dict(zip(["CH4", "C2H4", "C2H2"], pct))
        out["duval"] = None if pct is None else duval.classify_pct(*pct)
        out["iec"] = iec.classify(h2, ch4, c2h6, c2h4, c2h2)
        if total_fault > 0:
            df = pd.DataFrame([g])
            proba = self.model.predict_proba(make_features(df))[0]
            out["rf_proba"] = dict(zip(self.classes, proba.round(3)))
            out["rf_set"] = [c for c, p in zip(self.classes, proba) if 1 - p <= self.q]
            out["rf_top"] = self.classes[int(proba.argmax())]
        else:
            out["rf_proba"], out["rf_set"], out["rf_top"] = {}, [], None
            out["warnings"].append("CH4, C2H4 and C2H2 are all zero: no fault-gas pattern to classify.")
        over = [k for k, v in g.items() if v > self.train_max[k]]
        if over:
            out["warnings"].append("Above the range seen in the training data for " + ", ".join(over)
                                   + "; the forest is extrapolating.")
        out["warnings"].append("The forest has no PD class and was trained on augmented data with limited real-case validation.")
        if co is not None and co2 is not None:
            r = paper.co2_co_ratio(co2, co)
            out["co2_co"] = {"ratio": float(r), "band": paper.co2_co_band(r)}
        if fal_ppm is not None and fal_ppm > 0:
            out["furan"] = {"dp_chendong": float(paper.dp_chendong(fal_ppm)),
                            "dp_depablo": float(paper.dp_depablo(fal_ppm)),
                            "condition": paper.furan_condition(fal_ppm)}
        return out
