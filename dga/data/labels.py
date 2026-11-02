"""Label vocabulary shared by all datasets.

Different sources name fault classes differently; everything is mapped to a
canonical set here. Anything not recognised becomes None.
"""
import pandas as pd

CANON = {
    "normal": "Normal", "no fault": "Normal", "none": "Normal",
    "pd": "PD", "partial discharge": "PD", "corona discharge": "PD",
    "d1": "D1", "low intensity discharge": "D1", "low energy discharge": "D1",
    "d2": "D2", "high intensity discharge": "D2", "high energy discharge": "D2",
    "t1": "T1", "t2": "T2", "t3": "T3",
    "t1&t2": "T1&T2", "t1/t2": "T1&T2", "t1+t2": "T1&T2",
    "t": "T", "thermal fault": "T",
    "dt": "DT",
}

# Label space on which rule-based (Duval) output and ML output can be compared.
# Duval's triangle has no "Normal" zone; T1/T2 are merged because one source merges them.
COARSE = {"PD": "PD", "D1": "D1", "D2": "D2", "T1": "T1&T2", "T2": "T1&T2",
          "T1&T2": "T1&T2", "T3": "T3", "DT": "DT"}

def standardize(series: pd.Series) -> pd.Series:
    """Map raw labels to canonical ones. Unknown or mixed labels (e.g. 'T3+D2') become None."""
    out = series.astype("string").str.strip().str.lower().map(CANON).astype(object)
    return out.where(out.notna(), None)

def to_coarse(series: pd.Series) -> pd.Series:
    return series.map(COARSE)
