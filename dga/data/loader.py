import warnings
import pandas as pd
from .labels import standardize

CORE = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]   # present in every dataset
OPTIONAL = ["CO", "CO2"]                        # absent from some datasets

def load_dga(path, label_col="fault", group_col=None, drop_unlabelled=True):
    """Load a DGA CSV. Requires the five core gases; CO/CO2 are optional.

    Returns a DataFrame with a standardised `fault` column. Rows whose label is
    missing, mixed or unrecognised are dropped (with a warning) when
    drop_unlabelled=True.
    """
    df = pd.read_csv(path)
    missing = [c for c in CORE + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    gases = CORE + [c for c in OPTIONAL if c in df.columns]
    for c in gases:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if (df[gases] < 0).any().any():
        raise ValueError(f"{path}: negative gas concentration found")
    df = df.dropna(subset=CORE).copy()
    df["fault"] = standardize(df[label_col])
    n_bad = int(df["fault"].isna().sum())
    if drop_unlabelled and n_bad:
        warnings.warn(f"{path}: dropping {n_bad} rows with missing/mixed/unknown label")
        df = df.dropna(subset=["fault"])
    if group_col is not None and group_col not in df.columns:
        raise ValueError(f"{path}: group column '{group_col}' not found")
    return df.reset_index(drop=True)
