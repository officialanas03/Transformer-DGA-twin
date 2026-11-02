import numpy as np
import pandas as pd

EPS = 1e-6

def three_gas_pct(df):
    g = df[["CH4", "C2H4", "C2H2"]]
    return g.div(g.sum(axis=1).clip(lower=EPS), axis=0) * 100

def log_ratios(df, use_co=False):
    r = {
        "r_c2h2_c2h4": np.log10((df.C2H2 + EPS) / (df.C2H4 + EPS)),
        "r_ch4_h2":    np.log10((df.CH4 + EPS) / (df.H2 + EPS)),
        "r_c2h4_c2h6": np.log10((df.C2H4 + EPS) / (df.C2H6 + EPS)),
    }
    if use_co:
        if "CO" not in df.columns or "CO2" not in df.columns or df[["CO", "CO2"]].isna().any().any():
            raise ValueError("use_co=True needs complete CO and CO2 columns")
        r["r_co2_co"] = np.log10((df.CO2 + EPS) / (df.CO + EPS))
    return pd.DataFrame(r)

def make_features(df, use_co=False):
    """Five-gas features by default, so a model trained on one dataset can be
    tested on another that lacks CO/CO2."""
    return pd.concat([three_gas_pct(df).add_prefix("pct_"), log_ratios(df, use_co)], axis=1)
