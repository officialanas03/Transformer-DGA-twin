"""IEC 60599 three-ratio method (basic gas ratios).

PROVENANCE: limits follow the IEC 60599 fault table as reproduced in
 - Karimi et al., Table 4 (dspace.upt.ro BUPT_ART_Karimi_f.pdf), and
 - TNB Research seminar slides, "IEC 60599: Interpretation of DGA".
Where they differ (T2's C2H4/C2H6 range) the TNB table is used: 1-4 for T2.
Other editions of the standard differ slightly (for example D2 up to 3.0 in some
tables), so the limits are in THRESHOLDS and can be changed. Check them against
the edition you cite.

Ratios: R2 = C2H2/C2H4, R1 = CH4/H2, R5 = C2H4/C2H6.
Each fault is a set of conditions; a sample gets a label only when exactly one
fault matches. No match -> "undetermined". Several matches -> "ambiguous".
Ranges written "a-b" are inclusive; ">" and "<" are strict.
The standard also applies only when gas levels exceed typical values; that
screening is not done here.
"""
import numpy as np
import pandas as pd

# (r2 = C2H2/C2H4, r1 = CH4/H2, r5 = C2H4/C2H6); None = not significant (any value)
THRESHOLDS = {
    "PD": (None,                 lambda x: x < 0.1,        lambda x: x < 0.2),
    "D1": (lambda x: x > 1,      lambda x: 0.1 <= x <= 0.5, lambda x: x > 1),
    "D2": (lambda x: 0.6 <= x <= 2.5, lambda x: 0.1 <= x <= 1, lambda x: x > 2),
    "T1": (None,                 lambda x: x > 1,          lambda x: x < 1),
    "T2": (lambda x: x < 0.1,    lambda x: x > 1,          lambda x: 1 <= x <= 4),
    "T3": (lambda x: x < 0.2,    lambda x: x > 1,          lambda x: x > 4),
}

def _ratio(num, den):
    if den == 0:
        return np.inf if num > 0 else np.nan     # 0/0 cannot be evaluated
    return num / den

def ratios(h2, ch4, c2h6, c2h4, c2h2):
    return _ratio(c2h2, c2h4), _ratio(ch4, h2), _ratio(c2h4, c2h6)

def matches(r2, r1, r5):
    out = []
    for fault, conds in THRESHOLDS.items():
        ok = True
        for cond, val in zip(conds, (r2, r1, r5)):
            if cond is None:
                continue
            if np.isnan(val) or not cond(val):
                ok = False
                break
        if ok:
            out.append(fault)
    return out

def classify(h2, ch4, c2h6, c2h4, c2h2):
    m = matches(*ratios(h2, ch4, c2h6, c2h4, c2h2))
    if len(m) == 1:
        return m[0]
    return "undetermined" if not m else "ambiguous:" + "/".join(m)

def classify_df(df):
    return pd.Series([classify(r.H2, r.CH4, r.C2H6, r.C2H4, r.C2H2) for r in df.itertuples()],
                     index=df.index, dtype=object)
