"""Duval Triangle 1 (CH4, C2H4, C2H2), expressed as percentage rules.

PROVENANCE: the nine breakpoints (98, 20, 23, 40, 50, 4, 13, 29, 15) are the
numbers printed on Fig. 1 of M. Duval, "A Review of Faults Detectable by
Gas-in-Oil Analysis in Transformers", IEEE Electr. Insul. Mag. 18(3), 2002.
The way they combine into zones was reconstructed from that figure's layout
and checked against published worked examples (see tests/test_duval.py).
It has NOT been checked line-by-line against the IEC 60599 / IEEE C57.104
text. Compare it with the official figure before citing it.

Conventions: percentages are of CH4 + C2H4 + C2H2. Lower bounds are inclusive,
upper bounds exclusive; PD is CH4 >= 98. Zones are tested in the order below,
so T3 takes precedence over the DT strip it overlaps (C2H2 in [13, 15)).
"""
import numpy as np
import pandas as pd

ZONE_ORDER = ["PD", "T1", "T2", "T3", "D1", "D2", "DT"]

def percentages(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total <= 0:
        return None
    return 100 * ch4 / total, 100 * c2h4 / total, 100 * c2h2 / total

def classify_pct(p_ch4, p_c2h4, p_c2h2):
    if p_ch4 >= 98:
        return "PD"
    if p_c2h2 < 4 and p_c2h4 < 20:
        return "T1"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50:
        return "T2"
    if p_c2h2 < 15 and p_c2h4 >= 50:
        return "T3"
    if p_c2h2 >= 13 and p_c2h4 < 23:
        return "D1"
    if p_c2h2 >= 13 and p_c2h4 >= 23 and (p_c2h4 < 40 or p_c2h2 >= 29):
        return "D2"
    return "DT"

def classify(ch4, c2h4, c2h2):
    """Zone for gas concentrations in ppm. Returns None if all three are zero."""
    p = percentages(ch4, c2h4, c2h2)
    return None if p is None else classify_pct(*p)

def classify_df(df):
    """Vectorised wrapper; expects columns CH4, C2H4, C2H2."""
    return pd.Series([classify(a, b, c) for a, b, c in zip(df.CH4, df.C2H4, df.C2H2)],
                     index=df.index, dtype=object)
