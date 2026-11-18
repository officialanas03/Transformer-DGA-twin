"""Sensitivity of the DP result to the ASSUMED thermal model.

Luo et al. do not describe their temperature model, so any thermal model here is a placeholder. This module builds
variants that differ in the assumed top-oil temperature and heat split between the windings, fits the film coefficient
of each to the paper's printed hot spot (61.29 C at load 0.74; a FIT), and then reports the unfitted checks:
the printed overload hot spot (112.84 C at load 1.3) and where the paper places the hot spot (LV winding).
"""
import numpy as np
import pandas as pd
from twin import thermal_fv as tf

HOT_SPOT_TARGET = 61.29      # C, printed (Luo et al., Sec. 4.1)
OVERLOAD_TARGET = 112.84     # C, printed (Sec. 4.3)

def fit_film_coefficient(t_top, q_hv_factor=1.0, target=HOT_SPOT_TARGET, nz=60, tol=1e-4):
    """Film coefficient [W/m2K] for which the LV maximum at load 0.74 equals `target` (bisection on a log scale)."""
    f = lambda hf: tf.solve_model(0.74, hf, t_top=t_top, q_hv_factor=q_hv_factor, nz=nz)["LV"]["T_max"] - target
    lo, hi = 10.0, 3000.0
    if f(lo) < 0 or f(hi) > 0:
        raise ValueError("target not reachable with these settings")
    while hi / lo > 1 + tol:
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return float(np.sqrt(lo * hi))

def field_frame(model):
    """Finite-volume model result as a snapshot DataFrame (node, x_m, y_m, T_C) like the ANSYS exports."""
    rows, n = [], 0
    for w in ("LV", "HV"):
        m = model[w]
        for j, y in enumerate(m["y"]):
            for i, r in enumerate(m["r"]):
                n += 1
                rows.append((n, r, y, m["T"][j, i]))
    return pd.DataFrame(rows, columns=["node", "x_m", "y_m", "T_C"])

def variant(t_top, q_hv_factor=1.0, nz=60):
    """Fit a variant and return its snapshots at load 0.74 and 1.3 plus the unfitted checks."""
    hf = fit_film_coefficient(t_top, q_hv_factor, nz=nz)
    base = tf.solve_model(0.74, hf, t_top=t_top, q_hv_factor=q_hv_factor, nz=nz)
    over = tf.solve_model(1.3, hf, t_top=t_top, q_hv_factor=q_hv_factor, nz=nz)
    return dict(t_top=t_top, q_hv_factor=q_hv_factor, hf=hf, base=field_frame(base), overload=field_frame(over),
                lv_max=base["LV"]["T_max"], hv_max=base["HV"]["T_max"], lv_min=base["LV"]["T_min"],
                lv_overload=over["LV"]["T_max"], hv_overload=over["HV"]["T_max"],
                lv_y_rel=base["LV"]["y_at_max"] / 1.2)
