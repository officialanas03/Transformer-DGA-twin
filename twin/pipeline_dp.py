"""Hot-spot temperature (from an ANSYS export or a surrogate) to DP along a winding.

Follows Luo et al. only as far as the paper allows: temperature enters through alpha_T, the rate constant is
FITTED to Table 2 row I (see luo_chain.fit_k_scale), moisture is a labelled VARIANT. With the ASSUMED placeholder
thermal model the result is a demonstration of the pipeline, not a finding about any transformer.
"""
import numpy as np
from twin.dingwang import alpha_T, alpha_C
from twin.luo_chain import DAYS_PER_YEAR, dp_from_exposure, fit_k_scale, moisture_for_year

# Table 2 row I year 1 (printed in Luo et al.); used only to fix the rate constant
TABLE2_I_YEAR1 = 1167.52

def dp_profile(theta_c, years, variant="none", k_scale=None, c_ref=0.18):
    """DP after `years` of constant operation at temperature theta_c (scalar or array, C)."""
    if k_scale is None:
        k_scale = fit_k_scale(TABLE2_I_YEAR1)
    theta = np.asarray(theta_c, float)
    t_eff = np.zeros_like(theta)
    for r in range(1, int(years) + 1):
        c = moisture_for_year(variant, r)
        f = alpha_T(theta)
        if c is not None:
            f = f * alpha_C(c, theta)
        t_eff = t_eff + DAYS_PER_YEAR * f
    return dp_from_exposure(t_eff, k_scale, c_ref)

def dp_map_from_snapshot(df, years, variant="none", k_scale=None):
    """DP at every node of an ANSYS snapshot (needs a T_C column)."""
    out = df.copy()
    out["DP"] = dp_profile(df.T_C.to_numpy(), years, variant, k_scale)
    return out

def hot_spot_dp(df, years, variant="none", k_scale=None):
    """DP at the hottest node of each winding, with a note saying where that node is."""
    from twin.ansys_io import winding_stats
    s = winding_stats(df)
    return {w: dict(T=v["T_max"], y_rel=v["y_rel_at_max"], DP=float(dp_profile(v["T_max"], years, variant, k_scale)))
            for w, v in s.items()}


# ---------------------------------------------------------------------------------------------------------------
# Moisture distribution and DP per node (temperature field -> moisture -> DP)
#
# INTERPRETATION (not stated in this form by the paper): the paper says the moisture distribution at different years
# is obtained from the oil-paper equilibrium and the empirical total moisture C_r(r) (Eq. 13). Here the equilibrium
# moisture shape C_eq(T_node) is scaled so that its weighted mean equals C_r(r). Scaling removes the dependence on the
# absolute prefactor of Eq. 1 (which is unresolved); only the temperature dependence of Eq. 1 matters.
# The weights are proportional to radius (axisymmetric volume), an ASSUMPTION.
# ---------------------------------------------------------------------------------------------------------------
from twin import moisture as _mo

def moisture_field_pct(df, year, c_oil_ppm=10.0):
    """Moisture [percent] at every node of a snapshot in operating year `year`."""
    ceq = np.asarray(_mo.equilibrium_paper_moisture_pct(df.T_C.to_numpy(), c_oil_ppm), float)
    w = df.x_m.to_numpy()
    mean = float(np.sum(ceq * w) / np.sum(w))
    return ceq * float(_mo.total_moisture_pct(year)) / mean

def dp_map_with_moisture(df, years, df_overload=None, overload_hours=0.0, k_scale=None, c_ref=0.18, c_oil_ppm=10.0):
    """DP after `years` at every node. Moisture enters only through alpha_C (VARIANT: master curve frozen at c_ref).
    df_overload: snapshot at the overload load (same mesh); `overload_hours` per year are spent at that field."""
    if k_scale is None:
        k_scale = fit_k_scale(TABLE2_I_YEAR1)
    if df_overload is not None:
        if len(df_overload) != len(df) or not np.array_equal(df_overload.node.to_numpy(), df.node.to_numpy()):
            raise ValueError("overload snapshot must have the same nodes as the base snapshot")
    elif overload_hours:
        raise ValueError("overload_hours needs an overload snapshot")
    T = df.T_C.to_numpy()
    t_eff = np.zeros(len(df))
    for r in range(1, int(years) + 1):
        c = moisture_field_pct(df, r, c_oil_ppm)
        days = (DAYS_PER_YEAR - overload_hours / 24.0) * alpha_T(T) * alpha_C(c, T)
        if df_overload is not None and overload_hours:
            To = df_overload.T_C.to_numpy()
            days = days + overload_hours / 24.0 * alpha_T(To) * alpha_C(c, To)
        t_eff = t_eff + days
    out = df.copy()
    out["C_pct"] = moisture_field_pct(df, years, c_oil_ppm)
    out["DP"] = dp_from_exposure(t_eff, k_scale, c_ref)
    return out

def position_summary(map_df, geom=None):
    """DP at the top (90 % height), middle (50 %) and bottom (10 %) of each winding: lowest value across the thickness
    at the nearest mesh height. Same layout as Table 3 of the paper."""
    from twin.ansys_io import winding, Geometry
    g = geom or Geometry()
    d = winding(map_df, g)
    res = {}
    for w in ("LV", "HV"):
        s = d[d.winding == w]
        res[w] = {}
        for name, frac in (("top", 0.9), ("middle", 0.5), ("bottom", 0.1)):
            y = s.y_m.iloc[(s.y_m - frac * g.height).abs().argmin()]
            res[w][name] = float(s[np.isclose(s.y_m, y)].DP.min())
    return res

def hot_spot_summary(map_df, geom=None):
    """Hottest node of each winding in a map with columns T_C and DP: temperature, relative height and DP there."""
    from twin.ansys_io import winding, Geometry
    g = geom or Geometry()
    d = winding(map_df, g)
    out = {}
    for w, s in d.groupby("winding"):
        i = s.T_C.idxmax()
        out[w] = dict(T=float(s.T_C[i]), y_rel=float(s.y_m[i] / g.height), DP=float(s.DP[i]))
    return out
