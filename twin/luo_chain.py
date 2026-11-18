"""Hot-spot-to-DP chain after Luo et al. (2023), with explicit, labelled variants.

Master curve (Eq. 5, 11) at a reference moisture C_ref, shifted in time by alpha_T (Eq. 6) and,
optionally, alpha_C (Eq. 10):

    t_eff = sum_years  hours_at_T * alpha_T(T) * alpha_C(C_year)          [days of reference exposure]
    DP    = DP0 * (1 - w*(C_ref) * (1 - exp(-k_DP(C_ref) * k_scale * t_eff)))

What is NOT from the paper:
 * k_scale: a multiplier on k_DP, FITTED to Table 2 row I year 1 (the time unit of k_DP is not stated).
 * The moisture at the hot spot is not given. Variants below ASSUME it:
     "none"      alpha_C = 1 (temperature only, row I)
     "eq13"      C(year r) = Eq. 13 total moisture of year r (assumption: hot spot sees total paper moisture)
     "const0p70" C = 0.70 % every year (Eq. 13 at year 3, the paper's stated starting distribution)
 * Master-curve moisture C_ref is fixed at 0.18 % (the paper's levelling-off condition) because Eq. 11's
   k_DP cannot be evaluated at the moistures the paper reports. This is a VARIANT, not the paper's method.
"""
import numpy as np
from twin.dingwang import DingWangParams, w_star, k_dp, alpha_T, alpha_C
from twin.moisture import total_moisture_pct

DAYS_PER_YEAR = 365.0
P = DingWangParams()

def moisture_for_year(variant, r):
    if variant == "none":
        return None
    if variant == "eq13":
        return float(total_moisture_pct(r))
    if variant == "const0p70":
        return 0.70
    raise ValueError(f"unknown variant {variant!r}")

def effective_days_per_year(theta_hot, variant, r, overload_hours=0.0, theta_overload=None):
    """Reference-condition days accumulated in operating year r."""
    c = moisture_for_year(variant, r)
    ac = 1.0 if c is None else float(alpha_C(c, theta_hot, P))
    days = (DAYS_PER_YEAR - overload_hours / 24.0) * float(alpha_T(theta_hot, P)) * ac
    if overload_hours:
        c_o = 1.0 if c is None else float(alpha_C(c, theta_overload, P))
        days += overload_hours / 24.0 * float(alpha_T(theta_overload, P)) * c_o
    return days

def dp_from_exposure(t_eff, k_scale=1.0, c_ref=0.18):
    ws, k = w_star(c_ref), k_dp(c_ref) * k_scale
    return P.dp0 * (1.0 - ws * (1.0 - np.exp(-k * t_eff)))

def fit_k_scale(dp_year1, theta_hot=61.29, c_ref=0.18):
    """k_scale so that temperature-only exposure for 1 year gives dp_year1. A FIT, not a prediction."""
    ws = w_star(c_ref)
    t1 = effective_days_per_year(theta_hot, "none", 1)
    w = 1.0 - dp_year1 / P.dp0
    return -np.log(1.0 - w / ws) / (k_dp(c_ref) * t1)

def table2_row(years, variant, k_scale, theta_hot=61.29, overload_hours=0.0, theta_overload=112.84, c_ref=0.18):
    """DP at the hot spot after each year in `years` (sorted ints)."""
    out, t_eff = {}, 0.0
    for r in range(1, max(years) + 1):
        t_eff += effective_days_per_year(theta_hot, variant, r, overload_hours, theta_overload)
        if r in years:
            out[r] = float(dp_from_exposure(t_eff, k_scale, c_ref))
    return out

def years_to_reach(dp_target, variant, k_scale, theta_hot=61.29, c_ref=0.18, max_years=2000):
    """First operating year at which DP <= dp_target, or None if the master curve never gets there
    (target at or below the levelling-off DP)."""
    if dp_target <= P.dp0 * (1.0 - w_star(c_ref)):
        return None
    t_eff = 0.0
    for r in range(1, max_years + 1):
        t_eff += effective_days_per_year(theta_hot, variant, r)
        if dp_from_exposure(t_eff, k_scale, c_ref) <= dp_target:
            return r
    return None

def relative_rate_moisture_only(c_hi_pct, c_lo_pct, theta_hot=61.29, include_k_dp=False):
    """Ratio of the moisture-driven ageing rate at c_hi vs c_lo at a fixed temperature:
    alpha_C(c_hi)/alpha_C(c_lo), optionally times Eq. 11's k_DP(c_hi)/k_DP(c_lo)."""
    r = float(alpha_C(c_hi_pct, theta_hot, P) / alpha_C(c_lo_pct, theta_hot, P))
    if include_k_dp:
        r *= k_dp(c_hi_pct) / k_dp(c_lo_pct)
    return r
