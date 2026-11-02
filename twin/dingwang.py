"""Saturating cellulose-ageing model with time-temperature-moisture superposition.

Source: H. Luo, L. Cheng, L. Yang, X. Zhao, R. Liao, Y. Zhang, "A novel approach to building digital
twin transformers by combining virtual-real sensing: An example of degree of polymerization
distribution", Measurement 222 (2023) 113714, Eqs. (5), (6), (10)-(12), built on Ding and Wang,
Cellulose 15 (2008) 205-224.

    w_DP = 1 - DP_t/DP_0 = w*(C) * (1 - exp(-k_DP(C) * t_eff))
    t_eff = sum( alpha_T * alpha_C * dt )
    alpha_T = exp( Ea/R * (1/Tref - 1/T) )          Ea = 91.4 kJ/mol, Tref = 363.15 K
    alpha_C = exp( -gamma * C / (R T) )             gamma = -1096.1125, C in percent
    w*(C)   = 26.38 C^2 - 0.776 C + 0.7912          C as a FRACTION (0.18 % -> 0.0018)
    k_DP(C) = -1.55 C^2 - 0.2015 C + 0.0013         C as a fraction

Things the paper does not settle, and what this module does about them:
 - The time unit of k_DP is not stated. The paper's Table 2 is closest to DAYS, but a first-principles
   run with days undershoots the table's rate by about a quarter (see run_luo_check.py), so the time
   unit is a parameter and the rate can be calibrated to a measurement.
 - Eq. (11) as printed gives k_DP <= 0 above about 0.61 % moisture, although the paper reports moisture
   up to 3.2 %. This module refuses to evaluate k_DP where it is not positive instead of clipping.
 - The levelling-off DP DP0 * (1 - w*) = 252.15 in the paper is reproduced (252.13) only with C as a fraction.
"""
from dataclasses import dataclass
import numpy as np

R_GAS = 8.314

@dataclass(frozen=True)
class DingWangParams:
    dp0: float = 1200.0
    Ea: float = 91.4e3          # J/mol
    T_ref: float = 363.15       # K
    gamma: float = -1096.1125   # fitted in the paper, C in percent

def w_star(c_pct):
    c = c_pct / 100.0
    return 26.38 * c**2 - 0.776 * c + 0.7912

def k_dp(c_pct):
    c = c_pct / 100.0
    k = -1.55 * c**2 - 0.2015 * c + 0.0013
    if k <= 0:
        raise ValueError(f"k_DP from the paper's Eq. (11) is not positive at {c_pct}% moisture; "
                         "the printed coefficients cannot be used at this moisture level")
    return k

def moisture_limit_pct():
    """Moisture (percent) at which the printed k_DP reaches zero."""
    a, b, c0 = -1.55, -0.2015, 0.0013
    return 100.0 * (-b - np.sqrt(b * b - 4 * a * c0)) / (2 * a)

def alpha_T(theta_c, p=DingWangParams()):
    return np.exp(p.Ea / R_GAS * (1.0 / p.T_ref - 1.0 / (np.asarray(theta_c, float) + 273.15)))

def alpha_C(c_pct, theta_c, p=DingWangParams()):
    return np.exp(-p.gamma * c_pct / (R_GAS * (np.asarray(theta_c, float) + 273.15)))

class DingWangAgeing:
    def __init__(self, c_pct=0.18, params=DingWangParams(), use_moisture_factor=False):
        self.c_pct, self.p, self.use_c = c_pct, params, use_moisture_factor
        self.k_scale = 1.0      # calibration multiplier on k_DP
        self.t_eff = 0.0        # accumulated equivalent time at the reference condition (time units of dt)

    @property
    def dp_levelling_off(self):
        return self.p.dp0 * (1.0 - w_star(self.c_pct))

    def advance(self, theta_c, dt):
        f = alpha_T(theta_c, self.p)
        if self.use_c:
            f = f * alpha_C(self.c_pct, theta_c, self.p)
        self.t_eff += float(np.sum(f * dt))

    def dp(self):
        ws, k = w_star(self.c_pct), k_dp(self.c_pct) * self.k_scale
        return self.p.dp0 * (1.0 - ws * (1.0 - np.exp(-k * self.t_eff)))

    def calibrate_to_measurement(self, dp_measured):
        """Scale k_DP so the model reproduces a measured DP at the current accumulated exposure."""
        if self.t_eff <= 0:
            raise ValueError("no exposure accumulated yet")
        if not self.dp_levelling_off < dp_measured <= self.p.dp0:
            raise ValueError(f"measured DP must lie between the levelling-off value "
                             f"({self.dp_levelling_off:.1f}) and DP0 ({self.p.dp0:.0f})")
        w = 1.0 - dp_measured / self.p.dp0
        self.k_scale = -np.log(1.0 - w / w_star(self.c_pct)) / (k_dp(self.c_pct) * self.t_eff)
        return self.k_scale
