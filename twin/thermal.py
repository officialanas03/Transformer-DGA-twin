"""Top-oil and hot-spot temperature model (IEEE C57.91 / IEC 60076-7 differential form).

    tau_o  d(theta_o)/dt = [(1 + R K^2)/(1 + R)]^n * dtheta_or - (theta_o - theta_a)
    tau_w  d(theta_h)/dt = dtheta_hr * K^(2m)                 - (theta_h - theta_o)

The structure of both equations was checked against Rev. Roum. Sci. Tech. Electrotech.
Energ. (journal.iem.pub.ro, article 49) and the hot-spot decomposition
theta_h = theta_a + dtheta_TO + dtheta_H against IEC 60076-7 / IEEE C57.91 as quoted there.

IMPORTANT: the numbers in ThermalParams are ILLUSTRATIVE placeholders typical of an
ONAN unit. They were NOT verified against the standards' tables. Replace them with
the values from the transformer's heat-run test report before using any result.
The IEC exponential-equation form (k11, k21, k22) is not implemented here.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ThermalParams:
    dtheta_or: float = 55.0   # top-oil rise over ambient at rated load [K]
    dtheta_hr: float = 23.0   # hot-spot rise over top-oil at rated load [K]
    R: float = 4.5            # ratio of load losses to no-load losses at rated load
    n: float = 0.8            # oil exponent (x in IEC)
    m: float = 0.8            # winding exponent (y in IEC)
    tau_o_min: float = 180.0  # oil time constant [min]
    tau_w_min: float = 4.0    # winding time constant [min]

def steady_state(K, theta_a, p: ThermalParams):
    """Steady-state top-oil and hot-spot temperatures for constant load K."""
    theta_o = theta_a + p.dtheta_or * ((1 + p.R * K**2) / (1 + p.R)) ** p.n
    theta_h = theta_o + p.dtheta_hr * K ** (2 * p.m)
    return theta_o, theta_h

def simulate(K, theta_a, dt_min, p: ThermalParams = ThermalParams(), theta_o0=None, theta_h0=None):
    """Simulate temperatures for load K(t) [p.u.] and ambient theta_a(t) [C], both arrays
    sampled every dt_min minutes. Uses the exact solution of each first-order equation over
    a step (inputs held constant), so it is stable for any step size."""
    K = np.asarray(K, float); theta_a = np.asarray(theta_a, float)
    if K.shape != theta_a.shape:
        raise ValueError("K and theta_a must have the same length")
    if (K < 0).any():
        raise ValueError("load factor cannot be negative")
    n_steps = len(K)
    o, h = np.empty(n_steps), np.empty(n_steps)
    so, sh = steady_state(K[0], theta_a[0], p)
    to = so if theta_o0 is None else theta_o0
    th = sh if theta_h0 is None else theta_h0
    ao, aw = np.exp(-dt_min / p.tau_o_min), np.exp(-dt_min / p.tau_w_min)
    for i in range(n_steps):
        o_ult = theta_a[i] + p.dtheta_or * ((1 + p.R * K[i] ** 2) / (1 + p.R)) ** p.n
        to = o_ult + (to - o_ult) * ao
        h_ult = to + p.dtheta_hr * K[i] ** (2 * p.m)
        th = h_ult + (th - h_ult) * aw
        o[i], h[i] = to, th
    return pd.DataFrame({"K": K, "theta_a": theta_a, "theta_o": o, "theta_h": h})
