"""Steady-state 2D axisymmetric winding temperature by finite volumes (numpy only).

Solves the SAME ASSUMED model as ansys/luo_thermal.inp: two copper rectangles with uniform volumetric heat
generation, conduction inside, and convection (film coefficient hf) on all outer edges to oil whose temperature
rises linearly from t_bot at y=0 to t_top at y=height. The windings only interact through the oil, so each is solved
separately. All numbers are PLACEHOLDERS (not from Luo et al.). Used here to (a) cross-check the ANSYS exports and
(b) generate model-based snapshots for the surrogate when ANSYS runs are not available.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Assumed:
    lv: tuple = (0.250, 0.290)
    hv: tuple = (0.340, 0.400)
    height: float = 1.2
    k_cu: float = 385.0
    q0: float = 1.0e5            # W/m3 at load 0.74
    load0: float = 0.74
    t_bot: float = 20.0
    t_top: float = 40.0

def heat_generation(load, a=Assumed()):
    """ASSUMED: losses proportional to current squared."""
    return a.q0 * (load / a.load0) ** 2

def solve_winding(r_in, r_out, height, q, hf, t_oil, k, nr=16, nz=120):
    """Cell-centred temperatures [nz, nr] and cell-centre coordinates (r, y)."""
    dr, dz = (r_out - r_in) / nr, height / nz
    r = r_in + dr * (np.arange(nr) + 0.5)
    y = dz * (np.arange(nz) + 0.5)
    idx = lambda i, j: j * nr + i
    n = nr * nz
    A = np.zeros((n, n)); b = np.zeros(n)
    def gb(area, dist):                    # film in series with half-cell conduction
        return 1.0 / (1.0 / (hf * area) + dist / (2 * k * area))
    for j in range(nz):
        for i in range(nr):
            p = idx(i, j)
            rw, re = r[i] - dr / 2, r[i] + dr / 2
            vol = r[i] * dr * dz                       # per radian
            b[p] += q * vol
            if i > 0:
                g = k * rw * dz / dr; A[p, p] += g; A[p, idx(i - 1, j)] -= g
            else:
                g = gb(rw * dz, dr / 2); A[p, p] += g; b[p] += g * t_oil(y[j])
            if i < nr - 1:
                g = k * re * dz / dr; A[p, p] += g; A[p, idx(i + 1, j)] -= g
            else:
                g = gb(re * dz, dr / 2); A[p, p] += g; b[p] += g * t_oil(y[j])
            ar = r[i] * dr
            if j > 0:
                g = k * ar / dz; A[p, p] += g; A[p, idx(i, j - 1)] -= g
            else:
                g = gb(ar, dz / 2); A[p, p] += g; b[p] += g * t_oil(0.0)
            if j < nz - 1:
                g = k * ar / dz; A[p, p] += g; A[p, idx(i, j + 1)] -= g
            else:
                g = gb(ar, dz / 2); A[p, p] += g; b[p] += g * t_oil(height)
    return np.linalg.solve(A, b).reshape(nz, nr), r, y

def solve_model(load, hf, a=Assumed(), t_top=None, q_hv_factor=1.0, nr_lv=10, nr_hv=14, nz=120):
    """Both windings. Returns dict with fields and maxima. t_top overrides the assumed top-oil temperature."""
    tt = a.t_top if t_top is None else t_top
    oil = lambda y: a.t_bot + (tt - a.t_bot) * y / a.height
    q = heat_generation(load, a)
    out = {}
    for name, (r0, r1), qf, nr in (("LV", a.lv, 1.0, nr_lv), ("HV", a.hv, q_hv_factor, nr_hv)):
        T, r, y = solve_winding(r0, r1, a.height, q * qf, hf, oil, a.k_cu, nr=nr, nz=nz)
        i = np.unravel_index(np.argmax(T), T.shape)
        out[name] = dict(T=T, r=r, y=y, T_max=float(T.max()), y_at_max=float(y[i[0]]), T_min=float(T.min()))
    return out

def hot_spot(load, hf, **kw):
    """Hottest temperature in either winding and which winding it is in."""
    m = solve_model(load, hf, **kw)
    w = max(m, key=lambda k: m[k]["T_max"])
    return m[w]["T_max"], w
