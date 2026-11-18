"""Moisture in oil-paper insulation: Luo et al. (2023) Eqs. (1)-(4) and (13), implemented as printed
where they are clear, with every unclear part flagged below.

Eq. 1  oil-paper equilibrium (T in C):
    C1      = 10^(7.42 - 1670/(273.15+T))                    oil saturation moisture [ppm]
    P1      = exp(9.3876 - 3826.36/(T+273.15-45.47))         water vapour saturation pressure [MPa]
    P3      = P1 / P0                                        dimensionless
    P_V     = (C_oil / C1) * P3                              oil relative saturation x P3
    C_paper = 2.173e-5 * P_V^0.6685 * exp(4725.6/(273.15+T)) [percent]
Eq. 2  D = D0 * exp(k0*C0 + E*(1/T0 - 1/T))
Eq. 4  series solution of dC/dt = D d2C/dx2, surface value C_s at x=0, no flux at x=l.
Eq. 13 C_r = 0.7 * exp(0.0295*(r-3))   [percent, r = operating year]

FLAGS (things the paper does not settle; none of these were verified against another source):
 F1 Labels: Fig. 1 calls the first quantity C1 and the second P1; Eq. 1 on p.5 calls them P1 and P2 and
    inserts P3. The chain of operations is the same; this module uses Fig. 1's names.
 F2 P0 is not given. ASSUMED 0.101325 MPa (1 standard atmosphere). P1 is then dimensionally MPa: my own
    check at 60 C gives 0.0200 MPa, close to the textbook 19.9 kPa for water, which supports MPa.
 F3 Units of P_V in the last line of Eq. 1 are not stated. P_V is taken as dimensionless (ratio x P3, i.e.
    atmospheres). Independent check (web, Nov 2026): the same Freundlich-type equation appears with prefactor
    2.173e-7 (not 2.173e-5) in a review, MDPI Energies 17(9):2089 (2024), Eq. 14, attributed there to Fessler
    and Rouse; that page gives no units for P_V. My own arithmetic: P_V in Pa with 2.173e-7 gives absurd values
    (27 % at 60 C, 4 % relative saturation); P_V in mmHg with 2.173e-7 gives values within about 20 % of this
    implementation (1.03 % vs 1.22 % at 60 C). So the paper's form looks consistent with the literature, but
    its prefactor is printed 100 times larger than in that review and the unit is unconfirmed.
    UNRESOLVED: 10 ppm oil gives 5.7 % paper moisture at 20 C here, while Fig. 9 (10 ppm, cool bottom) has a
    colour scale of about 0.5 to 0.8 %. Do not quote absolute values from this function as findings.
 F4 Printed Eq. 3 (dC/dx = C at x=0) is dimensionally inconsistent. Eq. 4 is the solution for a SURFACE
    VALUE at x=0 and no flux at x=l, so that is what is implemented: C(0,t) = C_s. Eq. 4 as printed has
    zero initial moisture and uses the symbol C0 for the surface value; here C_i (initial) is a parameter.
 F5 The unit of C0 inside D (k0*C0) is not stated; percent is ASSUMED. E = 8074 is taken as kelvin.
 F6 D depends on C, but Eq. 4 needs a constant D. D is evaluated at the surface value (an assumption).
 F7 The pressboard thickness l is not given; the caller must supply it (no default).
"""
import numpy as np

P0_MPA = 0.101325          # ASSUMED (flag F2)
D0 = 1.34e-13              # m2/s, Eq. 2
K0 = 0.5
E_D = 8074.0               # K, unit assumed (F5)
T0_K = 298.0

def oil_saturation_ppm(theta_c):
    """C1: water solubility limit of the oil [ppm] (Eq. 1, first line)."""
    return 10.0 ** (7.42 - 1670.0 / (273.15 + np.asarray(theta_c, float)))

def vapour_pressure_mpa(theta_c):
    """P1 of Eq. 1 / Fig. 1 [MPa, see F2]."""
    t = np.asarray(theta_c, float) + 273.15
    return np.exp(9.3876 - 3826.36 / (t - 45.47))

def equilibrium_paper_moisture_pct(theta_c, c_oil_ppm):
    """C_paper [percent] in equilibrium with oil holding c_oil_ppm at temperature theta_c [C]."""
    c_oil_ppm = np.asarray(c_oil_ppm, float)
    if np.any(c_oil_ppm < 0):
        raise ValueError("oil moisture cannot be negative")
    p_v = c_oil_ppm / oil_saturation_ppm(theta_c) * (vapour_pressure_mpa(theta_c) / P0_MPA)
    return 2.173e-5 * p_v ** 0.6685 * np.exp(4725.6 / (273.15 + np.asarray(theta_c, float)))

def relative_saturation(theta_c, c_oil_ppm):
    return np.asarray(c_oil_ppm, float) / oil_saturation_ppm(theta_c)

def diffusion_coefficient(theta_c, c0_pct):
    """D [m2/s] from Eq. 2 (flags F5, F6)."""
    t = np.asarray(theta_c, float) + 273.15
    return D0 * np.exp(K0 * c0_pct + E_D * (1.0 / T0_K - 1.0 / t))

def diffusion_series(x, t_s, d, l, c_s, c_i=0.0, n_terms=200):
    """Eq. 4 generalised to an initial value c_i: slab of thickness l, C(0,t)=c_s, dC/dx(l,t)=0.
    With c_i = 0 it is Eq. 4 as printed (C0 -> c_s). x in m (scalar or array), t_s in s, d in m2/s."""
    x = np.asarray(x, float)
    if l <= 0:
        raise ValueError("thickness l must be positive")
    if t_s < 0:
        raise ValueError("time cannot be negative")
    if t_s == 0:
        return np.where(x <= 0, c_s, c_i)
    n = np.arange(n_terms)[:, None] if x.ndim else np.arange(n_terms)
    a = (0.5 + n) * np.pi
    term = ((-1.0) ** n / (2 * n + 1)) * np.exp(-t_s * d * a ** 2 / l ** 2) * np.cos(a * (l - x) / l)
    return c_s - (4.0 / np.pi) * (c_s - c_i) * term.sum(axis=0)

def diffusion_fd(x, t_s, d, l, c_s, c_i=0.0, n_nodes=401, n_steps=4000):
    """Independent check of diffusion_series: implicit finite differences (backward Euler), same
    boundary conditions. Used only in tests."""
    xs = np.linspace(0.0, l, n_nodes)
    h = xs[1] - xs[0]
    dt = t_s / n_steps
    r = d * dt / h ** 2
    u = np.full(n_nodes, float(c_i)); u[0] = c_s
    A = np.zeros((n_nodes, n_nodes))
    A[0, 0] = 1.0
    for i in range(1, n_nodes - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -r, 1 + 2 * r, -r
    A[-1, -1], A[-1, -2] = 1 + 2 * r, -2 * r          # ghost-node no-flux condition
    Ainv = np.linalg.inv(A)
    for _ in range(n_steps):
        b = u.copy(); b[0] = c_s
        u = Ainv @ b
    return np.interp(x, xs, u)

def mean_paper_moisture_pct(x_grid, t_s, d, l, c_s, c_i=0.0):
    """Average moisture across the thickness (trapezoid)."""
    prof = diffusion_series(x_grid, t_s, d, l, c_s, c_i)
    trap = getattr(np, "trapezoid", None) or np.trapz     # numpy 1.x / 2.x
    return float(trap(prof, x_grid) / (x_grid[-1] - x_grid[0]))

def total_moisture_pct(r_year):
    """Eq. 13: total paper moisture [percent] at operating year r (empirical, ref. [36] of the paper)."""
    return 0.7 * np.exp(0.0295 * (np.asarray(r_year, float) - 3.0))
