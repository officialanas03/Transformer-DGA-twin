"""Luo et al. (2023) reproduction report: what is reproduced, what is only fitted, what is impossible.
Everything labelled PLACEHOLDER is an assumption of this repository, not a paper value."""
import numpy as np
from twin import luo_chain as lc
from twin import moisture as mo
from twin.dingwang import k_dp, w_star, alpha_C, moisture_limit_pct
from twin.paper_numbers import load, value

R = load()
def row(first): return {y: value(f"c{n}", R) for y, n in zip((1, 10, 25, 40, 48), range(first, first + 5))}
T2 = {"I": row(67), "II": row(72), "III": row(77)}
Y = [1, 10, 25, 40, 48]
ks = lc.fit_k_scale(T2["I"][1])

print("== Moisture step (Eq. 1, 2, 4, 13) - uses P0 = 1 atm (assumed) and l = 3 mm (PLACEHOLDER)")
for th in (20, 40, 60, 80):
    print(f"   oil 10 ppm at {th:>2} C: relative saturation {float(mo.relative_saturation(th, 10)):.3f}, "
          f"equilibrium paper moisture {float(mo.equilibrium_paper_moisture_pct(th, 10)):.3f} %")
l = 3e-3
for th in (40, 60):
    cs = float(mo.equilibrium_paper_moisture_pct(th, 10)); d = float(mo.diffusion_coefficient(th, cs))
    t_half = next(t for t in np.geomspace(1, 1e10, 4000) if float(mo.diffusion_series(l, t, d, l, cs, 0.0)) >= 0.5 * cs)
    print(f"   {th} C: D = {d:.2e} m2/s, far side of 3 mm reaches 50 % of surface value after {t_half/3600:.1f} h")
t_match = next(t for t in np.arange(20, 120, 0.5) if float(mo.equilibrium_paper_moisture_pct(t, 10)) <= 0.75)
print(f"   MISMATCH with Fig. 9 (scale about 0.5-0.8 % at 10 ppm, cool bottom): Eq. 1 as read here needs about {t_match:.0f} C to give 0.75 %. "
      "The equation's form matches a literature review (prefactor there 2.173e-7, unit unstated); P_V unit unresolved (see twin/moisture.py, flag F3).")
print("   Eq. 13: year 3 -> %.2f %%, year 48 -> %.2f %% (paper's local maximum: 3.2 %%)" % (mo.total_moisture_pct(3), mo.total_moisture_pct(48)))

print("\n== Table 2 (hot spot, DP). paper vs model, model - paper")
print("   rate: k_DP x %.3f, FITTED to row I year 1 (printed constants alone give about 26 %% less per year)" % ks)
cases = [("I   temperature only", "none", 0, T2["I"]),
         ("II  + moisture (VARIANT eq13)", "eq13", 0, T2["II"]),
         ("II  + moisture (VARIANT const 0.70 %)", "const0p70", 0, T2["II"]),
         ("III + 24 h overload (VARIANT eq13)", "eq13", 24, T2["III"]),
         ("III + 24 h overload (VARIANT const 0.70 %)", "const0p70", 24, T2["III"])]
for name, var, ol, ref in cases:
    got = lc.table2_row(Y, var, ks, overload_hours=ol)
    print(f"   {name}")
    print("     year    " + "".join(f"{y:>9}" for y in Y))
    print("     paper   " + "".join(f"{ref[y]:9.2f}" for y in Y))
    print("     model   " + "".join(f"{got[y]:9.2f}" for y in Y))
    print("     diff    " + "".join(f"{got[y]-ref[y]:+9.2f}" for y in Y))

print("\n== Why Eq. 11 cannot be used at the paper's moisture levels (C as a fraction)")
print(f"   k_DP(C) = 0.0013 - 0.2015 C - 1.55 C^2 is zero at C = {moisture_limit_pct():.3f} % and negative above it.")
for c in (0.0, 0.18, 0.3, 0.5, 0.6):
    print(f"   C = {c:4.2f} %: k_DP = {0.0013-0.2015*c/100-1.55*(c/100)**2:+.6f}  (relative to 0.18 %: {(0.0013-0.2015*c/100-1.55*(c/100)**2)/k_dp(0.18):5.2f})")
print(f"   Even where k_DP > 0, 0.5 % vs 0.18 %: k_DP falls to {k_dp(0.5)/k_dp(0.18):.2f}x while alpha_C rises "
      f"{lc.relative_rate_moisture_only(0.5, 0.18):.2f}x, net {lc.relative_rate_moisture_only(0.5, 0.18, include_k_dp=True):.2f}x: "
      "as printed, moisture SLOWS ageing.")

print("\n== Other claims")
print(f"   '7.4 times faster at 2 % than at 0.2 %': alpha_C alone gives {lc.relative_rate_moisture_only(2.0, 0.2):.2f}x at 61.29 C; "
      "with Eq. 11's k_DP the ratio is undefined (k_DP < 0 at 2 %).")
print(f"   'time to DP 250 at 61.29 C': levelling-off DP at 0.18 % is {1200*(1-w_star(0.18)):.2f} > 250, so Eq. 5 never reaches 250.")
print(f"   alpha_C at 61.29 C: {float(alpha_C(0.18, 61.29)):.3f} at 0.18 %, {float(alpha_C(2.0, 61.29)):.3f} at 2 %")

print("\n== Status")
print("   REPRODUCED : levelling-off DP 252.13 vs 252.15 (C as a fraction); Eq. 13 values; Eq. 4 series (vs finite differences)")
print("   FITTED     : Table 2 row I (one rate constant on year 1; later years within 1 DP) - confirms form, not rate")
print("   VARIANTS   : rows II, III with assumed moisture and a frozen 0.18 % master curve - not the paper's method")
print("   IMPOSSIBLE : rows II, III as printed (Eq. 11); the 7.4x claim; DP 250 target; hot-spot moisture is not given")
