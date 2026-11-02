"""Synthetic demonstration of the thermal + ageing core with one measurement-based recalibration.

EVERYTHING HERE IS SYNTHETIC OR ASSUMED. The load and ambient profiles are invented, the thermal
parameters are placeholders (see twin/thermal.py), the starting ageing constant assumes
DP 1000 -> 200 after 180,000 h at a 110 C hot spot (check against IEC 60076-7 before relying on it),
and the "measured" 2-FAL value is made up to show how a measurement changes the prediction.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twin.thermal import ThermalParams, simulate
from twin.ageing import (KineticAgeing, calibrate_A_from_life, arrhenius_factor,
                         relative_ageing_rate, remaining_life_hours, DP_END_OF_LIFE)
from dga.ageing.paper import dp_chendong

HOURS = 24 * 365
t = np.arange(HOURS)                                      # hourly samples
day, year = 2 * np.pi * (t % 24) / 24, 2 * np.pi * t / HOURS
K = np.clip(0.75 + 0.15 * np.sin(day - 2.0) + 0.05 * np.cos(year), 0.2, None)   # synthetic load, p.u.
theta_a = 18 + 12 * np.cos(year - np.pi) + 4 * np.sin(day - 2.5)                  # synthetic ambient, C

df = simulate(K, theta_a, dt_min=60.0, p=ThermalParams())
exposure_per_year = float(np.sum(arrhenius_factor(df.theta_h)))                  # hours-equivalent
V = relative_ageing_rate(df.theta_h, upgraded=True)
print(f"hot-spot: mean {df.theta_h.mean():.1f} C, max {df.theta_h.max():.1f} C")
print(f"IEC relative ageing (thermally upgraded paper): mean V = {V.mean():.3f} -> {V.sum():.0f} equivalent hours per calendar year")

A0 = calibrate_A_from_life(110.0, 180000.0)
years = np.arange(0, 61)
dp_assumed = 1.0 / (1.0 / 1000.0 + A0 * exposure_per_year * years)

# Recalibrate at year 8 with a (synthetic) 2-FAL measurement
T_MEAS, FAL = 8, 0.8
dp_meas = dp_chendong(FAL)
m = KineticAgeing(dp0=1000.0, A=A0, E=exposure_per_year * T_MEAS)
print(f"year {T_MEAS}: model DP = {m.dp():.0f}; synthetic 2-FAL {FAL} ppm -> Chendong DP = {dp_meas:.0f}")
rul_before = remaining_life_hours(m, exposure_per_year / HOURS) / HOURS
A1 = m.calibrate_to_measurement(dp_meas)
rul_after = remaining_life_hours(m, exposure_per_year / HOURS) / HOURS
print(f"A recalibrated: {A0:.3e} -> {A1:.3e}  (x{A1 / A0:.1f})")
print(f"remaining life to DP {DP_END_OF_LIFE:.0f}: {rul_before:.1f} years before the measurement, {rul_after:.1f} years after")
dp_recal = 1.0 / (1.0 / 1000.0 + A1 * exposure_per_year * years)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(t[:24 * 7] / 24, df.theta_h[:24 * 7], label="winter week hot-spot")
mid = HOURS // 2
ax[0].plot(t[:24 * 7] / 24, df.theta_h[mid:mid + 24 * 7].values, label="summer week hot-spot")
ax[0].set_xlabel("day"); ax[0].set_ylabel("temperature [C]"); ax[0].legend(); ax[0].set_title("Synthetic hot-spot temperature")
ax[1].plot(years, dp_assumed, label="assumed constants")
ax[1].plot(years, dp_recal, label="recalibrated at year 8")
ax[1].scatter([T_MEAS], [dp_meas], color="k", zorder=3, label="synthetic furan-derived DP")
ax[1].axhline(DP_END_OF_LIFE, color="gray", ls="--", label="DP 200")
ax[1].set_xlabel("years"); ax[1].set_ylabel("DP"); ax[1].legend(); ax[1].set_ylim(0, 1050); ax[1].set_title("Kinetic DP projection")
fig.tight_layout(); fig.savefig("twin_demo.png", dpi=130)
print("saved twin_demo.png")
