"""What can and cannot be reproduced from Luo et al., Measurement 222 (2023) 113714."""
import numpy as np
from twin.dingwang import DingWangAgeing, w_star, k_dp, alpha_T, moisture_limit_pct

TABLE2_I = {1: 1167.52, 10: 920.92, 25: 648.27, 40: 486.52, 48: 429.17}
MONTHLY_HOTSPOT = [61.5, 53.0, 61.3, 59.0, 51.1, 51.4, 48.4, 49.3, 53.2, 60.6, 70.6, 75.6]   # Fig. 6 bar labels, C

print("1) Levelling-off DP at 0.18 % moisture:", round(DingWangAgeing(0.18).dp_levelling_off, 2), "(paper: 252.15)")

m = DingWangAgeing(0.18); m.advance(np.array([61.29]), 365.0); m.calibrate_to_measurement(TABLE2_I[1])
print("\n2) Table 2 row I, rate calibrated on year 1 only (a fit, not a prediction):")
for y, dp in TABLE2_I.items():
    mm = DingWangAgeing(0.18); mm.k_scale = m.k_scale; mm.advance(np.array([61.29]), 365.0 * y)
    print(f"   year {y:>2}: paper {dp:8.2f}  model {mm.dp():8.2f}  diff {mm.dp() - dp:+.2f}")

a = alpha_T(np.array(MONTHLY_HOTSPOT)).mean()
print("\n3) Rate from first principles (printed k_DP x mean alpha_T of the 12 monthly hot-spot values):")
for unit, t in [("per day", 365.0), ("per hour", 8760.0)]:
    print(f"   k_DP {unit}: {k_dp(0.18) * a * t:.4f} per year")
need = m.k_scale * k_dp(0.18) * alpha_T(61.29) * 365.0
print(f"   rate implied by Table 2 row I: {-np.log(1 - (1 - TABLE2_I[1]/1200)/w_star(0.18)):.4f} per year")
print("   -> 'per day' undershoots by about", f"{100*(1 - k_dp(0.18)*a*365/(-np.log(1 - (1 - TABLE2_I[1]/1200)/w_star(0.18)))):.0f} %;",
      "the time unit of k_DP is not stated in the paper")

print(f"\n4) Printed Eq. (11): k_DP reaches zero at {moisture_limit_pct():.2f} % moisture, but the paper reports moisture up to 3.2 %.")
print("   Table 2 rows II and III (moisture) cannot be reproduced: the moisture at the hot spot is not given.")
