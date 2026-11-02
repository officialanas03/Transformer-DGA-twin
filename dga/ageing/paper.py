"""Cellulose-ageing indicators from DGA and furan data.

All relations here are EMPIRICAL and vary with oil type, paper type and
transformer design. Treat results as indicators, not measurements.

Sources (verified while building this project):
 - Chendong: DP = (1.51 - log10(F)) / 0.0035, F = 2-FAL in ppm (mg/kg), quoted for
   150 <= DP <= 1000. As given in Mueller-Schmidt et al. style comparisons
   (IEEE Trans. Power Delivery 28(2), 2013, Eq. 2) and VPS technical note.
 - De Pablo: DP = 1850 / (F + 2.3), quoted for 150 <= DP <= 600 (same IEEE TPD paper).
 - CIGRE-type condition bands for 2-FAL (as tabulated in an Univ. of Craiova paper):
   0-0.1 ppm good (DP 1200-700), 0.1-1 moderate (700-450), 1-10 high ageing (450-250),
   >10 high risk (<250).
 - CO2/CO: a normal band of 3 to 10 is quoted in LAPSE:2023.26404. Check the meaning of
   values below 3 and above 10 against the IEC 60599 text before drawing conclusions.
"""
import numpy as np

def co2_co_ratio(co2, co):
    """CO2/CO from ppm values. Returns inf if CO is 0 and CO2 > 0, nan if both are 0."""
    if co == 0:
        return np.inf if co2 > 0 else np.nan
    return co2 / co

def co2_co_band(ratio):
    if np.isnan(ratio):
        return "not evaluable"
    if ratio < 3:
        return "below 3"
    if ratio <= 10:
        return "3 to 10 (normal band)"
    return "above 10"

def dp_chendong(fal_ppm):
    """Estimated DP from 2-FAL (ppm = mg/kg). Valid only if the result is 150-1000."""
    if fal_ppm <= 0:
        raise ValueError("2-FAL must be positive")
    return (1.51 - np.log10(fal_ppm)) / 0.0035

def dp_depablo(fal_ppm):
    """Estimated DP from 2-FAL (ppm). Valid only if the result is 150-600."""
    if fal_ppm < 0:
        raise ValueError("2-FAL cannot be negative")
    return 1850.0 / (fal_ppm + 2.3)

def furan_condition(fal_ppm):
    """CIGRE-type band for 2-FAL content (ppm)."""
    if fal_ppm < 0:
        raise ValueError("2-FAL cannot be negative")
    if fal_ppm <= 0.1:
        return "good"
    if fal_ppm <= 1:
        return "moderate ageing"
    if fal_ppm <= 10:
        return "high ageing, risk of fault"
    return "high risk of fault"

def in_validity_range(dp, lo, hi):
    return lo <= dp <= hi
