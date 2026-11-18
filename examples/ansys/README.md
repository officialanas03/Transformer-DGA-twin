# ANSYS example exports (own work, ASSUMED geometry)

`T_load0p74.csv` and `T_load1p30.csv` were produced by `ansys/luo_thermal.inp` (film coefficient 72 W/m2K, load 0.74 and 1.3).
Geometry, losses, oil profile and film coefficient are PLACEHOLDERS, not from Luo et al. (2023). The film coefficient was
tuned so the LV maximum is near the paper's 61.29 C: that is a fit. In these files the HV winding is hotter than the LV
winding, which contradicts the paper (LV hotter than HV).
