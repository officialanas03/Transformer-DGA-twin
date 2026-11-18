"""How much of the Table 3 mismatch is the assumed thermal model? Every variant is fitted to 61.29 C only (a FIT);
the overload value and the LV-versus-HV ordering are unfitted checks. ALL VARIANTS ARE PLACEHOLDERS."""
import pandas as pd
from twin import thermal_variants as tv, pipeline_dp as pdp

PAPER_T3 = {"LV": (252.15, 676.33, 927.5), "HV": (645.33, 811.4, 879.04)}
rows = []
for name, tt, qf in [("A  oil 20-40 C, same loss density (ANSYS case)", 40, 1.0),
                     ("B  oil 20-40 C, HV loss density x0.6", 40, 0.6),
                     ("C  oil 20-50 C, HV x0.6", 50, 0.6),
                     ("D  oil 20-60 C, HV x0.6", 60, 0.6)]:
    v = tv.variant(tt, qf)
    s = pdp.position_summary(pdp.dp_map_with_moisture(v["base"], 48))
    rows.append(dict(variant=name, HF=round(v["hf"], 1), LV_min=round(v["lv_min"], 1), HV_max=round(v["hv_max"], 1),
                     LV_hotter=v["lv_max"] > v["hv_max"], LV_at_1p3=round(v["lv_overload"], 1),
                     LV_top_mid_bot=" / ".join(f"{s['LV'][k]:.0f}" for k in ("top", "middle", "bottom")),
                     HV_top_mid_bot=" / ".join(f"{s['HV'][k]:.0f}" for k in ("top", "middle", "bottom"))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(df.to_string(index=False))
print("\npaper: hot spot 61.29 C (fitted here), 112.84 C at load 1.3, coolest about 20 C, LV hotter than HV")
print("paper Table 3 year 48 LV top/mid/bottom 252 / 676 / 928, HV 645 / 811 / 879 (model: case II, moisture field, 48 years)")
