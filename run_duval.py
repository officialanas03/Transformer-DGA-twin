"""Duval Triangle 1 against the labelled datasets."""
import warnings
import pandas as pd
from dga.data.loader import load_dga
from dga.data.labels import to_coarse
from dga.rules.duval import classify_df

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

aug = load_dga("data/dga_augmented.csv")
aug["duval"] = to_coarse(classify_df(aug))
aug["truth"] = aug.fault.where(aug.fault == "Normal", to_coarse(aug.fault))
print("AUGMENTED SET: Duval zone (rows) vs label (columns), counts")
print(pd.crosstab(aug.duval, aug.truth))
nn = aug[aug.truth != "Normal"]
print(f"\nagreement on non-Normal rows: {(nn.duval == nn.truth).mean():.3f}  (n={len(nn)})")

ext = load_dga("data/dga_iec_25cases.csv")
ext["duval"] = to_coarse(classify_df(ext))
ext["truth"] = to_coarse(ext.fault)
print("\n25 REAL CASES (single-label rows):")
print(ext[["H2", "CH4", "C2H4", "C2H2", "truth", "duval"]].to_string())
print(f"\nagreement: {(ext.duval == ext.truth).mean():.3f}  (n={len(ext)})")
sub = ext[ext.truth != "PD"]
print(f"agreement excluding PD: {(sub.duval == sub.truth).mean():.3f}  (n={len(sub)})")
