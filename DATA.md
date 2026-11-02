# Data

The datasets used for the studies in this repository are **not** distributed here. I (the repository owner)
have not confirmed that their licences allow redistribution. Without them:

- the Duval, IEC and ageing code, the twin core and the Streamlit app all work;
- the random forest and the study scripts that need `data/` (`run_baseline.py`, `run_duval.py`,
  `run_compare.py`, `run_uncertainty.py`) are unavailable, and the tests that need them are skipped.

To reproduce everything, place these files in `data/` (same column names):

| File | Contents | Source and licence |
|---|---|---|
| `dga_augmented.csv` | 4,244 rows: H2, CH4, C2H6, C2H4, C2H2, CO, CO2, fault (Normal, D1, D2, T1&T2, T3) | _To be filled in by the repository owner_ |
| `dga_iec_25cases.csv` | 25 transformers: H2, CH4, C2H6, C2H4, C2H2, fault, fault_description | _To be filled in by the repository owner_ |
| `dga_illias2018_s1.csv` | S1 Table of Illias and Liang, PLOS ONE 13(1): e0191366 (2018), CC BY 4.0. Not used for training: its D2 rows have physically implausible acetylene values | PLOS ONE supporting information |
