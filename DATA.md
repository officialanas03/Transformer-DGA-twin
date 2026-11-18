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

## Files that are in the repository

| File | What it is |
|---|---|
| `reference/luo2023_numbers.csv` | Numbers printed in Luo et al. (2023), with PDF page and equation/figure/table reference; figure readings are marked approximate. No paper text or figures are included, and the paper PDF is not distributed |
| `examples/ansys/*.csv` | Own ANSYS exports from the assumed placeholder model (see the README in that folder). Not measurements |

The paper's own data are "available on request" and have not been obtained.

## Where real data might come from (checked Nov 2026; nothing here is committed)

No open dataset was found that pairs winding temperature, moisture and DP, which is what Luo et al. use. What exists:

| Source | What it holds | Licence and caveats |
|---|---|---|
| ETT (Electricity Transformer Temperature), GitHub and Hugging Face | Load covariates (HUFL, HULL, MUFL, MULL, LUFL, LULL) and oil temperature (OT) for two transformers in a province in China, 2016-07 to 2018-07, 1-minute, 15-minute and hourly files | The Hugging Face card says CC BY 4.0. The GitHub copy I opened has a LICENSE file whose type I did not read: check the official repository before use. OT is top-oil temperature, not winding hot spot, and there is no ambient, moisture or DP column. Useful to test the thermal model and the Kalman filter on real load and oil temperature |
| UK Power Station Transformer DGA data 2010-2015, Zenodo 10.5281/zenodo.5796243 | DGA records from 13 UK power-station transformers (Lewis, Wild, Allcock) | CC BY 4.0; cite Lewis et al. Gas data only (column names are in the archive's readme, not checked): no DP, moisture or temperature |
| Published tables, e.g. Advances in Science, Technology and Engineering Systems Journal 6(6), South African transformers | 2-FAL and measured DP for five transformers (Table 3) and furan-DP model comparisons | CC BY-SA 4.0 (share-alike): cite, and take care before redistributing numbers. Tiny, but real: can test the Chendong and De Pablo correlations |
| IEEE DataPort (DGA and hot-spot keyword pages) | Some DGA datasets | Needs an account; licence differs per dataset; not checked |
| CIGRE TB 735 (Lessard et al., post-mortem DP of a 48-year-old transformer) | The DP distribution Luo et al. compare with | Paywalled; not retrieved |

Other routes (not verified, suggestions only): your department's high-voltage lab, utility or manufacturer contacts through your professor, and the authors of
Luo et al. (no reply so far). If none works, state in the report that no real data were used.
