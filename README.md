# dga-twin

Dissolved gas analysis (DGA) diagnosis and a first thermal/ageing core for a transformer digital twin.
Research prototype. Not validated for operational use.

## What is in the repository

| Part | Files | What it does |
|---|---|---|
| Data handling | `dga/data/` | Loader that needs the five core gases (CO and CO2 optional) and one shared label vocabulary |
| Rule-based diagnosis | `dga/rules/duval.py`, `iec.py` | Duval triangle 1 and the IEC 60599 ratio method |
| Machine learning | `dga/ml/`, `dga/eval/` | Features, random forest baseline, calibration metrics, split-conformal prediction sets |
| Ageing indicators | `dga/ageing/paper.py` | CO2/CO bands, 2-FAL to DP (Chendong, De Pablo), furan condition bands |
| Twin core | `twin/thermal.py`, `twin/ageing.py`, `twin/dingwang.py` | Top-oil/hot-spot model, IEC relative ageing rate, Ekenstam/Arrhenius kinetic DP, and the saturating Ding-Wang/Luo DP model with time-temperature-moisture superposition; both DP models can be recalibrated to a measurement |
| App | `app.py` | Streamlit app: DGA diagnosis tab and a twin tab |
| Studies | `run_luo_check.py` (what can be reproduced from Luo et al. 2023), `run_baseline.py`, `run_duval.py`, `run_compare.py`, `run_uncertainty.py`, `run_twin_demo.py` | Reproduce the results below |

## Data

Three datasets in `data/`, all derived from files supplied by the author (sources not independently verified):

- `dga_augmented.csv`: 4,244 rows with classes Normal, D1, D2, T1&T2, T3. No transformer IDs.
- `dga_iec_25cases.csv`: 25 transformers from a published table, with free-text fault descriptions; 20 have a stated class (3 mixed).
- `dga_illias2018_s1.csv`: S1 Table of Illias and Liang (PLOS ONE 2018). **Not used for training**: all 32 "high intensity discharge" rows have acetylene of 538,000 to 684,000 ppm, which is physically implausible, and 14 rows are duplicates.

## Method notes

- **Duval triangle 1** zones were reconstructed from the nine breakpoints printed in Duval (2002), Fig. 1, and checked against nine published worked examples. Not checked line by line against IEC 60599 / IEEE C57.104.
- **IEC ratios** follow the standard's table as reproduced in two published sources; a sample is labelled only when exactly one fault matches, otherwise "undetermined" or "ambiguous". Limits differ slightly between editions and are in one dictionary.
- **Random forest** uses five-gas features (three-gas percentages and three log-ratios) so it can be tested on datasets without CO2. Trained without the Normal class and without PD.
- **Conformal sets**: split conformal with score 1 - p(true class).
- **Twin**: differential-form top-oil and hot-spot equations (exact first-order update per step); IEC relative ageing rate; Ekenstam kinetics with Arrhenius rate, Ea about 113 kJ/mol. The pre-exponential factor has no default and must be calibrated.

## Results

| Study | Result |
|---|---|
| Random forest, 5-fold CV inside the augmented set (no groups) | accuracy about 0.88 |
| Same forest tested on 13 real cases | accuracy about 0.62 |
| Real cases (20 with a stated class): Duval / IEC / random forest | 0.60 / 0.50 / 0.57 (IEC answers on 14 of 20, right on 10) |
| Duval vs augmented labels (non-Normal rows) | agrees on 62% |
| "Normal" rows placed by Duval in a discharge zone | about 89% |
| Conformal coverage inside the augmented set (target 90%) | about 89% |
| Conformal coverage on the 16 real non-PD cases | about 79 to 81% |

With 13 to 20 real cases these comparisons cannot separate the methods: one case moves a score by 5 points or more.

## Reproduction check against Luo et al. (Measurement 222, 2023)

`python run_luo_check.py` prints the details. In short:

- The levelling-off DP is reproduced (252.13 against 252.15), but only if moisture in Eq. (11) is a fraction (0.18 % = 0.0018).
- Table 2 row I (temperature only) is reproduced to within 0.8 DP at years 1, 10, 25, 40 and 48, **after fitting one rate constant on year 1**. This confirms the model form, not the rate.
- From the printed constants and the monthly hot-spot values (Fig. 6) the rate comes out about 26 % too low if k_DP is per day, and far too high if per hour; the paper does not state the unit.
- Eq. (11) as printed gives k_DP <= 0 above about 0.62 % moisture, yet the paper reports moisture up to 3.2 %, so rows II and III of Table 2 (moisture) cannot be reproduced and the printed coefficients should not be used above that level.

## Limitations

1. The augmented set has no transformer IDs, so cross-validation scores are probably optimistic; the drop from 0.88 to 0.62 on real cases suggests leakage, distribution shift, or both.
2. The "Normal" label in the augmented set conflicts with Duval's triangle on most rows and the class is excluded from the main model.
3. The forest has no PD class. Confident wrong answers occur on real cases (for example case 13 at 89% confidence).
4. Conformal coverage held inside the augmented set but fell short on real cases, as expected when data are not exchangeable with the calibration set.
5. Thermal parameters in `twin/thermal.py` are illustrative placeholders. The starting ageing constant in the demo assumes DP 1000 to 200 after 180,000 h at 110 C hot spot; verify against IEC 60076-7.
6. The IEC exponential-equation form (k11, k21, k22) and the Emsley ageing model are not implemented.
7. The IEC 60599 gas-level screening (typical values) is not applied.
8. Not yet built: surrogate layer from CFD data, Kalman/Bayesian live calibration, FastAPI service.

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q                 # 58 tests
python run_baseline.py
python run_duval.py
python run_compare.py
python run_uncertainty.py
python run_twin_demo.py             # writes twin_demo.png (synthetic)
streamlit run app.py
```

## Key references

- Duval, M. "A Review of Faults Detectable by Gas-in-Oil Analysis in Transformers", IEEE Electr. Insul. Mag. 18(3), 2002.
- IEC 60599 and IEC 60076-7; IEEE C57.104 and C57.91.
- Chendong (1991) and De Pablo furan-DP correlations, as compared in IEEE Trans. Power Delivery 28(2), 2013.
- Emsley and Stevens, Cellulose 1, 1994 (cellulose degradation kinetics).
