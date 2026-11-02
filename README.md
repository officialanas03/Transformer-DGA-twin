# Transformer DGA diagnosis and digital-twin core

[!\[tests](https://github.com/officialanas03/Transformer-DGA-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/officialanas03/Transformer-DGA-twin/actions)

Dissolved gas analysis (DGA) diagnosis for oil-filled transformers, with uncertainty estimates, and a first
thermal and insulation-ageing core for a transformer digital twin. **Research prototype: not validated for operational use.**

<!-- Live demo: add the Streamlit link here after deployment -->

**Live demo:** *https://anas-transformer-dga-twin01.streamlit.app/*

!\[Pipeline overview](docs/images/architecture.png)

## What it does

* **Diagnose a gas sample** with Duval triangle 1 and the IEC 60599 ratio method, optionally alongside a random
forest that returns a *set* of plausible faults with a coverage target (conformal prediction).
* **Process a whole CSV** of samples in the app and download the results.
* **Estimate insulation condition** from CO2/CO and from 2-FAL (furan) results via published correlations.
* **Simulate hot-spot temperature and paper ageing** (DP) with two published models, and recalibrate them to a measured DP.

!\[Duval triangle](docs/images/duval\_triangle.png)

## Quick start

```bash
git clone https://github.com/officialanas03/Transformer-DGA-twin.git
cd Transformer-DGA-twin
python -m venv .venv
# activate: Windows PowerShell  .venv\\Scripts\\Activate.ps1   |   macOS/Linux  source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

The datasets used for the studies are not redistributed (see [DATA.md](DATA.md)). Without them the app runs in
rules-only mode (Duval, IEC, ageing indicators, twin) and data-dependent tests are skipped.

## Repository layout

|Part|Files|Purpose|
|-|-|-|
|Rule-based diagnosis|`dga/rules/duval.py`, `iec.py`|Duval triangle 1; IEC 60599 ratio method (reports "undetermined" or "ambiguous" instead of guessing)|
|Machine learning|`dga/ml/`, `dga/eval/`, `dga/pipeline.py`|Features, random forest, calibration metrics, split-conformal sets, one-call diagnosis|
|Ageing indicators|`dga/ageing/paper.py`|CO2/CO bands, 2-FAL to DP (Chendong, De Pablo), furan condition bands|
|Twin core|`twin/thermal.py`, `twin/ageing.py`, `twin/dingwang.py`|Top-oil/hot-spot model; IEC relative ageing rate; Ekenstam and Ding-Wang/Luo DP kinetics|
|App|`app.py`|Streamlit app: single sample, batch CSV, twin|
|Studies|`run\_\*.py`, `docs/make\_figures.py`|Reproduce the results and figures below|

## Results

On 20 real transformers with a stated fault class (3 of them mixed faults), the three methods cannot be separated:
one case moves a score by 5 points.

!\[Method comparison](docs/images/method\_comparison.png)

|Study|Result|
|-|-|
|Random forest, 5-fold CV inside the augmented set (no transformer IDs)|accuracy about 0.88|
|Same forest tested on 13 real cases|accuracy about 0.62|
|Duval vs the augmented set's labels (non-Normal rows)|agree on 62%|
|"Normal" rows that Duval places in a discharge zone|about 89%|

!\[Conformal coverage](docs/images/conformal\_coverage.png)

Conformal sets reach their 90% target inside the data they were calibrated on and fall short on real cases, as
expected when the real cases do not resemble the calibration data.

### Reproduction check against Luo et al. (Measurement 222, 2023)

!\[Luo et al. Table 2](docs/images/luo\_table2.png)

The levelling-off DP is reproduced (252.13 against 252.15, with moisture as a fraction in Eq. 11) and Table 2's
temperature-only row is matched within 0.8 DP after fitting one rate constant on year 1. That confirms the model
form, not the rate: from the printed constants the rate comes out about 26% too low if k\_DP is per day, and the
paper states no unit. Eq. (11) as printed gives k\_DP <= 0 above about 0.62% moisture although the paper reports
moisture up to 3.2%, so its moisture rows cannot be reproduced.

### Twin demonstration (synthetic)

!\[Twin demo](docs/images/twin\_demo.png)

Everything in this figure is synthetic or assumed: invented load and ambient profiles, placeholder thermal
parameters, an assumed life of 180,000 h to DP 200 at a 110 C hot spot, and a made-up 2-FAL result. It shows how one
measurement can overturn an assumed ageing rate; it says nothing about a real transformer.

## Method notes

* **Duval triangle 1** zones were reconstructed from the nine breakpoints printed in Duval (2002), Fig. 1, and checked against nine
published worked examples. They have not been checked line by line against IEC 60599 or IEEE C57.104.
* **IEC ratios** follow the standard's table as reproduced in two published sources; limits differ slightly between editions and
sit in one dictionary. The standard's gas-level screening is not applied.
* **Random forest** uses five-gas features (three-gas percentages and three log-ratios), is trained without the "Normal" class
and has no PD class.
* **Conformal sets**: split conformal with score 1 - p(true class).
* **Twin**: differential-form top-oil and hot-spot equations (exact first-order update per step); IEC relative ageing rate;
Ekenstam kinetics (Ea about 113 kJ/mol, pre-exponential factor calibrated, no default); Ding-Wang/Luo saturating model.

## Limitations

1. The augmented dataset has no transformer IDs, so cross-validation scores are probably optimistic.
2. Its "Normal" label conflicts with Duval's triangle on most rows, so that class is excluded.
3. The forest cannot output PD and can be confidently wrong on real cases.
4. With 13 to 20 real cases, method comparisons are not statistically meaningful.
5. Thermal parameters are illustrative placeholders; replace them with the heat-run test report values.
6. The IEC exponential thermal form, the Emsley ageing model and the IEC gas-level screening are not implemented.
7. Not yet built: a surrogate layer from CFD data, Kalman/Bayesian live calibration, an API service.
8. The batch tab of a public deployment should not be used with confidential data.

## Key references

* Duval, M. "A Review of Faults Detectable by Gas-in-Oil Analysis in Transformers", IEEE Electr. Insul. Mag. 18(3), 2002.
* Luo, H. et al. "A novel approach to building digital twin transformers by combining virtual-real sensing", Measurement 222, 2023.
* IEC 60599, IEC 60076-7, IEEE C57.104, IEEE C57.91.
* Chendong (1991) and De Pablo furan-DP correlations, as compared in IEEE Trans. Power Delivery 28(2), 2013.
* Ding, H.Z. and Wang, Z.D., "On the degradation evolution equations of cellulose", Cellulose 15, 2008.
* Illias, H.A. and Liang, W.Z., PLOS ONE 13(1): e0191366, 2018.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Datasets are not covered; see [DATA.md](DATA.md).

