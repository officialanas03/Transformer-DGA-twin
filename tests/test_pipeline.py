from pathlib import Path
import pandas as pd
import pytest
from dga.pipeline import Diagnoser, diagnose_rules, diagnose_table, MAX_BATCH_ROWS

DATA = Path(__file__).resolve().parents[1] / "data" / "dga_augmented.csv"
_D = None
def diag():
    """The forest, or skip the test when the (unshipped) training data is missing."""
    global _D
    if not DATA.exists():
        pytest.skip("data/dga_augmented.csv not available")
    if _D is None:
        _D = Diagnoser(DATA)
    return _D

# ---- tests that need no data ----
def test_rules_only_arcing_case():
    out = diagnose_rules(h2=500, ch4=100, c2h6=20, c2h4=200, c2h2=600)
    assert out["duval"] in ("D1", "D2")

def test_rules_all_zero_and_negative():
    assert diagnose_rules(10, 0, 0, 0, 0)["duval"] is None
    with pytest.raises(ValueError):
        diagnose_rules(-1, 0, 0, 0, 0)

def test_rules_ageing_fields():
    out = diagnose_rules(100, 50, 20, 30, 1, co=500, co2=3000, fal_ppm=1.0)
    assert out["co2_co"]["ratio"] == 6.0
    assert out["furan"]["dp_chendong"] == pytest.approx(1.51 / 0.0035)

def test_batch_table_without_forest():
    df = pd.DataFrame({"H2": [500, 10, -5, None], "CH4": [100, 0, 1, 1], "C2H6": [20, 0, 1, 1],
                       "C2H4": [200, 0, 1, 1], "C2H2": [600, 0, 1, 1]})
    res = diagnose_table(df)
    assert len(res) == 4
    assert res.duval.iloc[0] in ("D1", "D2")
    assert pd.isna(res.duval.iloc[1])                      # all fault gases zero
    assert "invalid" in res.note.iloc[2] and "invalid" in res.note.iloc[3]

def test_batch_table_validation():
    with pytest.raises(ValueError):
        diagnose_table(pd.DataFrame({"H2": [1]}))
    big = pd.DataFrame({c: [1.0] * (MAX_BATCH_ROWS + 1) for c in ["H2", "CH4", "C2H6", "C2H4", "C2H2"]})
    with pytest.raises(ValueError):
        diagnose_table(big)

# ---- tests that need the training data (skipped without it) ----
def test_forest_agrees_on_a_discharge_case():
    out = diag().diagnose(h2=500, ch4=100, c2h6=20, c2h4=200, c2h2=600)
    assert out["rf_top"] in ("D1", "D2") and out["rf_top"] in out["rf_set"]

def test_out_of_range_warning():
    out = diag().diagnose(1e9, 1, 1, 1, 1)
    assert any("training data" in w for w in out["warnings"])

def test_batch_table_with_forest():
    df = pd.DataFrame({"H2": [500, 50], "CH4": [100, 40], "C2H6": [20, 10], "C2H4": [200, 90], "C2H2": [600, 1]})
    res = diagnose_table(df, diag())
    assert res.rf_top.notna().all()
