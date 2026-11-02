from pathlib import Path
import pytest
from dga.pipeline import Diagnoser

DATA = Path(__file__).resolve().parents[1] / "data" / "dga_augmented.csv"
_D = None
def diag():
    global _D
    if _D is None:
        _D = Diagnoser(DATA)
    return _D

def test_known_arcing_case_is_discharge_in_every_method():
    out = diag().diagnose(h2=500, ch4=100, c2h6=20, c2h4=200, c2h2=600)
    assert out["duval"] in ("D1", "D2")
    assert out["rf_top"] in ("D1", "D2")
    assert set(out["rf_set"]) >= {out["rf_top"]}

def test_output_fields_and_ageing():
    out = diag().diagnose(100, 50, 20, 30, 1, co=500, co2=3000, fal_ppm=1.0)
    assert out["co2_co"]["ratio"] == 6.0 and out["co2_co"]["band"].startswith("3 to 10")
    assert out["furan"]["dp_chendong"] == pytest.approx(1.51 / 0.0035)

def test_all_fault_gases_zero_is_handled():
    out = diag().diagnose(10, 0, 0, 0, 0)
    assert out["duval"] is None and out["rf_set"] == [] and out["warnings"]

def test_negative_rejected():
    with pytest.raises(ValueError):
        diag().diagnose(-1, 0, 0, 0, 0)

def test_out_of_range_warning():
    out = diag().diagnose(1e9, 1, 1, 1, 1)
    assert any("training data" in w for w in out["warnings"])
