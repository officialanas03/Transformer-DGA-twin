import math
import pytest
from twin import paper_numbers as pn
from twin import dingwang as dw
from twin import moisture as mo

ROWS = pn.load()

def v(i):
    return pn.value(i, ROWS)

def test_file_is_well_formed():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids))
    for r in ROWS:
        assert r["pdf_page"].isdigit() and 1 <= int(r["pdf_page"]) <= 12
        assert r["location"]

def test_missing_values_raise():
    with pytest.raises(ValueError):
        pn.value("c14", ROWS)        # pressboard thickness is not given

def test_assumptions_are_flagged():
    kinds = {r["id"]: r["kind"] for r in ROWS}
    assert kinds["c09"] == "assumption"          # P0
    assert kinds["c14"] == "not_given"           # thickness l
    for i in ("c23", "c24", "c25", "c26", "c60"):
        assert kinds[i] == "figure_reading_approx"

def test_dingwang_constants_match_file():
    assert dw.DingWangParams().dp0 == v("c15")
    assert dw.DingWangParams().Ea == pytest.approx(v("c16") * 1e3)
    assert dw.DingWangParams().T_ref == v("c17")
    assert dw.DingWangParams().gamma == v("c18")
    assert 8.314 * v("c17") == pytest.approx(v("c22"), abs=0.01)
    # Eq. 11 coefficients: evaluate the code at two points and compare with the file's polynomials
    for c_pct in (0.1, 0.4):
        c = c_pct / 100
        assert dw.w_star(c_pct) == pytest.approx(v("c27") * c**2 + v("c28") * c + v("c29"))
        assert dw.k_dp(c_pct) == pytest.approx(v("c30") * c**2 + v("c31") * c + v("c32"))

def test_levelling_off_close_to_printed():
    assert dw.DingWangAgeing(v("c34")).dp_levelling_off == pytest.approx(v("c33"), abs=0.05)

def test_moisture_constants_match_file():
    assert mo.D0 == v("c10") and mo.K0 == v("c11") and mo.E_D == v("c12") and mo.T0_K == v("c13")
    assert mo.P0_MPA == v("c09")
    assert mo.total_moisture_pct(v("c58")) == pytest.approx(v("c56"))
    assert mo.total_moisture_pct(10) == pytest.approx(v("c56") * math.exp(v("c57") * 7))
    # Eq. 1 constants, checked through two evaluated values
    assert mo.oil_saturation_ppm(50.0) == pytest.approx(10 ** (v("c01") - v("c02") / 323.15))
    assert mo.vapour_pressure_mpa(50.0) == pytest.approx(math.exp(v("c03") - v("c04") / (323.15 - v("c05"))))

def test_table2_rows_in_file_are_ordered_decreasing():
    for first in (67, 72, 77):
        vals = [v(f"c{n}") for n in range(first, first + 5)]
        assert vals == sorted(vals, reverse=True)
    assert v("c81") == v("c33")          # year-48 case III equals the levelling-off DP printed in Sec. 3.3
