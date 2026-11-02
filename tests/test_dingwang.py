import numpy as np
import pytest
from twin.dingwang import (DingWangAgeing, w_star, k_dp, alpha_T, alpha_C, moisture_limit_pct, DingWangParams)

TABLE2_I = {1: 1167.52, 10: 920.92, 25: 648.27, 40: 486.52, 48: 429.17}   # Luo et al. 2023, Table 2, row I

def test_levelling_off_dp_matches_paper():
    m = DingWangAgeing(c_pct=0.18)
    assert m.dp_levelling_off == pytest.approx(252.15, abs=0.1)

def test_translational_factors():
    assert float(alpha_T(90.0)) == pytest.approx(1.0, abs=1e-3)           # 363.15 K is the reference
    assert float(alpha_T(61.29)) < 0.1 < float(alpha_T(90.0))
    assert float(alpha_C(5.0, 90.0)) == pytest.approx(6.0, abs=0.3)       # paper's Fig. 4 reaches about 6 at 5 %
    assert float(alpha_C(0.0, 70.0)) == 1.0

def test_k_dp_refuses_moisture_where_the_printed_formula_fails():
    assert k_dp(0.18) > 0
    assert 0.55 < moisture_limit_pct() < 0.7
    with pytest.raises(ValueError):
        k_dp(2.0)

def test_table2_row_one_is_reproduced_after_calibrating_the_rate_at_year_one():
    # FIT CHECK, not an independent prediction: one rate calibrated on year 1, years 10-48 then follow.
    m = DingWangAgeing(c_pct=0.18)
    per_year = 365.0
    m.advance(np.array([61.29]), per_year)
    m.calibrate_to_measurement(TABLE2_I[1])
    assert m.dp() == pytest.approx(TABLE2_I[1], abs=1e-6)
    for yrs, dp in TABLE2_I.items():
        mm = DingWangAgeing(c_pct=0.18); mm.k_scale = m.k_scale
        mm.advance(np.array([61.29]), per_year * yrs)
        assert mm.dp() == pytest.approx(dp, abs=1.0)

def test_hotter_ages_faster_and_never_below_levelling_off():
    cool, hot = DingWangAgeing(), DingWangAgeing()
    cool.advance(np.array([60.0]), 3650.0); hot.advance(np.array([90.0]), 3650.0)
    assert hot.dp() < cool.dp() < 1200.0
    hot.advance(np.array([120.0]), 1e7)
    assert hot.dp() == pytest.approx(hot.dp_levelling_off, abs=0.5)

def test_moisture_factor_accelerates():
    dry, wet = DingWangAgeing(c_pct=0.18), DingWangAgeing(c_pct=0.18, use_moisture_factor=True)
    dry.advance(np.array([70.0]), 3650.0); wet.advance(np.array([70.0]), 3650.0)
    assert wet.t_eff > dry.t_eff

def test_calibration_guards():
    m = DingWangAgeing()
    with pytest.raises(ValueError):
        m.calibrate_to_measurement(800)                                    # nothing accumulated yet
    m.advance(np.array([70.0]), 100.0)
    with pytest.raises(ValueError):
        m.calibrate_to_measurement(m.dp_levelling_off - 1)                 # below the levelling-off DP
