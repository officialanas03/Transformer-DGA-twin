import numpy as np
import pytest
from dga.ml.uncertainty import expected_calibration_error, conformal_threshold, prediction_sets, coverage_and_size
from dga.ageing.paper import dp_chendong, dp_depablo, furan_condition, co2_co_ratio, co2_co_band

def test_ece_zero_for_perfectly_calibrated_certain_model():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(probs, [0, 1, 0]) == 0.0

def test_ece_detects_overconfidence():
    probs = np.tile([0.95, 0.05], (100, 1))
    y = np.array([0] * 60 + [1] * 40)          # 60% accurate but 95% confident
    assert expected_calibration_error(probs, y) == pytest.approx(0.35, abs=1e-9)

def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    def draw(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(size=(n, 3)); logits[np.arange(n), y] += 1.5
        p = np.exp(logits); return p / p.sum(1, keepdims=True), y
    pc, yc = draw(2000); pt, yt = draw(5000)
    res = coverage_and_size(prediction_sets(pt, conformal_threshold(pc, yc, 0.1)), yt)
    assert 0.88 <= res["coverage"] <= 0.93

def test_chendong_reference_points():
    assert dp_chendong(1.0) == pytest.approx(1.51 / 0.0035)     # about 431
    assert dp_chendong(0.1) > dp_chendong(1.0) > dp_chendong(10.0)
    with pytest.raises(ValueError):
        dp_chendong(0)

def test_depablo_and_bands():
    assert dp_depablo(0) == pytest.approx(1850 / 2.3)
    assert furan_condition(0.05) == "good" and furan_condition(5) == "high ageing, risk of fault"
    assert furan_condition(20) == "high risk of fault"

def test_co2_co():
    assert co2_co_ratio(3000, 500) == 6.0
    assert co2_co_band(6.0) == "3 to 10 (normal band)" and co2_co_band(2.0) == "below 3"
    assert np.isinf(co2_co_ratio(100, 0)) and np.isnan(co2_co_ratio(0, 0))
