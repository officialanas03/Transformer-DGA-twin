import numpy as np
import pytest
from twin.thermal import ThermalParams, steady_state, simulate
from twin.ageing import (relative_ageing_rate, KineticAgeing, calibrate_A_from_life,
                         arrhenius_factor, remaining_life_hours, loss_of_life_hours)

P = ThermalParams()

def test_steady_state_rated_load():
    o, h = steady_state(1.0, 20.0, P)
    assert o == pytest.approx(20 + P.dtheta_or) and h == pytest.approx(o + P.dtheta_hr)

def test_simulation_converges_to_steady_state():
    df = simulate(np.full(3000, 1.2), np.full(3000, 25.0), dt_min=10.0, p=P)
    o, h = steady_state(1.2, 25.0, P)
    assert df.theta_o.iloc[-1] == pytest.approx(o, abs=1e-3) and df.theta_h.iloc[-1] == pytest.approx(h, abs=1e-3)

def test_oil_step_response_time_constant():
    # start at steady state for K=0.5, step to K=1: after one tau, 63.2% of the way
    o0, _ = steady_state(0.5, 20.0, P); o1, _ = steady_state(1.0, 20.0, P)
    n = int(P.tau_o_min)                      # one tau in 1-minute steps
    df = simulate(np.full(n, 1.0), np.full(n, 20.0), dt_min=1.0, p=P, theta_o0=o0)
    frac = (df.theta_o.iloc[-1] - o0) / (o1 - o0)
    assert frac == pytest.approx(1 - np.exp(-1), abs=0.01)

def test_zero_load_goes_to_ambient():
    df = simulate(np.zeros(4000), np.full(4000, 15.0), dt_min=10.0, p=ThermalParams(dtheta_hr=0.0), theta_o0=60.0, theta_h0=60.0)
    assert df.theta_o.iloc[-1] == pytest.approx(15 + P.dtheta_or * (1 / (1 + P.R)) ** P.n, abs=1e-3)

def test_input_validation():
    with pytest.raises(ValueError):
        simulate([1, 2], [20], 10.0)
    with pytest.raises(ValueError):
        simulate([-1], [20], 10.0)

def test_relative_ageing_rates():
    assert relative_ageing_rate(98, upgraded=False) == pytest.approx(1.0)
    assert relative_ageing_rate(104, upgraded=False) == pytest.approx(2.0)
    assert relative_ageing_rate(110, upgraded=True) == pytest.approx(1.0)
    assert relative_ageing_rate(98, upgraded=True) == pytest.approx(np.exp(15000/383 - 15000/371))

def test_loss_of_life_hours():
    assert loss_of_life_hours(np.full(24, 110.0), 1.0, upgraded=True) == pytest.approx(24.0)

def test_kinetic_calibration_hits_the_assumed_life():
    A = calibrate_A_from_life(110.0, 180000.0)
    m = KineticAgeing(dp0=1000.0, A=A)
    m.advance(np.full(1800, 110.0), 100.0)                  # 180,000 h in 100 h steps
    assert m.dp() == pytest.approx(200.0, rel=1e-6)

def test_hotter_ages_faster():
    A = calibrate_A_from_life(110.0, 180000.0)
    cool, hot = KineticAgeing(A=A), KineticAgeing(A=A)
    cool.advance(np.full(1000, 90.0), 10.0); hot.advance(np.full(1000, 120.0), 10.0)
    assert hot.dp() < cool.dp() < 1000.0

def test_calibration_to_measurement_and_remaining_life():
    m = KineticAgeing(dp0=1000.0); m.advance(np.full(876, 95.0), 10.0)   # 8760 h
    m.calibrate_to_measurement(600.0)
    assert m.dp() == pytest.approx(600.0, rel=1e-9)
    rate = float(arrhenius_factor(95.0))
    left = remaining_life_hours(m, rate)
    m.advance(np.full(int(left), 95.0), 1.0)
    assert m.dp() == pytest.approx(200.0, rel=1e-3)

def test_errors():
    with pytest.raises(ValueError):
        KineticAgeing().dp()
    m = KineticAgeing(); 
    with pytest.raises(ValueError):
        m.calibrate_to_measurement(500)
