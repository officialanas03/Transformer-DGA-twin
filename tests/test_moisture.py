import numpy as np
import pytest
from twin import moisture as mo
from twin.dingwang import k_dp, moisture_limit_pct

def test_vapour_pressure_near_textbook():
    # my own sanity check against the textbook boiling point, not a paper value
    assert mo.vapour_pressure_mpa(100.0) == pytest.approx(0.101325, rel=0.01)
    assert mo.vapour_pressure_mpa(20.0) == pytest.approx(0.00234, rel=0.1)

def test_solubility_and_equilibrium_trends():
    assert mo.oil_saturation_ppm(80) > mo.oil_saturation_ppm(20)
    assert mo.equilibrium_paper_moisture_pct(60, 20) > mo.equilibrium_paper_moisture_pct(60, 10)
    assert mo.equilibrium_paper_moisture_pct(20, 10) > mo.equilibrium_paper_moisture_pct(80, 10)   # cool paper holds more
    assert mo.equilibrium_paper_moisture_pct(60, 0) == 0.0

def test_negative_oil_moisture_rejected():
    with pytest.raises(ValueError):
        mo.equilibrium_paper_moisture_pct(60, -1)

@pytest.mark.parametrize("t_frac", [0.02, 0.1, 0.5, 2.0])
def test_series_matches_finite_differences(t_frac):
    l, d = 3e-3, 1e-12
    t = t_frac * l**2 / d
    x = np.linspace(0, l, 7)
    a = mo.diffusion_series(x, t, d, l, c_s=1.0, c_i=0.2)
    b = mo.diffusion_fd(x, t, d, l, c_s=1.0, c_i=0.2)
    assert np.max(np.abs(a - b)) < 5e-3

def test_series_boundaries_and_limits():
    l, d = 3e-3, 1e-12
    assert mo.diffusion_series(0.0, 1e5, d, l, 1.0, 0.2) == pytest.approx(1.0, abs=1e-9)   # surface value
    assert mo.diffusion_series(np.array([0.0, l]), 0.0, d, l, 1.0, 0.2)[1] == pytest.approx(0.2)
    long_t = 50 * l**2 / d
    assert mo.diffusion_series(l, long_t, d, l, 1.0, 0.2) == pytest.approx(1.0, abs=1e-6)
    h = 1e-7
    grad = (mo.diffusion_series(l, 0.3 * l**2 / d, d, l, 1.0) - mo.diffusion_series(l - h, 0.3 * l**2 / d, d, l, 1.0)) / h
    assert abs(grad) < 1e-3 * (1.0 / l)                                                     # no flux at x = l

def test_series_equals_printed_eq4_for_zero_initial():
    l, d, t, c0, x = 3e-3, 1e-12, 2e3, 0.8, 1e-3
    n = np.arange(300)
    printed = c0 - 4 * c0 / np.pi * np.sum((-1.0) ** n / (2 * n + 1) * np.exp(-t * d * ((0.5 + n) * np.pi) ** 2 / l**2)
                                           * np.cos((0.5 + n) * np.pi * (l - x) / l))
    assert mo.diffusion_series(x, t, d, l, c0, 0.0) == pytest.approx(printed)

def test_series_input_checks():
    with pytest.raises(ValueError):
        mo.diffusion_series(0.0, 1.0, 1e-12, 0.0, 1.0)
    with pytest.raises(ValueError):
        mo.diffusion_series(0.0, -1.0, 1e-12, 1e-3, 1.0)

def test_diffusion_coefficient_rises_with_temperature_and_moisture():
    assert mo.diffusion_coefficient(80, 1.0) > mo.diffusion_coefficient(40, 1.0)
    assert mo.diffusion_coefficient(60, 3.0) > mo.diffusion_coefficient(60, 1.0)
    assert mo.diffusion_coefficient(24.85, 0.0) == pytest.approx(mo.D0, rel=1e-6)    # T = T0, C = 0

def test_eq13_values_and_conflict_with_eq11():
    assert mo.total_moisture_pct(3) == pytest.approx(0.7)
    assert mo.total_moisture_pct(48) == pytest.approx(2.64, abs=0.01)
    # Eq. 13 gives 0.70 % at year 3, above the 0.615 % where Eq. 11's k_DP reaches zero
    assert mo.total_moisture_pct(3) > moisture_limit_pct()
    with pytest.raises(ValueError):
        k_dp(float(mo.total_moisture_pct(3)))
